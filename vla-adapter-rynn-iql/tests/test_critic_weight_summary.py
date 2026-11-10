import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vla_rynn_iql.critic_weight_summary import summarize, summarize_files, beta_comparison, VALUES, REWARDS
from vla_rynn_iql.data import iter_unique_replay_chunks, prepare_dataset, replay_chunks


def episode(name, split="train", *, branch=False, resume=None, terminal=None, ends=(8, 16, 24)):
    starts = (0, *ends[:-1])
    return {"run_id": name, "split": split, "kind": "branch" if branch else "original",
        "root_run_id": "root" if branch else name, "parent_run_id": "root" if branch else None,
        "resume_step": resume, "terminal_step": terminal, "recorded_action_count": ends[-1],
        "chunks": [{"start": a, "end": b, "length": b-a,
            "action_source": "human" if branch and a >= resume else "policy",
            "copied_prefix": branch and b <= resume} for a, b in zip(starts, ends)]}


def rows_for(training, probe):
    rows = []
    for label, manifest, split in (("train", training, "train"), ("validation", training, "validation"),
                                    ("rollout", probe, "train")):
        for ep, index in iter_unique_replay_chunks(manifest["episodes"], split=split):
            chunk = replay_chunks(ep)[index]
            rows.append({"split": label, "run_id": ep["run_id"], "start": chunk["start"],
                "end": chunk["end"], "chunk_length": chunk["length"], "action_source": chunk["action_source"],
                "q1": .2, "q2": .3, "min_q": .2, "v": 0., "advantage": .2,
                **dict.fromkeys(REWARDS, None if label == "rollout" else 0.)})
    summary = {"kind": "iql_critic_diagnostic", "dataset_sha256": training["dataset_sha256"],
        "probe_dataset_sha256": probe["dataset_sha256"],
        "counts": {label: sum(row["split"] == label for row in rows) for label in ("train", "validation", "rollout")}}
    return rows, summary


@pytest.fixture
def inputs():
    training = {"dataset_sha256": "train-sha", "episodes": [episode("root"),
        episode("branch", branch=True, resume=10, terminal=15, ends=(8, 10, 16, 24)),
        episode("another", branch=True, resume=10, terminal=15, ends=(8, 10, 16, 24)),
        episode("validation", "validation"),
        episode("failed-human", branch=True, resume=8)]}
    probe = {"dataset_sha256": "probe-sha", "episodes": [episode("probe")]}
    rows, diagnostic = rows_for(training, probe)
    return rows, training, probe, diagnostic


def test_exact_groups_offsets_no_duplicate_parent_and_split_isolation(inputs):
    result = summarize(*inputs)
    train = result["groups_by_split"]["train"]
    assert train["human_pre_success"]["count"] == 2
    assert train["human_post_success"]["count"] == 2
    assert train["failed_parent"]["count"] == 3
    assert train["branch_handoff_exact"]["count"] == 3
    assert train["parent_chunk_covering_handoff"]["count"] == 1
    assert [p["parent_offset"] for p in result["handoff_pairs"]] == [2, 2, 0]
    assert all(not p["same_state_verified"] for p in result["handoff_pairs"])
    assert result["groups_by_split"]["validation"]["failed_parent"]["count"] == 3
    assert result["groups_by_split"]["rollout"]["failed_parent"]["count"] == 3
    for row in result["rows"]:
        assert row["weight"] == pytest.approx(np.exp(.6))
        if row["run_id"] == "failed-human":
            assert "human_pre_success" not in row["groups"]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nonfinite", "advantage", "min_q", "source", "boundary", "count", "identity", "rollout_reward"])
def test_rejects_corrupt_or_incomplete_diagnostic_rows(inputs, failure):
    rows, training, probe, diagnostic = copy.deepcopy(inputs)
    if failure == "duplicate": rows.append(rows[0])
    elif failure == "missing": rows.pop()
    elif failure == "nonfinite": rows[0]["v"] = float("nan")
    elif failure == "advantage": rows[0]["advantage"] = .9
    elif failure == "min_q": rows[0]["min_q"] = .9
    elif failure == "source": rows[0]["action_source"] = "human"
    elif failure == "boundary": rows[0]["end"] += 1
    elif failure == "count": diagnostic["counts"]["train"] += 1
    elif failure == "identity": diagnostic["dataset_sha256"] = "changed"
    elif failure == "rollout_reward": rows[-1]["reward"] = 1
    with pytest.raises(ValueError):
        summarize(rows, training, probe, diagnostic)


def test_extreme_weights_clamped_without_overflow(inputs):
    rows, training, probe, diagnostic = inputs
    for index, row in enumerate(rows):
        value = 1e300 if index % 2 else -1e300
        row.update(q1=value, q2=value, min_q=value, advantage=value)
    result = summarize(rows, training, probe, diagnostic)
    assert all(np.isfinite(row["weight"]) and 0 <= row["weight"] <= 20 for row in result["rows"])


def test_real_manifest_selection_and_exclusive_readonly_output(configured, tmp_path):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    rows, summary = rows_for(manifest, manifest)
    summary.update(training_config_sha256=configured.digest, probe_config_sha256=configured.digest,
                   train_steps=10000, batch_size=64, seed=7, reward_config={"gamma": .99})
    directory = tmp_path / "result"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "critic.pt").write_bytes(b"synthetic identity artifact; not loaded")
    with (directory / "qv.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protected = {p: p.read_bytes() for p in directory.iterdir()}
    target = tmp_path / "summary.json"
    payload = summarize_files(configured, configured, directory, target)
    assert payload["counts"] == summary["counts"]
    assert payload["diagnostic_budget"] == {"train_steps": 10000, "batch_size": 64, "seed": 7}
    assert set(payload["beta_comparison"]) == {"3.0", "0.3"}
    for key in ("3.0", "0.3"):
        assert payload["beta_comparison"][key]["groups_by_split"]["train"]["all"]["count"] == summary["counts"]["train"]
    assert len(payload["input_sha256"]) == 5
    assert all(p.read_bytes() == data for p, data in protected.items())
    with pytest.raises(FileExistsError):
        summarize_files(configured, configured, directory, target)
    summary["training_config_sha256"] = "wrong"
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="configuration identity"):
        summarize_files(configured, configured, directory, tmp_path / "other.json")


def test_two_beta_same_rows_cap_fraction_and_quantiles(inputs):
    validated = summarize(*inputs)["rows"]
    for index, row in enumerate(validated):
        row["advantage"] = [-1., 0., 2., 20.][index % 4]
    protected = copy.deepcopy(validated)
    result = beta_comparison(validated)
    assert validated == protected
    for beta in (3., .3):
        actual = result[str(beta)]["groups_by_split"]["train"]["all"]
        a = np.asarray([row["advantage"] for row in validated if row["split"] == "train"])
        weights = np.minimum(np.exp(beta * a), 20.)
        assert actual["mean"] == pytest.approx(weights.mean())
        assert actual["p05"] == pytest.approx(np.quantile(weights, .05))
        assert actual["p50"] == pytest.approx(np.quantile(weights, .5))
        assert actual["p95"] == pytest.approx(np.quantile(weights, .95))
        assert actual["cap_fraction"] == pytest.approx(np.mean(weights == 20))
    with pytest.raises(ValueError, match="cap=20"):
        beta_comparison(validated, maximum=10)
