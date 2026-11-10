"""Read-only, exact-row summaries of a frozen critic diagnostic."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import iter_unique_replay_chunks, load_manifest, replay_chunks
from .io import sha256_file


GROUPS = ("human_pre_success", "human_post_success", "failed_parent",
          "branch_handoff_exact", "parent_chunk_covering_handoff")
VALUES = ("q1", "q2", "min_q", "v", "advantage")
REWARDS = ("reward", "bootstrap_mask", "bellman_target", "td_error", "observed_return")


def statistics(values):
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return dict.fromkeys(("mean", "p05", "p50", "p95", "max")) | {"count": 0}
    return {"count": len(data), "mean": float(data.mean()), "max": float(data.max()),
            **dict(zip(("p05", "p50", "p95"), map(float, np.quantile(data, [.05, .5, .95]))))}


def summarize(rows, training, probe, diagnostic, *, include_post_success=True,
              probe_include_post_success=True, beta=3., maximum=20.):
    if (not np.isfinite(beta) or beta < 0 or not np.isfinite(maximum) or maximum <= 0):
        raise ValueError("Invalid IQL weight settings")
    if (diagnostic.get("kind") != "iql_critic_diagnostic"
            or diagnostic.get("dataset_sha256") != training["dataset_sha256"]
            or diagnostic.get("probe_dataset_sha256") != probe["dataset_sha256"]):
        raise ValueError("Diagnostic dataset identity mismatch")
    expected = {}
    for label, manifest, split, post in (("train", training, "train", include_post_success),
            ("validation", training, "validation", include_post_success),
            ("rollout", probe, "train", probe_include_post_success)):
        episodes = manifest["episodes"]
        if len({ep["run_id"] for ep in episodes}) != len(episodes):
            raise ValueError("Duplicate manifest run ID")
        for episode, index in iter_unique_replay_chunks(episodes, split=split, include_post_success=post):
            chunk = replay_chunks(episode)[index]
            key = (label, episode["run_id"], int(chunk["start"]), int(chunk["end"]))
            if key in expected:
                raise ValueError("Duplicate manifest chunk")
            expected[key] = episode, chunk
    selected = {}
    for source in rows:
        try:
            key = (source["split"], source["run_id"], int(source["start"]), int(source["end"]))
            length = int(source["chunk_length"])
            values = {name: float(source[name]) for name in VALUES}
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Malformed diagnostic row") from error
        if key not in expected or key in selected:
            raise ValueError(f"Unexpected or duplicate diagnostic row: {key}")
        episode, chunk = expected[key]
        if length != key[3] - key[2] or source["action_source"] != chunk["action_source"]:
            raise ValueError(f"Chunk metadata mismatch: {key}")
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"Non-finite critic value: {key}")
        if (not np.isclose(values["min_q"], min(values["q1"], values["q2"]), rtol=1e-5, atol=1e-5)
                or not np.isclose(values["advantage"], values["min_q"] - values["v"], rtol=1e-5, atol=1e-5)):
            raise ValueError(f"Inconsistent Q/V/advantage: {key}")
        for name in REWARDS:
            value = source.get(name)
            if key[0] == "rollout":
                if value not in (None, ""):
                    raise ValueError("Rollout must not have reward/TD labels")
            elif value in (None, "") or not np.isfinite(float(value)):
                raise ValueError(f"Missing/non-finite diagnostic label {name}: {key}")
        terminal = episode.get("terminal_step")
        groups = []
        if chunk["action_source"] == "human" and terminal is not None:
            groups.append("human_post_success" if key[2] >= terminal + 1 else "human_pre_success")
        if episode.get("kind") != "branch" and terminal is None:
            groups.append("failed_parent")
        # Clamp before multiplication to avoid overflow for extreme finite A.
        log_weight = min(values["advantage"], np.log(maximum) / beta) * beta if beta else 0.
        weight = float(np.exp(min(log_weight, np.log(maximum))))
        selected[key] = {**source, "start": key[2], "end": key[3], "chunk_length": length,
                         **values, "weight": weight, "groups": groups,
                         "copied_prefix": bool(chunk.get("copied_prefix", False))}
    if set(selected) != set(expected):
        raise ValueError(f"Missing diagnostic rows: {len(set(expected) - set(selected))}")
    counts = {split: sum(key[0] == split for key in selected) for split in ("train", "validation", "rollout")}
    if counts != diagnostic.get("counts"):
        raise ValueError("Diagnostic summary row counts mismatch")
    pairs = []
    for manifest, rollout in ((training, False), (probe, True)):
        episodes = {ep["run_id"]: ep for ep in manifest["episodes"]}
        for ep in manifest["episodes"]:
            if ep.get("kind") != "branch" or ep.get("resume_step") is None:
                continue
            step = int(ep["resume_step"])
            split = "rollout" if rollout else ep["split"]
            exact = [key for key in selected if key[:2] == (split, ep["run_id"])
                     and key[2] == step and selected[key]["action_source"] == "human"]
            parent_id = ep.get("parent_run_id") or ep.get("root_run_id")
            parent = episodes.get(parent_id)
            parent_split = "rollout" if rollout else parent["split"] if parent else None
            covering = [key for key in selected if key[:2] == (parent_split, parent_id)
                        and key[2] <= step < key[3]]
            if len(exact) > 1 or len(covering) > 1:
                raise ValueError("Ambiguous handoff chunk")
            for keys, name in ((exact, "branch_handoff_exact"), (covering, "parent_chunk_covering_handoff")):
                if keys and name not in selected[keys[0]]["groups"]:
                    selected[keys[0]]["groups"].append(name)
            pairs.append({"branch_run_id": ep["run_id"], "parent_run_id": parent_id,
                "handoff_step": step, "branch_row": list(exact[0]) if exact else None,
                "parent_row": list(covering[0]) if covering else None,
                "parent_offset": step - covering[0][2] if covering else None,
                "same_state_verified": False})
    ordered = [selected[key] for key in sorted(selected)]
    summaries = {split: {group: statistics([row["weight"] for row in ordered
                    if row["split"] == split and (group == "all" or group in row["groups"])])
                    for group in ("all", *GROUPS)} for split in counts}
    return {"schema_version": 1, "kind": "critic_weight_summary", "beta": beta, "cap": maximum,
        "counts": counts, "groups_by_split": summaries, "handoff_pairs": pairs, "rows": ordered,
        "caveats": ["Post-fit unique chunks, not W4 training minibatches or warmup weights.",
                    "Parent covering chunks are not verified same-state comparisons, even when offset is zero.",
                    "Rows are counted once per group; handoff_pairs preserves repeated parent references.",
                    "Train, validation, and rollout are separate populations; no shared expectile is fitted."]}


def summarize_files(training_config, probe_config, result: Path, output: Path):
    result, output = Path(result), Path(output)
    diagnostic_path, rows_path = result / "summary.json", result / "qv.csv"
    diagnostic = json.loads(diagnostic_path.read_text())
    if (training_config.digest != diagnostic["training_config_sha256"]
            or probe_config.digest != diagnostic["probe_config_sha256"]):
        raise ValueError("Diagnostic configuration identity mismatch")
    paths = [diagnostic_path, rows_path, result / "critic.pt", training_config.path, probe_config.path,
             *(Path(c.section("paths")["work_dir"]) / "dataset_manifest.json"
               for c in (training_config, probe_config))]
    before = {str(path): sha256_file(path) for path in paths}
    with rows_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    payload = summarize(rows, load_manifest(training_config), load_manifest(probe_config), diagnostic,
        include_post_success=training_config.section("data").get("include_post_success", True),
        probe_include_post_success=probe_config.section("data").get("include_post_success", True),
        beta=training_config.section("iql")["beta"], maximum=training_config.section("iql")["max_advantage_weight"])
    payload["input_sha256"] = before
    payload["diagnostic_budget"] = {key: diagnostic[key] for key in ("train_steps", "batch_size", "seed")}
    payload["reward_config"] = diagnostic["reward_config"]
    if before != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("Diagnostic inputs changed during summarization")
    if any(output.resolve() == path.resolve() for path in paths):
        raise ValueError("Output must not overwrite an input")
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload
