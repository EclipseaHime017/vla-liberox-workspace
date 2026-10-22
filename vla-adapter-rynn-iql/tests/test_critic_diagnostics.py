from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import _episode
from vla_rynn_iql import critic_diagnostics as diagnostics
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.data import load_manifest, prepare_dataset
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.models import model_config, replay_stats
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import load_reward_index


def _stats(size):
    return {"q01": [-1.] * size, "q99": [1.] * size}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def diagnostic_inputs(configured, tmp_path):
    root = tmp_path / "normalization"
    root.mkdir()
    (root / "dataset_statistics.json").write_text(json.dumps({"libero_object_no_noops": {
        "action": _stats(7), "proprio": _stats(8)}}))
    configured.raw["model"].update(base_checkpoint=str(root), stats_key="libero_object", base_revision=None)
    configured.raw["reward"].update(source="sparse", gamma=.8, accumulate_primitive_steps=False)
    configured.raw["iql"]["critic_image_size"] = 16
    prepare_dataset(configured)
    index = load_reward_index(configured)
    path = Path(index["episodes"][0]["reward_path"]).parent / "reward_manifest.json"
    configured.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="test-version")
    source = tmp_path / "probe-data"
    _episode(source, "probe", action_count=19)
    run_path = next(source.rglob("run.json"))
    run = json.loads(run_path.read_text())
    run["open_loop_steps"] = 10
    run_path.write_text(json.dumps(run))
    raw = copy.deepcopy(configured.raw)
    raw["data"]["allow_no_success"] = True
    raw["paths"].update(dataset_sources=[str(source)], work_dir=str(tmp_path / "probe-work"))
    raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None)
    probe = LoadedConfig(tmp_path / "probe.yaml", raw)
    prepare_dataset(probe)
    return configured, probe


def test_critic_fit_is_actor_free_preserves_inputs_and_produces_real_plots(diagnostic_inputs, monkeypatch, tmp_path):
    training, probe = diagnostic_inputs
    from vla_rynn_iql import models, training as actor_training, rewards

    def forbidden(*args, **kwargs):
        raise AssertionError("Actor loading, training, reward generation or export is forbidden")

    monkeypatch.setattr(models, "model_backend", forbidden)
    monkeypatch.setattr(actor_training, "train", forbidden)
    monkeypatch.setattr(rewards, "materialize_reward_manifest", forbidden)
    before = {str(path): sha256_file(path) for path in Path(training.section("paths")["work_dir"]).rglob("*") if path.is_file()}
    for config in (training, probe):
        for episode in load_manifest(config)["episodes"]:
            for key in ("trajectory_path", "observations_path"):
                before[episode[key]] = sha256_file(Path(episode[key]))
    progress = []
    output = tmp_path / "result"
    summary = diagnostics.run_critic_diagnostic(training, probe, output,
        train_steps=2, batch_size=3, seed=12, device="cpu", progress=progress.append)
    assert summary["actor_loaded_for_training"] is False
    assert summary["actor_updated"] is False
    assert summary["reward_generated"] is False
    assert summary["critic_horizon"] == 10
    assert [item["train_step"] for item in progress] == [1, 2]
    assert summary["counts"]["train"] > 0
    assert summary["counts"]["rollout"] == 2
    for name in summary["artifacts"]:
        assert (output / name).stat().st_size > 0
    for name in ("training.png", "qv.png"):
        with Image.open(output / name) as image:
            assert image.width > 100 and image.height > 100
            assert np.asarray(image).std() > 0
    rows = list(csv.DictReader((output / "qv.csv").open()))
    for row in rows:
        assert float(row["advantage"]) == pytest.approx(float(row["min_q"]) - float(row["v"]))
        if row["split"] == "rollout":
            assert all(row[key] == "" for key in diagnostics.REWARD_COLUMNS)
        else:
            assert all(row[key] != "" for key in diagnostics.REWARD_COLUMNS)
    with np.load(output / "qv.npz", allow_pickle=False) as arrays:
        probe_rows = arrays["split"] == "rollout"
        assert np.isnan(arrays["reward"][probe_rows]).all()
        assert np.isfinite(arrays["q1"]).all()
    checkpoint = torch.load(output / "critic.pt", weights_only=False)
    assert set(checkpoint["critic"]) == {"model", "q_optimizer", "value_optimizer"}
    assert all(key.startswith(("q1.", "q2.", "target_q1.", "target_q2.", "value."))
               for key in checkpoint["critic"]["model"])
    assert before == {path: sha256_file(Path(path)) for path in before}


def test_fixed_seed_batch_and_rng_are_reproducible(diagnostic_inputs, monkeypatch, tmp_path):
    training, probe = diagnostic_inputs
    monkeypatch.setattr(diagnostics, "_plots", lambda *args: None)
    initial = torch.random.get_rng_state().clone()
    for name in ("first", "same", "different"):
        diagnostics.run_critic_diagnostic(training, probe, tmp_path / name,
            train_steps=2, batch_size=2, seed=4 if name != "different" else 8, device="cpu")
    assert torch.equal(initial, torch.random.get_rng_state())
    assert (tmp_path / "first/qv.csv").read_bytes() == (tmp_path / "same/qv.csv").read_bytes()
    assert (tmp_path / "first/metrics.jsonl").read_bytes() == (tmp_path / "same/metrics.jsonl").read_bytes()
    assert (tmp_path / "first/qv.csv").read_bytes() != (tmp_path / "different/qv.csv").read_bytes()


def test_exact_batch_size_padding_and_cancel_at_each_boundary(diagnostic_inputs, monkeypatch, tmp_path):
    training, probe = diagnostic_inputs
    original = diagnostics.PixelIQL.update
    batches = []
    progress = []

    def update(agent, batch):
        batches.append(batch["actions"].shape)
        assert batch["action_mask"].shape == (4, 10)
        assert not batch["action_mask"][:, 8:].any()
        return original(agent, batch)

    monkeypatch.setattr(diagnostics.PixelIQL, "update", update)
    output = tmp_path / "canceled"
    with pytest.raises(InterruptedError, match="batch boundary"):
        diagnostics.run_critic_diagnostic(training, probe, output,
            train_steps=10, batch_size=4, seed=4, device="cpu", progress=progress.append,
            stop_requested=lambda: len(progress) == 1)
    assert batches == [torch.Size([4, 10, 7])]
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 1
    assert not (output / "summary.json").exists()
    assert not (output / "critic.pt").exists()


def test_padding_preserves_chunk_actions_and_masks():
    sample = {"actions": torch.arange(21.).reshape(3, 7), "action_mask": torch.tensor([True, True, False]),
              "proprio": torch.zeros(8)}
    batch = diagnostics._batch([sample], 5, torch.device("cpu"))
    assert batch["actions"].shape == (1, 5, 7)
    assert torch.equal(batch["actions"][0, :3], sample["actions"])
    assert not batch["actions"][0, 3:].any()
    assert batch["action_mask"].tolist() == [[True, True, False, False, False]]
    with pytest.raises(ValueError, match="cover every"):
        diagnostics._batch([sample], 2, torch.device("cpu"))


@pytest.mark.parametrize("cumulative", [False, True])
def test_finite_returns_use_the_same_chunk_discount_and_terminal_mask(diagnostic_inputs, cumulative):
    config, _ = diagnostic_inputs
    action, proprio = replay_stats(config.section("model"))
    replay = ReplayDataset(config, action, proprio, reward_index=diagnostics._existing_rewards(config))
    values = diagnostics._finite_returns(replay, .8, cumulative)
    for episode, _, annotation in replay.items:
        run_id = episode["run_id"]
        with np.load(annotation) as source:
            rewards = source["final_reward"]
        chunks = replay.chunks[run_id]
        for index, chunk in enumerate(chunks):
            total, factor = 0., 1.
            for successor, reward in zip(chunks[index:], rewards[index:]):
                total += factor * float(reward)
                factor *= .8 ** (int(successor["length"]) if cumulative else 1)
            assert values[(run_id, chunk["start"])] == pytest.approx(total)
    last = max((replay[index] for index in range(len(replay))), key=lambda item: item["start"])
    assert last["bootstrap_mask"].item() == 0


@pytest.mark.parametrize("change", ["missing", "discount", "corrupt", "source", "probe_source"])
def test_missing_changed_or_incompatible_inputs_fail_without_reward_recomputation(diagnostic_inputs, monkeypatch, tmp_path, change):
    training, probe = diagnostic_inputs
    from vla_rynn_iql import rewards
    monkeypatch.setattr(rewards, "_adapt_pinned_training_rewards", lambda *a: pytest.fail("must not recompute rewards"))
    if change == "missing":
        training.raw["reward"]["manifest_path"] = None
    elif change == "discount":
        training.raw["reward"]["gamma"] = .7
    elif change == "corrupt":
        Path(training.section("reward")["manifest_path"]).write_text("{}")
    else:
        config = probe if change == "probe_source" else training
        episode = load_manifest(config)["episodes"][0]
        with Path(episode["trajectory_path"]).open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        diagnostics.run_critic_diagnostic(training, probe, tmp_path / "out", train_steps=1, batch_size=1, seed=0, device="cpu")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
def test_replay_stats_loads_json_without_importing_actor_backend(tmp_path, monkeypatch, family):
    from vla_rynn_iql import models
    monkeypatch.setattr(models, "model_backend", lambda *a: pytest.fail("no model backend"))
    settings = model_config({"model": {"family": family}})
    settings.update(base_checkpoint=str(tmp_path), base_revision=None)
    if family == "pi05":
        path = tmp_path / "assets" / settings["stats_key"] / "norm_stats.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"norm_stats": {"actions": _stats(7), "state": _stats(8)}}))
    else:
        (tmp_path / "config.json").write_text(json.dumps({"norm_stats": {settings["stats_key"]: {
            "action": _stats(7), "proprio": _stats(8)}}}))
    action, proprio = replay_stats(settings)
    assert action["q01"] == [-1.] * 7
    assert proprio["q99"] == [1.] * 8
    assert action.get("codec") == ("openpi_quantile" if family == "pi05" else None)


@pytest.mark.parametrize("bad", [None, [0.], [float("nan")] * 7, [True] * 7])
def test_replay_stats_rejects_invalid_normalization(tmp_path, bad):
    settings = model_config({"model": {"base_checkpoint": str(tmp_path), "base_revision": None}})
    (tmp_path / "dataset_statistics.json").write_text(json.dumps({settings["stats_key"]: {
        "action": {"q01": bad, "q99": [1.] * 7}, "proprio": _stats(8)}}))
    with pytest.raises(ValueError, match="normalization"):
        replay_stats(settings)
