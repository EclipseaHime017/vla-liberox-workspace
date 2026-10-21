"""Portable main-style recordings retain their meaning in server replay."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from test_replay import _stats
from test_server_inputs import portable_export
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.data import load_manifest
from vla_rynn_iql.io import atomic_json, sha256_file
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import load_reward_index
from vla_rynn_iql.server_inputs import discover_tasks, open_run_source, prepare_global_inputs
from vla_rynn_iql.server_replay import CachedReplayDataset, build_replay_cache
from vla_rynn_iql.stage_rewards import build_stage_annotation


def _record_producers(config):
    root = Path(config.section("paths")["dataset_sources"][0])
    for path in root.rglob("run.json"):
        run = json.loads(path.read_text())
        branch = run["kind"] == "branch"
        segments = ([dict(start=0, end=5, origin_run_id="root", open_loop_steps=8),
                     dict(start=5, end=12, origin_run_id="branch", open_loop_steps=4),
                     dict(start=12, end=22, origin_run_id="branch", open_loop_steps=3)]
                    if branch else [dict(start=0, end=17, origin_run_id="root", open_loop_steps=8)])
        handoffs = [5, 12] if branch else []
        run.update(trajectory_segments=segments, handoff_steps=handoffs,
                   open_loop_steps=3 if branch else 8)
        atomic_json(path, run)
        trajectory = path.parent / "episodes/episode_000/trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as arrays:
            values = dict(arrays)
        if branch:
            values["action_source"] = np.asarray(["policy"] * 5 + ["human"] * 7 + ["policy_requery"] * 10)
        values["inference_query_step"] = np.asarray([0, 12, 15, 18, 21] if branch else [0, 7, 15])
        values["metadata_json"] = np.asarray(json.dumps({"task": run["task"],
            "trajectory_segments": segments, "handoff_steps": handoffs,
            "open_loop_steps": run["open_loop_steps"]}))
        np.savez_compressed(trajectory, **values)
        image = np.arange((len(values["env_action"]) + 1) * 16 * 16 * 3, dtype=np.uint8).reshape(-1, 16, 16, 3)
        np.savez_compressed(trajectory.with_name("trajectory_observations.npz"),
                            agentview_image=image, wrist_image=image + 1)


def _add_reward_source(root, config, source):
    if source not in {"stage", "rynnvalue"}:
        return
    config.raw["reward"].update(source=source, rynnvalue=source == "rynnvalue")
    index = load_reward_index(config) if source == "rynnvalue" else None
    entries = {entry["run_id"]: entry for entry in index["episodes"]} if index else {}
    prepared = load_manifest(config)
    for episode in prepared["episodes"]:
        target = next(root.rglob(f"{episode['run_id']}/episodes/episode_000/trajectory.npz")).parent
        if source == "stage":
            with np.load(target / "trajectory.npz", allow_pickle=False) as arrays:
                annotation = build_stage_annotation(run_id=episode["run_id"],
                    trajectory_sha256=episode["trajectory_sha256"], done=arrays["done"],
                    keyframes=[dict(step=6, kind="positive")])
            atomic_json(target / "stage_annotation.json", annotation)
        else:
            entry = entries[episode["run_id"]]
            destination = target / "rynnvalue_values.npz"
            shutil.copyfile(entry["reward_path"], destination)
            atomic_json(target / "trajectory_reward.rynnvalue.json", {
                "schema_version": 1, "source": source, "run_id": episode["run_id"],
                "trajectory_sha256": episode["trajectory_sha256"],
                "observations_sha256": episode["observations_sha256"],
                "values_file": destination.name, "values_sha256": sha256_file(destination),
                "reward_config": index["reward_config"], "episode": episode, "entry": entry,
                "prepared": {key: value for key, value in prepared.items() if key != "episodes"}})
    export = json.loads((root / "export.json").read_text())
    export["files"] = {str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
                       for path in root.rglob("*") if path.is_file() and path.name != "export.json"}
    atomic_json(root / "export.json", export)


@pytest.mark.parametrize("archive", [False, True], ids=["directory", "zip"])
@pytest.mark.parametrize("method,source", [("bc", "final"), ("iql", "final"),
    ("iql", "stage"), ("iql", "rynnvalue"), ("iql", "sparse")])
def test_main_origin_segments_survive_portable_export_and_server_cache(configured, tmp_path, method, source, archive):
    _record_producers(configured)
    root, config, frozen = portable_export(configured, tmp_path)
    _add_reward_source(root, config, source)
    config.raw["training"]["method"] = method
    config.raw["reward"].update(source=source, rynnvalue=source == "rynnvalue")
    Path(config.section("paths")["dataset_sources"][0]).rename(tmp_path / "disconnected-recordings")
    Path(config.section("paths")["work_dir"]).rename(tmp_path / "disconnected-work")
    before = {str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    input_path = Path(shutil.make_archive(str(tmp_path / "transfer"), "zip", root)) if archive else root
    with open_run_source(input_path) as imported:
        imported_before = {str(path.relative_to(imported)): sha256_file(path)
                           for path in imported.rglob("*") if path.is_file()}
        task, = discover_tasks(imported, "libero_x_vla")
        raw = prepare_global_inputs(task, tmp_path / "server-inputs", config.raw, source)
        loaded = LoadedConfig(tmp_path / "effective.yaml", raw)
        manifest = load_manifest(loaded)
        assert raw["data"]["include_post_success"] is False
        assert raw["data"]["success_consecutive_steps"] == 5
        assert {episode["run_id"]: episode["split"] for episode in manifest["episodes"]} == {
            member["run_id"]: member["split"] for member in frozen["members"]}
        branch = next(episode for episode in manifest["episodes"] if episode["run_id"] == "branch")
        assert branch["handoff_steps"] == [5, 12]
        assert [(chunk["start"], chunk["end"]) for chunk in branch["chunks"]] == [
            (0, 5), (5, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 22)]
        assert [segment["open_loop_steps"] for segment in branch["trajectory_segments"]] == [8, 4, 3]
        assert [segment["origin_run_id"] for segment in branch["trajectory_segments"]] == ["root", "branch", "branch"]
        assert all("policy_instruction" not in chunk for chunk in branch["chunks"])
        reward = load_reward_index(loaded) if method == "iql" else None
        stats = _stats(7), _stats(8)
        replay = (ReplayDataset(loaded, *stats, reward_index=reward) if reward
                  else ActionDataset(loaded, *stats))
        cache, hit = build_replay_cache(loaded, tmp_path / "cache", *stats, reward)
        assert not hit
        cached = CachedReplayDataset(cache)
        assert len(cached) == len(replay)
        masks = {}
        for index in range(len(replay)):
            sample, restored = replay[index], cached[index]
            assert sample.keys() == restored.keys()
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, restored[key]), key
                else:
                    assert value == restored[key], key
            assert sample["prompt"] == "do the task"
            if sample["run_id"] == "branch":
                masks[sample["start"]] = int(sample["actor_action_mask"].sum())
        assert masks[5] == 7 and masks[9] == 3 and masks[12] == 6 and masks[15] == 3
        assert imported_before == {str(path.relative_to(imported)): sha256_file(path)
                                   for path in imported.rglob("*") if path.is_file()}
    assert before == {str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    if archive:
        assert not imported.exists()
