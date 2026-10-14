import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import _episode
from test_replay import FakeAnnotator, _stats
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.data import prepare_dataset, load_manifest
from vla_rynn_iql.io import atomic_json, sha256_file, stable_hash
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index, reward_manifest_digest
from vla_rynn_iql.server_inputs import discover_tasks, prepare_global_inputs, saved_reward, open_run_source


def global_runs(configured):
    """PC-side global sidecars, independent of database/frozen dataset/cache paths."""
    config = copy.deepcopy(configured)
    config.raw["reward"].update(source="final", alpha=0., final_normalization="initial_chunk_v1")
    prepare_dataset(config)
    annotate_manifest(config, FakeAnnotator())
    index = load_reward_index(config)
    prepared = load_manifest(config)
    for episode, entry in zip(prepared["episodes"], index["episodes"]):
        directory = Path(episode["trajectory_path"]).parent
        destination = directory / f"trajectory_reward.{entry['reward_sha256']}.npz"
        shutil.copyfile(entry["reward_path"], destination)
        atomic_json(directory / "trajectory_reward.final.json", {
            "schema_version": 1, "source": "final", "run_id": episode["run_id"],
            "trajectory_sha256": episode["trajectory_sha256"],
            "observations_sha256": episode["observations_sha256"],
            "values_file": destination.name, "values_sha256": sha256_file(destination),
            "observations_fingerprint": [1, 2, 3, 4],
            "reward_config": index["reward_config"], "episode": episode,
            "prepared": {key: value for key, value in prepared.items() if key != "episodes"},
            "entry": entry,
        })
    root = Path(config.raw["paths"]["dataset_sources"][0])
    task, = discover_tasks(root, "libero_x_vla")
    return task, config, index


def portable_export(configured, tmp_path, *, include_unmarked=False):
    task, config, _ = global_runs(configured)
    root = Path(config.raw["paths"]["dataset_sources"][0])
    if include_unmarked:
        _episode(root, "unmarked")
        run_json = next(root.rglob("unmarked/run.json"))
        record = json.loads(run_json.read_text())
        record["root_session_id"] = "unmarked"
        atomic_json(run_json, record)
    destination = tmp_path / "export"
    members, exported = [], []
    for source in sorted(root.rglob("run.json")):
        record = json.loads(source.read_text())
        run_id = record["id"]
        target = destination / "runs" / "task" / run_id
        shutil.copytree(source.parent, target)
        episode = source.parent / "episodes" / "episode_000"
        with np.load(episode / "trajectory.npz") as arrays:
            count = len(arrays["env_action"])
        member = {"run_id": run_id, "root_run_id": record["root_session_id"],
                  "parent_run_id": record["parent_session_id"],
                  "resume_step": record["resume_step"] or 0, "end_step": count,
                  "split": "validation" if run_id == "unmarked" else "train",
                  "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in (("manifest", source), ("trajectory", episode / "trajectory.npz"),
                                       ("observations", episode / "trajectory_observations.npz"))}}
        members.append(member)
        exported.append({"run_id": run_id, "path": str(target.relative_to(destination)),
                         **{key: member[key] for key in ("split", "root_run_id", "parent_run_id")},
                         "evaluations": {}})
    immutable = {"task_id": task.task_id, "selection": {"mode": "manual"},
                 "validation_fraction": .3, "split_seed": 17, "success_consecutive_steps": 5,
                 "members": members}
    frozen = {"schema_version": 1, "id": "ds-export", "project_id": "libero_x_vla",
              **immutable, "include_post_success": False, "dataset_sha256": stable_hash(immutable)}
    atomic_json(destination / "dataset.json", frozen)
    files = {str(path.relative_to(destination)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted(destination.rglob("*")) if path.is_file()}
    atomic_json(destination / "export.json", {"schema_version": 1, "complete": True,
        "kind": "dataset_runs_export", "dataset_id": frozen["id"], "task_id": task.task_id,
        "dataset_sha256": frozen["dataset_sha256"], "runs_root": "runs", "members": exported, "files": files})
    return destination, config, frozen


@pytest.mark.parametrize("child_root", [False, True])
def test_portable_export_preserves_splits_and_reuses_rewards(configured, tmp_path, child_root):
    root, config, frozen = portable_export(configured, tmp_path)
    original = Path(config.raw["paths"]["dataset_sources"][0])
    shutil.rmtree(original)
    task, = discover_tasks(root / "runs" if child_root else root, "libero_x_vla")
    before = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    raw = prepare_global_inputs(task, tmp_path / "inputs", config.raw, "final")
    loaded = LoadedConfig(tmp_path / "effective.yaml", raw)
    assert raw["data"]["include_post_success"] is False
    assert raw["data"]["split_seed"] == 17
    assert {ep["run_id"]: ep["split"] for ep in load_manifest(loaded)["episodes"]} == {
        member["run_id"]: member["split"] for member in frozen["members"]}
    assert load_reward_index(loaded)["complete"]
    assert before == {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    assert json.loads((tmp_path / "inputs/export_provenance.json").read_text())["source_dataset_sha256"] == frozen["dataset_sha256"]


def test_export_bc_uses_all_members_without_evaluation_and_iql_rejects_missing(configured, tmp_path):
    root, config, _ = portable_export(configured, tmp_path, include_unmarked=True)
    task, = discover_tasks(root, "libero_x_vla")
    assert {run.run_id for run in task.selected(None)} == {"root", "branch", "unmarked"}
    with pytest.raises(ValueError, match="unmarked"):
        task.selected("final")
    config.raw["training"]["method"] = "bc"
    raw = prepare_global_inputs(task, tmp_path / "bc", config.raw, "final",
                                data_overrides={"include_post_success": True})
    assert raw["data"]["include_post_success"] is True
    assert len(load_manifest(LoadedConfig(config.path, raw))["episodes"]) == 3


def test_export_zip_uses_training_files_and_cleans_temporary_view(configured, tmp_path):
    root, config, _ = portable_export(configured, tmp_path)
    (root / "video.mp4").write_bytes(b"unused video")
    archive = Path(shutil.make_archive(str(tmp_path / "dataset"), "zip", root))
    with open_run_source(archive) as source:
        assert not (source / "video.mp4").exists()
        task, = discover_tasks(source, "libero_x_vla")
        raw = prepare_global_inputs(task, tmp_path / "inputs", config.raw, "final")
        assert load_reward_index(LoadedConfig(config.path, raw))["complete"]
    assert not source.exists()


@pytest.mark.parametrize("mutation", ["replace", "unbound_reward", "unused_reward"])
def test_export_keeps_selected_snapshot_and_bc_ignores_unused_rewards(configured, tmp_path, mutation):
    root, config, _ = portable_export(configured, tmp_path)
    task, = discover_tasks(root, "libero_x_vla")
    export = json.loads((root / "export.json").read_text())
    if mutation == "replace":
        export["dataset_name"] = "changed while queued"
    elif mutation == "unbound_reward":
        name = str(task.runs[0].rewards["final"].relative_to(root))
        export["files"].pop(name)
    else:
        _, values = saved_reward(task.runs[0], "final")
        values.write_bytes(b"unused reward is corrupt")
        config.raw["training"]["method"] = "bc"
    atomic_json(root / "export.json", export)
    if mutation == "unbound_reward":
        task, = discover_tasks(root, "libero_x_vla")
    if mutation == "unused_reward":
        raw = prepare_global_inputs(task, tmp_path / "bc", config.raw, "final")
        assert len(load_manifest(LoadedConfig(config.path, raw))["episodes"]) == 2
    else:
        with pytest.raises(ValueError):
            prepare_global_inputs(task, tmp_path / "invalid", config.raw, "final")


@pytest.mark.parametrize("mutation", ["bytes", "missing", "traversal", "incomplete", "split"])
def test_export_integrity_fails_before_training(configured, tmp_path, mutation):
    root, config, _ = portable_export(configured, tmp_path)
    path = root / "export.json"
    export = json.loads(path.read_text())
    if mutation in ("bytes", "missing"):
        target = next(root.rglob("trajectory.npz"))
        target.write_bytes(b"invalid") if mutation == "bytes" else target.unlink()
    elif mutation == "traversal":
        export["members"][0]["path"] = "../outside"
    elif mutation == "incomplete":
        export["complete"] = False
    else:
        export["members"][0]["split"] = "validation"
    atomic_json(path, export)
    with pytest.raises((ValueError, FileNotFoundError)):
        task, = discover_tasks(root, "libero_x_vla")
        prepare_global_inputs(task, tmp_path / "invalid", config.raw, "final")


def test_direct_copy_keeps_global_arrays_and_requires_no_original_pc(configured, tmp_path):
    task, config, index = global_runs(configured)
    expected = {entry["run_id"]: Path(entry["reward_path"]).read_bytes() for entry in index["episodes"]}
    root = Path(config.raw["paths"]["dataset_sources"][0])
    moved = tmp_path / "server-runs"
    shutil.copytree(root, moved)
    root.rename(tmp_path / "offline-recordings")
    Path(config.raw["paths"]["work_dir"]).rename(tmp_path / "offline-work")
    task, = discover_tasks(moved, "libero_x_vla")
    before = {p: sha256_file(p) for p in moved.rglob("*") if p.is_file()}
    raw = prepare_global_inputs(task, tmp_path / "runtime", config.raw, "final")
    loaded = LoadedConfig(tmp_path / "effective.yaml", raw)
    rewards = load_reward_index(loaded)
    for entry in rewards["episodes"]:
        assert Path(entry["reward_path"]).read_bytes() == expected[entry["run_id"]]
        assert str(moved) in load_manifest(loaded)["episodes"][0]["trajectory_path"]
    assert len(ReplayDataset(loaded, _stats(7), _stats(8), reward_index=rewards)) > 0
    assert len({ep["split"] for ep in load_manifest(loaded)["episodes"]}) == 1
    raw["reward"]["gamma"] = .92
    adapted = load_reward_index(loaded)
    assert adapted["reward_config"]["gamma"] == .92
    assert before == {p: sha256_file(p) for p in moved.rglob("*") if p.is_file()}


def test_scan_is_lightweight_and_uses_all_marked_runs_not_a_quota(configured, tmp_path, monkeypatch):
    task, config, _ = global_runs(configured)
    root = Path(config.raw["paths"]["dataset_sources"][0])
    _episode(root, "unmarked")
    # Both date layouts are accepted without rewriting paths or labels.
    branch = next(run for run in task.runs if run.run_id == "branch")
    branch.path.parent.rename(branch.path.parent.parent.parent / "branch")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Discovery must not decompress NPZ"))
    task, = discover_tasks(root, "libero_x_vla")
    assert len(task.runs) == 3
    assert {run.run_id for run in task.selected("final")} == {"root", "branch"}
    assert len(task.selected(None)) == 2


@pytest.mark.parametrize("mutation", ["values", "trajectory", "observations", "threshold", "unsafe", "boundaries"])
def test_corrupt_evaluations_fail_without_fallback(configured, tmp_path, mutation):
    task, config, _ = global_runs(configured)
    run = task.runs[0]
    metadata, values = saved_reward(run, "final")
    if mutation == "values":
        values.write_bytes(b"broken")
    elif mutation in {"trajectory", "observations"}:
        metadata[f"{mutation}_sha256"] = "0" * 64
    elif mutation == "threshold":
        metadata["prepared"]["success_consecutive_steps"] += 1
    elif mutation == "unsafe":
        metadata["values_file"] = "../elsewhere.npz"
    else:
        with np.load(values) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["boundary_steps"][1] += 1
        np.savez_compressed(values, **arrays)
        metadata["values_sha256"] = sha256_file(values)
    atomic_json(run.rewards["final"], metadata)
    with pytest.raises(ValueError):
        prepare_global_inputs(task, tmp_path / "runtime", config.raw, "final")


def test_native_rynn_sidecar_without_old_cache_or_dataset_metadata(configured, tmp_path):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    index = load_reward_index(configured)
    for episode, entry in zip(load_manifest(configured)["episodes"], index["episodes"]):
        root = Path(episode["trajectory_path"]).parent
        shutil.copyfile(entry["reward_path"], root / "rynnvalue_evaluation.npz")
        atomic_json(root / "rynnvalue_evaluation.json", {
            "schema_version": 6, "run_id": episode["run_id"],
            "trajectory_sha256": episode["trajectory_sha256"],
            "observations_sha256": episode["observations_sha256"],
            "values_sha256": sha256_file(root / "rynnvalue_evaluation.npz"),
            "reward_config": index["reward_config"],
        })
    task, = discover_tasks(Path(configured.raw["paths"]["dataset_sources"][0]), "libero_x_vla")
    raw = prepare_global_inputs(task, tmp_path / "runtime", configured.raw, "rynnvalue")
    assert len(load_reward_index(LoadedConfig(tmp_path / "effective.yaml", raw))["episodes"]) == 2


def test_bc_does_not_read_reward_arrays_and_sparse_needs_no_model(configured, tmp_path):
    task, config, _ = global_runs(configured)
    _, values = saved_reward(task.runs[0], "final")
    values.write_bytes(b"not used by BC")
    config.raw["training"]["method"] = "bc"
    raw = prepare_global_inputs(task, tmp_path / "bc", config.raw, "final")
    assert raw["reward"]["manifest_path"] is None
    config.raw["training"]["method"] = "iql"
    config.raw["reward"].update(source="sparse", rynnvalue=False)
    raw = prepare_global_inputs(task, tmp_path / "sparse", config.raw, "sparse")
    assert len(load_reward_index(LoadedConfig(tmp_path / "effective.yaml", raw))["episodes"]) == 2


def test_training_settings_do_not_change_input_identity(configured, tmp_path):
    task, config, _ = global_runs(configured)
    digests = []
    for index, steps in enumerate((100, 200)):
        config.raw["training"]["train_steps"] = steps
        config.raw["paths"]["output_dir"] = str(tmp_path / f"outputs{index}")
        raw = prepare_global_inputs(task, tmp_path / f"run{index}", config.raw, "final")
        current = LoadedConfig(tmp_path / "effective.yaml", raw)
        digests.append((load_manifest(current)["dataset_sha256"], reward_manifest_digest(load_reward_index(current))))
    assert digests[0] == digests[1]


def test_implicit_global_stage_reuses_keyframes_without_resaving(configured, tmp_path):
    from vla_rynn_iql.stage_rewards import build_stage_annotation
    task, config, _ = global_runs(configured)
    for run in task.runs:
        path = run.episode_dir / "trajectory.npz"
        with np.load(path) as arrays:
            labels = build_stage_annotation(run_id=run.run_id, trajectory_sha256=sha256_file(path),
                done=arrays["done"], keyframes=[{"step": 8, "kind": "positive"}])
        atomic_json(run.episode_dir / "stage_annotation.json", labels)
    task, = discover_tasks(Path(config.raw["paths"]["dataset_sources"][0]), "libero_x_vla")
    assert len(task.selected("stage")) == 2
    before = {run.run_id: (run.episode_dir / "stage_annotation.json").read_bytes() for run in task.runs}
    config.raw["reward"].update(source="stage", rynnvalue=False)
    raw = prepare_global_inputs(task, tmp_path / "stage-runtime", config.raw, "stage")
    reward = load_reward_index(LoadedConfig(tmp_path / "effective.yaml", raw))
    assert len(reward["episodes"]) == 2
    assert all((run.episode_dir / "stage_annotation.json").read_bytes() == before[run.run_id] for run in task.runs)

    # A neighbouring explicit p=4 reward must not change bare labels' default p=2.
    first = task.runs[0]
    config.raw["reward"]["stage_exponent"] = 4.
    p4 = prepare_global_inputs(task, tmp_path / "stage-p4", config.raw, "stage")
    p4_entry = load_reward_index(LoadedConfig(tmp_path / "p4.yaml", p4))["episodes"][0]
    # Pick the corresponding saved label descriptor independent of hash-name order.
    descriptors = [json.loads(path.read_text()) for path in (tmp_path / "stage-p4/metadata").glob("*.json")]
    metadata = next(item for item in descriptors if item["run_id"] == first.run_id)
    destination = first.episode_dir / "saved-stage.npz"
    shutil.copyfile(p4_entry["reward_path"], destination)
    metadata.update(schema_version=1, values_file=destination.name)
    atomic_json(first.episode_dir / "trajectory_reward.stage.json", metadata)
    task, = discover_tasks(Path(config.raw["paths"]["dataset_sources"][0]), "libero_x_vla")
    from test_server_pipeline import configuration
    from vla_rynn_iql.server_pipeline import training_settings
    server = configuration(tmp_path, task, configured)
    raw = training_settings(server, task, source="stage")
    assert raw["reward"]["stage_exponent"] == 2.
    raw = prepare_global_inputs(task, tmp_path / "mixed-stage", raw, "stage")
    result = load_reward_index(LoadedConfig(tmp_path / "mixed.yaml", raw))
    assert [entry["saved_reward_config"]["stage_exponent"] for entry in result["episodes"]] == [4., 2.]


def test_task_grouping_uses_identity_and_rejects_duplicates(configured):
    task, config, _ = global_runs(configured)
    run = task.runs[0]
    payload = json.loads(run.path.read_text())
    payload["task_id"] = "LEVEL2::task"
    atomic_json(run.path, payload)
    root = Path(config.raw["paths"]["dataset_sources"][0])
    assert [item.task_id for item in discover_tasks(root, "libero_x_vla")] == ["LEVEL1::task", "LEVEL2::task"]
    shutil.copytree(run.path.parent, root / "accidental-duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        discover_tasks(root, "libero_x_vla")
