import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "liberox-vla-adapter-terminal"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "liberox-vla-adapter-terminal/scripts"))
from backend.app.services.derived_reward_dataset import derive_gamma_dataset
from backend.app.services.training_dataset_service import TrainingDatasetService, _stable_hash
from backend.app.services.critic_diagnostic_service import CriticDiagnosticService
from vla_rynn_iql.config import validate_train_config
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index, load_pinned_reward_index
from test_rewards import CountingAnnotator
from test_stage_rewards import _save_annotations


def frozen(configured, tmp_path, source):
    configured.raw["reward"]["gamma"] = .9
    prepared_path = prepare_dataset(configured).manifest
    prepared = json.loads(prepared_path.read_text())
    annotate_manifest(configured, CountingAnnotator())
    if source == "final":
        _save_annotations(configured, prepared)
        configured.raw["reward"].update(source="final", alpha=.3, shaping_weight=.1,
            fusion_mode="additive", final_normalization="initial_chunk_v1")
    reward = load_reward_index(configured)
    folder = tmp_path / "datasets" / "original"
    version_dir = folder / "annotations" / "original-version"
    version_dir.mkdir(parents=True)
    members = []
    for ep in prepared["episodes"]:
        artifacts = {}
        for key, target in (("manifest", Path(ep["trajectory_path"]).parents[2] / "run.json"),
                            ("trajectory", ep["trajectory_path"]), ("observations", ep["observations_path"])):
            p = Path(target)
            artifacts[key] = {"path": str(p), "sha256": sha256_file(p), "size": p.stat().st_size}
        members.append({"run_id": ep["run_id"], "split": ep["split"], "artifacts": artifacts,
                        "root_run_id": ep["root_run_id"]})
    dataset = {"schema_version": 1, "id": "original", "name": "old", "task_id": "LEVEL1::task",
        "created_at": "before", "members": members, "selection": {"mode": "manual"},
        "validation_fraction": .2, "split_seed": 7, "success_consecutive_steps": 5,
        "integrity_status": "HEALTHY", "annotation_status": "READY", "include_post_success": True,
        "evaluation_versions": [{"id": "original-version", "status": "READY", "evaluator": source}],
        "evaluation_version_ids": {source: "original-version"}}
    dataset["dataset_sha256"] = _stable_hash(TrainingDatasetService._immutable_payload(dataset))
    prepared.update(source_dataset_id="original", source_dataset_sha256=dataset["dataset_sha256"])
    prepared_path.write_text(json.dumps(prepared))
    config_path = version_dir / "saved.yaml"
    config_path.write_text(yaml.safe_dump(configured.raw))
    reward_path = version_dir / "rewards.json"
    reward_path.write_text(json.dumps(reward))
    version = {"id": "original-version", "dataset_id": "original", "dataset_sha256": dataset["dataset_sha256"],
        "evaluator": source, "status": "READY", "complete": True, "parameters": {"gamma": .9},
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "prepared_manifest_path": str(prepared_path), "prepared_manifest_sha256": sha256_file(prepared_path),
        "reward_manifest_path": str(reward_path), "reward_manifest_sha256": sha256_file(reward_path)}
    (version_dir / "version.json").write_text(json.dumps(version))
    dataset_path = folder / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    return dataset_path, version, reward


@pytest.mark.parametrize("source", ["rynnvalue", "final"])
def test_registration_real_pinned_reduction_public_lookup_and_diagnostic_copy(configured, tmp_path, monkeypatch, source):
    path, version, old = frozen(configured, tmp_path, source)
    protected = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr("vla_rynn_iql.rewards.annotate_manifest", lambda *a, **k: pytest.fail("No VLM"))
    result = derive_gamma_dataset(path, "original-version", "P4 gamma .99")
    reader = object.__new__(TrainingDatasetService)
    reader.root = path.parent.parent
    reader.lock = threading.RLock()
    reader.repository = SimpleNamespace(upsert=lambda *a: None)
    child = reader.require_ready_for_training(result["dataset_id"])
    assert child["evaluation_version_ids"][source] == result["version_id"]
    assert child["members"] == reader.get("original")["members"]
    published = reader.validate_version(result["dataset_id"], result["version_id"])
    # Exact existing diagnostic reward-copy path accepts new READY version.
    copied = CriticDiagnosticService._copy_rewards(published, tmp_path / "diagnostic-rewards")
    assert copied.is_file()
    raw = yaml.safe_load(Path(published["config_path"]).read_text())
    config = validate_train_config(raw, Path(published["config_path"]))
    index = load_pinned_reward_index(config)
    assert index["reward_config"] == {**old["reward_config"], "gamma": .99}
    assert all(p.read_bytes() == data for p, data in protected.items())
    new_by_id = {e["run_id"]: e for e in index["episodes"]}
    differs = False
    for entry in old["episodes"]:
        with np.load(entry["reward_path"]) as before, np.load(new_by_id[entry["run_id"]]["reward_path"]) as after:
            np.testing.assert_array_equal(before["absolute_temporal_distance_seconds"], after["absolute_temporal_distance_seconds"])
            differs |= not np.array_equal(before["final_reward"], after["final_reward"])
    assert differs
    # Real registration/start entry freezes this derived version, not the old active.
    settings = configured.section("model")
    policy = {"policy_id": settings["base_id"], "label": "base", "family": settings["family"],
        "base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
        "content_sha256": "a" * 64, "base_revision": "a" * 64, "model_config": settings,
        "model_contract": model_contract(settings), **dict.fromkeys(("manifest", "action_head",
        "proprio_projector", "training_step", "compatibility_sha256"))}
    def freeze(catalog, task, destination):
        destination.mkdir()
        bddl, init = destination / "task.bddl", destination / "states.init"
        bddl.write_text("synthetic task"); init.write_bytes(b"synthetic init")
        return SimpleNamespace(paths=lambda _: (bddl, init), manifest={"task_revision_sha256": "f" * 64})
    monkeypatch.setattr("backend.app.services.task_registry.freeze_task", freeze)
    jobs = SimpleNamespace(datasets=reader, lock=threading.RLock(), jobs_root=tmp_path / "jobs",
        manager=SimpleNamespace(catalog=None), ui_config=SimpleNamespace(train_environment="vla-liberox"),
        _load_base_config=lambda *args: copy.deepcopy(configured.raw),
        preview_evaluation=lambda *args: {"policy_snapshot": policy, "task_snapshot": {},
                                         "config": {"open_loop_steps": 8, "control_hz": 20}},
        _new_job=lambda **kwargs: kwargs)
    queued = CriticDiagnosticService(jobs).start({"dataset_id": result["dataset_id"],
        "reward_source": source, "task_id": "LEVEL1::task", "policy_id": settings["base_id"],
        "max_steps": 1200, "open_loop_steps": 8, "seed": 7, "init_state_index": 0,
        "train_steps": 10000, "batch_size": 64})
    effective = yaml.safe_load(queued["config_path"].read_text())
    train = yaml.safe_load(Path(effective["training_config"]).read_text())
    assert queued["parameters"]["reward_version_id"] == result["version_id"]
    assert train["reward"]["gamma"] == .99 and train["reward"]["alpha"] == configured.raw["reward"]["alpha"]
    assert train["data"]["include_post_success"] is True
    assert not train["data"].get("bootstrap_failed_end", False)
    assert not train["reward"].get("zero_success_potential", False)
    assert all(p.read_bytes() == data for p, data in protected.items())


@pytest.mark.parametrize("field", ["hash", "split", "w5", "gamma_same"])
def test_bad_source_cannot_publish(configured, tmp_path, field):
    path, version, _ = frozen(configured, tmp_path, "rynnvalue")
    if field == "hash":
        Path(version["reward_manifest_path"]).write_text("{}")
    elif field in {"split", "w5"}:
        key = "prepared_manifest_path" if field == "split" else "config_path"
        p = Path(version[key])
        if field == "split":
            data = json.loads(p.read_text()); data["episodes"][0]["split"] = "wrong"; p.write_text(json.dumps(data))
        else:
            data = yaml.safe_load(p.read_text()); data["data"]["bootstrap_failed_end"] = True; p.write_text(yaml.safe_dump(data))
        version[key.replace("_path", "_sha256")] = sha256_file(p)
        (path.parent / "annotations/original-version/version.json").write_text(json.dumps(version))
    with pytest.raises(ValueError):
        derive_gamma_dataset(path, "original-version", "new", .9 if field == "gamma_same" else .99)
    assert len(list(path.parent.parent.glob("*/dataset.json"))) == 1
