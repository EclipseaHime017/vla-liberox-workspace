import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
import yaml

from conftest import _episode
from test_gamma_dataset import frozen
from test_replay import _stats
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import _split
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql import critic_diagnostics
from backend.app.services.p7_reward_dataset import derive_p7_dataset
from backend.app.services.training_dataset_service import TrainingDatasetService
from backend.app.services.critic_diagnostic_service import CriticDiagnosticService


@pytest.fixture
def p7_source(configured, tmp_path):
    source = tmp_path / "all55"
    roots = []
    for split, count in (("train", 4), ("validation", 1)):
        roots += [f"parent-{i}" for i in range(100) if _split(f"parent-{i}", 7, .2) == split][:count]
    for parent in roots:
        _episode(source, parent, action_count=17)
        for number in range(10):
            _episode(source, f"{parent}-branch-{number}", kind="branch", resume=5,
                     action_source="human", action_count=22, success=True, success_from=13)
        for path in source.rglob("run.json"):
            run = json.loads(path.read_text())
            if run["id"] == parent or run["id"].startswith(parent + "-branch-"):
                run.update(root_session_id=parent, parent_session_id=parent if run["kind"] == "branch" else None)
                path.write_text(json.dumps(run))
    configured.raw["paths"]["dataset_sources"] = [str(source)]
    configured.raw["data"].update(split_seed=7, validation_fraction=.2)
    configured.raw["reward"].update(shaping_weight=.1, accumulate_primitive_steps=False)
    configured.raw["iql"]["critic_image_size"] = 16
    return frozen(configured, tmp_path, "rynnvalue")


def test_p7_real_registration_existing_queue_and_w5_fit(p7_source, configured, tmp_path, monkeypatch):
    path, original, old = p7_source
    protected = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr("vla_rynn_iql.rewards.annotate_manifest", lambda *a, **k: pytest.fail("No VLM"))
    result = derive_p7_dataset(path, "original-version", "P7")
    reader = object.__new__(TrainingDatasetService)
    reader.root = path.parent.parent
    reader.lock = threading.RLock()
    reader.repository = SimpleNamespace(upsert=lambda *a: None)
    child = reader.require_ready_for_training(result["dataset_id"])
    assert json.loads(Path(result["dataset_manifest"]).read_text())["members"] == json.loads(path.read_text())["members"]
    assert len(child["members"]) == 55
    version = reader.validate_version(result["dataset_id"], result["version_id"])
    config = load_train_config(Path(version["config_path"]))
    assert config.raw["data"]["bootstrap_failed_end"] and not config.raw["data"]["include_post_success"]
    assert version["reward_config"] == {**old["reward_config"], "gamma": .99, "zero_success_potential": True}
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
        "reward_source": "rynnvalue", "task_id": "LEVEL1::task", "policy_id": settings["base_id"],
        "max_steps": 1200, "open_loop_steps": 8, "seed": 7, "init_state_index": 0,
        "train_steps": 10000, "batch_size": 64})
    spec = yaml.safe_load(queued["config_path"].read_text())
    assert spec["critic"] == {"train_steps": 10000, "batch_size": 64, "seed": 7}
    actual = load_train_config(Path(spec["training_config"]))
    assert actual.raw["data"]["bootstrap_failed_end"] and not actual.raw["data"]["include_post_success"]
    assert actual.raw["reward"]["zero_success_potential"] and actual.raw["reward"]["gamma"] == .99
    # Exercise real Q/V fit on the queue-created config, bounded synthetic budget.
    # Model normalization is substituted; there is no actor/model or VLM call.
    monkeypatch.setattr(critic_diagnostics, "replay_stats", lambda *_: (_stats(7), _stats(8)))
    monkeypatch.setattr(critic_diagnostics, "_plots", lambda *_: None)
    monkeypatch.setattr("vla_rynn_iql.models.model_backend", lambda *_: pytest.fail("No actor"))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = critic_diagnostics.run_critic_diagnostic(actual, actual, tmp_path / "fit",
            train_steps=2, batch_size=64, seed=7, device="cpu")
    finally:
        torch.set_num_threads(previous)
    assert not summary["actor_loaded_for_training"] and not summary["actor_updated"]
    rows = list(csv.DictReader((tmp_path / "fit/qv.csv").open()))
    train = [r for r in rows if r["split"] == "train"]
    assert len([r for r in train if "branch" not in r["run_id"] and int(r["end"]) == 17]) == 4
    assert len([r for r in train if "branch" in r["run_id"] and int(r["end"]) == 18]) == 40
    assert all(float(r["bootstrap_mask"]) == 1 for r in train if "branch" not in r["run_id"] and int(r["end"]) == 17)
    assert all(int(r["end"]) <= 18 for r in train if "branch" in r["run_id"])
    assert all(float(r["bootstrap_mask"]) == 0 for r in train if "branch" in r["run_id"] and int(r["end"]) == 18)
    assert all(p.read_bytes() == data for p, data in protected.items())


def test_p7_requires_original_55_fixed_split(configured, tmp_path):
    path, _, _ = frozen(configured, tmp_path, "rynnvalue")
    with pytest.raises(ValueError, match="55 members"):
        derive_p7_dataset(path, "original-version", "P7")
    assert len(list(path.parent.parent.glob("*/dataset.json"))) == 1


def test_p7_isolated_cli(p7_source, tmp_path):
    path, _, _ = p7_source
    root = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "isolated"
    for relative in ("liberox-vla-adapter-terminal/scripts/derive_p7_dataset.py",
                     "liberox-vla-adapter-terminal/backend/app/services/p7_reward_dataset.py"):
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    run = subprocess.run([sys.executable, str(isolated / "liberox-vla-adapter-terminal/scripts/derive_p7_dataset.py"),
        "--runtime-root", str(root), "--dataset-manifest", str(path),
        "--version-id", "original-version", "--name", "P7 isolated"], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    receipt = json.loads(run.stdout)
    assert receipt["critic_request"] == {"dataset_id": receipt["dataset_id"], "task_id": "LEVEL1::task",
        "reward_source": "rynnvalue", "train_steps": 10000, "batch_size": 64, "seed": 7, "init_state_index": 0}
