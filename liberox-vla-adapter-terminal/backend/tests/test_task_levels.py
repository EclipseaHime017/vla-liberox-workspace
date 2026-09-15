"""Task discovery must not load policies or construct simulation environments."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from backend.app.core.config import AdditionalTaskConfig, load_ui_config
from backend.app.services.task_catalog import ConfiguredTaskCatalog
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.workers.simulation_worker import PreviewService, SimulationManager
import backend.app.services.task_catalog as tasks_module
from test_ui_config_storage import VALID


def make_catalog(tmp_path, monkeypatch):
    scene = "SCENE_LEVEL4__T7_open_drawer"
    scene_dir = tmp_path / "libero/libero_x"
    for folder, suffix in (("bddl", "bddl"), ("init", "init")):
        path = scene_dir / folder / "LEVEL4" / f"{scene}.{suffix}"
        path.parent.mkdir(parents=True)
        path.write_text("fixture")
    language_dir = scene_dir / "LEVEL5"
    language_dir.mkdir()
    for variant in ("L5-1", "L5-2"):
        (language_dir / f"{variant}.jsonl").write_text(json.dumps({
            "task_id": "007", "variant": None, "task_desc": f"{variant} actual prompt",
        }))
    reads = []
    parses = []

    def load_states(_runtime, path):
        reads.append(path)
        return np.zeros((10, 5))

    def parse(path):
        parses.append(path)
        return {"language": "open drawer"}

    monkeypatch.setattr(tasks_module.direct, "load_initial_states", load_states)
    catalog = ConfiguredTaskCatalog(SimpleNamespace(parse_bddl_file=parse), tmp_path,
        SimpleNamespace(level="LEVEL4", task_name=scene), (
            AdditionalTaskConfig("LEVEL5", scene, ("L5-1", "L5-2"), True),
            AdditionalTaskConfig("LEVEL3", "missing_scene", optional=True),
        ))
    return catalog, reads, parses, scene


def test_level5_uses_level4_assets_and_distinct_official_prompts(tmp_path, monkeypatch):
    catalog, reads, parses, scene = make_catalog(tmp_path, monkeypatch)
    entries = catalog.list_tasks()
    assert len(entries) == 4
    assert len({entry["task_id"] for entry in entries}) == 4
    assert [entry["prompt"] for entry in entries[:3]] == [
        "open drawer", "L5-1 actual prompt", "L5-2 actual prompt"]
    for entry in entries[1:3]:
        assert entry["level"] == "LEVEL5" and entry["scene_level"] == "LEVEL4"
        assert entry["init_state_count"] == 10
        assert catalog.paths(entry["task_id"]) == catalog.paths(catalog.default_task_id)
        assert catalog.resolve_id("LEVEL5", scene) is None  # Never guess a prompt variant.
    assert not entries[-1]["available"]
    with pytest.raises(ValueError, match="unavailable"):
        catalog.initial_state(entries[-1]["task_id"])
    for _ in range(5):
        catalog.list_tasks()
    assert len(reads) == len(parses) == 1
    assert catalog.resolve_id("LEVEL4", scene) == catalog.default_task_id


def test_level5_attribute_key_and_invalid_records(tmp_path):
    folder = tmp_path / "libero/libero_x/LEVEL5"
    folder.mkdir(parents=True)
    path = folder / "L5-1.jsonl"
    path.write_text('{"task_id":"070","variant":"A2","task_desc":"large grey bowl"}\n')
    assert ConfiguredTaskCatalog._level5_prompts(tmp_path, "L5-1") == {("070", "A2"): "large grey bowl"}
    for content in ('{}', '[]', '{"task_id":"7","variant":[]}',
                    '{"task_id":"7","task_desc":""}', path.read_text() * 2):
        path.write_text(content)
        with pytest.raises(ValueError):
            ConfiguredTaskCatalog._level5_prompts(tmp_path, "L5-1")


def test_level5_identity_persisted_in_run_branch_and_batch_evaluation(tmp_path, monkeypatch):
    catalog, _, _, scene = make_catalog(tmp_path, monkeypatch)
    task_id = f"LEVEL5::{scene}::L5-2"
    policy = SimpleNamespace(policy_id="base", label="Base", base_checkpoint="base",
        manifest=None, compatibility_sha256=None, stats_key="stats",
        action_head=None, proprio_projector=None, training_step=None)
    manager = object.__new__(SimulationManager)
    manager.catalog = catalog
    manager.eval_config = SimpleNamespace(seed=0, disabled_policy_cameras=(), control_hz=20)
    manager.ui_config = SimpleNamespace(output_root=tmp_path / "runs", project_id="test")
    manager._policy_entry = lambda _: policy
    manager.policy_catalog = SimpleNamespace(refresh=lambda: None, entry=lambda _: policy)
    record = manager._new_record(kind="original", max_steps=16, open_loop_steps=8,
        task_id=task_id, seed=19, init_state_index=2)
    config = yaml.safe_load((record.output_dir / "config.yaml").read_text())
    assert config["task"]["task_id"] == task_id
    assert config["task"]["level"] == "LEVEL5"
    assert config["task"]["prompt"] == "L5-2 actual prompt"
    branch = manager._new_record(kind="branch", max_steps=16, open_loop_steps=8,
        task_id=catalog.default_task_id, seed=0, resume_step=3,
        parent={"id": record.id, "trajectory": "source.npz", "task_id": task_id,
            "policy_id": "base", "seed": 19, "init_state_index": 2})
    assert branch.task_id == task_id and branch.task_prompt == record.task_prompt
    assert branch.seed == 19 and branch.init_state_index == 2
    service = object.__new__(OfflineJobService)
    service.manager = manager
    snapshot, _ = service._evaluation_snapshots(task_id, "base")
    assert snapshot["prompt"] == record.task_prompt and snapshot["level"] == "LEVEL5"
    assert "/LEVEL4/" in snapshot["bddl_path"]


@pytest.mark.parametrize("extra", [
    "{level: LEVEL5, task_name: drawer}",
    "{level: LEVEL5, task_name: drawer, prompt_variants: [L5-6]}",
    "{level: LEVEL5, task_name: drawer, prompt_variants: [L5-1, L5-1]}",
    "{level: LEVEL5, task_name: drawer, prompt_variants: [{}]}",
    "{level: LEVEL4, task_name: drawer, prompt_variants: [L5-1]}",
    "{level: LEVEL6, task_name: drawer}",
    "{level: LEVEL2, task_name: drawer, optional: 'true'}",
])
def test_task_extension_config_rejects_invalid_values(tmp_path, extra):
    path = tmp_path / "ui.yaml"
    path.write_text(VALID.replace("additional_tasks: []", f"additional_tasks: [{extra}]"))
    with pytest.raises((ValueError, TypeError)):
        load_ui_config(path)


def test_preview_reuses_physical_scene_between_language_variants():
    created, closed = [], []

    def create(bddl, _config, **kwargs):
        created.append((bddl, kwargs["seed"]))
        return object()

    manager = SimpleNamespace(
        eval_config=object(), lock=threading.RLock(),
        ui_config=SimpleNamespace(preview_fps=100, preview_width=4, preview_height=4, jpeg_quality=85),
        catalog=SimpleNamespace(paths=lambda task_id: (Path("other" if task_id == "other" else "shared"), Path("init"))),
        simulator=SimpleNamespace(create=create, close=closed.append, restore=lambda *_: None,
            render_operator_preview=lambda *_: np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    preview = PreviewService(manager)
    try:
        for task_id, seed in (("L4", 0), ("L5-1", 0), ("L5-2", 0), ("other", 0), ("other", 1)):
            target = SimpleNamespace(id=task_id, task_id=task_id, seed=seed,
                latest_frame_version=0, preview_event=threading.Event())
            preview.submit(target, np.zeros(3))
            assert target.preview_event.wait(3)
            assert target.preview_error is None
        assert created == [(Path("shared"), 0), (Path("other"), 0), (Path("other"), 1)]
    finally:
        preview.close()
    assert len(closed) == 3
