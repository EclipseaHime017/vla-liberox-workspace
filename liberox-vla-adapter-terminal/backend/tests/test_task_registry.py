from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from backend.app.services.task_registry import RegistryTaskCatalog, TaskSnapshot, freeze_task
import backend.app.services.task_catalog as legacy


def custom_catalog(tmp_path, monkeypatch):
    (tmp_path / "scene.bddl").write_text("(:language complete both steps)")
    (tmp_path / "states.init").write_bytes(b"states")
    variant = dict(task_id="custom::two_steps", level="LEVEL1", task_name="two_steps",
                   goal_prompt="complete both steps", bddl="scene.bddl", init="states.init",
                   instructions=[dict(id="first", text="open drawer"), dict(id="second", text="place bowl")])
    family = dict(schema_version=1, family_id="custom", label="Custom task", variants={"scene": variant})
    (tmp_path / "family.yaml").write_text(yaml.safe_dump(family))
    (tmp_path / "registry.yaml").write_text(yaml.safe_dump(dict(schema_version=1,
        default_task_id=variant["task_id"], families={"custom": "family.yaml"})))
    reads = []
    def load_states(_runtime, path):
        reads.append(path)
        return np.zeros((2, 4))
    monkeypatch.setattr(legacy.direct, "load_initial_states", load_states)
    runtime = SimpleNamespace(parse_bddl_file=lambda path: {"language": Path(path).read_text()})
    return RegistryTaskCatalog(runtime, tmp_path / "registry.yaml"), reads


def test_custom_task_instructions_and_scene_snapshot_are_independent_of_registry_edits(tmp_path, monkeypatch):
    catalog, reads = custom_catalog(tmp_path, monkeypatch)
    task = catalog.metadata(catalog.default_task_id)
    assert task["prompt"] == "complete both steps"
    assert [item["text"] for item in task["instructions"]] == ["open drawer", "place bowl"]
    catalog.list_tasks()
    catalog.list_tasks()
    assert len(reads) == 1
    frozen = freeze_task(catalog, catalog.default_task_id, tmp_path / "record/task_snapshot")
    frozen_bddl, frozen_init = frozen.paths(catalog.default_task_id)
    before = frozen_bddl.read_bytes(), frozen_init.read_bytes()
    (tmp_path / "scene.bddl").write_text("changed scene")
    (tmp_path / "states.init").write_bytes(b"new states")
    reloaded = TaskSnapshot.load(catalog.runtime, frozen.manifest_path)
    assert reloaded.metadata(catalog.default_task_id)["prompt"] == "complete both steps"
    assert before == (frozen_bddl.read_bytes(), frozen_init.read_bytes())
    with pytest.raises(ValueError, match="changed"):
        freeze_task(catalog, catalog.default_task_id, tmp_path / "another")
    frozen_bddl.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        TaskSnapshot.load(catalog.runtime, frozen.manifest_path)


def test_invalid_instruction_ids_are_rejected(tmp_path, monkeypatch):
    catalog, _ = custom_catalog(tmp_path, monkeypatch)
    path = tmp_path / "family.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["variants"]["scene"]["instructions"][1]["id"] = "first"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Duplicate instruction"):
        RegistryTaskCatalog(catalog.runtime, tmp_path / "registry.yaml")
