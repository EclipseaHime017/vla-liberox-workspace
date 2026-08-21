from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import backend.app.services.task_catalog as task_catalog
from backend.app.core.config import AdditionalTaskConfig
from backend.app.policies.vla_adapter import VLAAdapterPolicyProvider
from backend.app.services.task_catalog import ConfiguredTaskCatalog
from backend.app.workers.simulation_worker import SimulationManager


def test_configured_task_catalog_validates_and_loads_selected_state(
    tmp_path: Path, monkeypatch
):
    resolved: list[tuple[str, str]] = []

    def resolve_task(_root, level, task_name):
        resolved.append((level, task_name))
        bddl = tmp_path / f"{task_name}.bddl"
        init = tmp_path / f"{task_name}.init"
        bddl.write_text("task", encoding="utf-8")
        init.write_bytes(b"init")
        return bddl, init

    monkeypatch.setattr(task_catalog.direct, "resolve_task", resolve_task)
    monkeypatch.setattr(
        task_catalog.direct,
        "load_initial_states",
        lambda _runtime, init: [np.asarray([len(init.name)], dtype=np.float64)],
    )
    runtime = SimpleNamespace(
        parse_bddl_file=lambda path: {"language": f"prompt for {Path(path).stem}"}
    )
    config = SimpleNamespace(level="LEVEL1", task_name="task_a")
    catalog = ConfiguredTaskCatalog(
        runtime,
        tmp_path,
        config,
        (AdditionalTaskConfig(level="LEVEL1", task_name="task_b"),),
    )

    assert resolved == [("LEVEL1", "task_a"), ("LEVEL1", "task_b")]
    assert [item["prompt"] for item in catalog.list_tasks()] == [
        "prompt for task_a",
        "prompt for task_b",
    ]
    task_b = "LEVEL1::task_b"
    assert catalog.initial_state(task_b).shape == (1,)
    assert catalog.resolve_id("LEVEL1", "unknown") is None


def test_reused_policy_provider_updates_open_loop_steps():
    provider = object.__new__(VLAAdapterPolicyProvider)
    provider._lock = threading.RLock()
    provider.cfg = SimpleNamespace(num_open_loop_steps=8)
    provider.components = object()

    provider.load(3)

    assert provider.cfg.num_open_loop_steps == 3


class _DraftCatalog:
    default_task_id = "LEVEL1::task_a"

    def entry(self, task_id):
        if task_id not in {"LEVEL1::task_a", "LEVEL1::task_b"}:
            raise ValueError(task_id)
        return SimpleNamespace(task_id=task_id)

    def metadata(self, task_id):
        self.entry(task_id)
        task_name = task_id.split("::", 1)[1]
        return {
            "task_id": task_id,
            "level": "LEVEL1",
            "task_name": task_name,
            "prompt": f"prompt {task_name}",
        }


class _PolicyCatalog:
    def entry(self, policy_id):
        if policy_id not in {"base", "overlay"}:
            raise ValueError(policy_id)
        return SimpleNamespace(
            policy_id=policy_id,
            label="Base" if policy_id == "base" else "Rynn IQL",
            base_checkpoint="VLA-Adapter/LIBERO-Object-Pro",
            manifest=None if policy_id == "base" else Path("/registry/overlay/policy.yaml"),
            compatibility_sha256=None if policy_id == "base" else "compat",
        )


def test_draft_does_not_create_output_until_start(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.lock = threading.RLock()
    manager.active_session_id = None
    manager.draft = None
    manager.catalog = _DraftCatalog()
    manager.policy_catalog = _PolicyCatalog()
    manager.eval_config = SimpleNamespace(
        seed=0, disabled_policy_cameras=()
    )
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager._launch_draft_preview = lambda draft: (
        setattr(draft, "latest_jpeg", b"jpeg"),
        setattr(draft, "preview_status", "READY"),
    )
    started = {}

    def create_original(
        max_steps,
        open_loop_steps,
        task_id=None,
        policy_id="base",
        initial_jpeg=None,
        seed=None,
        disabled_policy_cameras=None,
    ):
        started.update(
            max_steps=max_steps,
            open_loop_steps=open_loop_steps,
            task_id=task_id,
            policy_id=policy_id,
            initial_jpeg=initial_jpeg,
            seed=seed,
            disabled_policy_cameras=disabled_policy_cameras,
        )
        return {"id": "started"}

    manager.create_original = create_original

    draft = manager.create_draft("LEVEL1::task_a", 300, 8)
    assert draft["preview_status"] == "PREPARING"
    assert list(tmp_path.iterdir()) == []
    updated = manager.update_draft(
        task_id="LEVEL1::task_b", max_steps=120, open_loop_steps=2
    )
    assert updated["preview_revision"] == 2
    assert list(tmp_path.iterdir()) == []

    assert manager.start_draft() == {"id": "started"}
    assert manager.draft is None
    assert started == {
        "max_steps": 120,
        "open_loop_steps": 2,
        "task_id": "LEVEL1::task_b",
        "policy_id": "base",
        "initial_jpeg": b"jpeg",
        "seed": 0,
        "disabled_policy_cameras": (),
    }


def test_branch_inherits_parent_policy(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.catalog = _DraftCatalog()
    manager.policy_catalog = _PolicyCatalog()
    manager._persist_effective_config = lambda _record: None
    record = manager._new_record(
        kind="branch",
        max_steps=50,
        open_loop_steps=2,
        policy_id="base",
        parent={
            "id": "parent",
            "root_session_id": "parent",
            "trajectory": "/tmp/source.npz",
            "task_id": "LEVEL1::task_a",
            "policy_id": "overlay",
        },
        resume_step=10,
    )
    assert record.policy_id == "overlay"
    assert record.policy_label == "Rynn IQL"
