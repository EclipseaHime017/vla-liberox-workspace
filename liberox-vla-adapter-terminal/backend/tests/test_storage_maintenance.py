import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services.storage_maintenance import StorageMaintenance, StorageRequestGate
from backend.app.services.inherited_reward_inputs import reconstruct_episode, offline_module
from backend.app.services.dataset_transfer import export_dataset
from backend.app.storage.files import legacy_session_id
from backend.app.storage.paths import clear_storage_cache, storage_path, storage_lease
from backend.app.workers.simulation_worker import SimulationManager
from vla_rynn_iql.run_layout import migrate_layout, rollback_migration
from vla_rynn_iql.portable_dataset import read_bundle
import test_global_reward_inheritance as fixtures
from test_training_platform import make_run


@pytest.fixture(autouse=True)
def isolated_host_processes(monkeypatch):
    # These temporary datasets are not used by the developer's running UI.
    # Keep real flock/job checks, but isolate the legacy /proc compatibility scan.
    from vla_rynn_iql import run_layout
    root = Path(__file__).resolve().parents[3] / "vla-adapter-rynn-iql"
    for module in (run_layout, offline_module(root, "run_layout")):
        monkeypatch.setattr(module, "_legacy_processes", lambda: iter(()))


def recording(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "make_run", lambda _, *a, **kw: make_run(tmp_path / "projects/test", *a, **kw))
    jobs, dataset, run = fixtures.setup_recording(tmp_path, monkeypatch)
    fixtures.native_rynn(jobs, dataset, run)
    return jobs, dataset, run


def test_frozen_sources_global_labels_and_export_survive_relocation(tmp_path, monkeypatch):
    jobs, dataset, run = recording(tmp_path, monkeypatch)
    old = Path(run["trajectory"])
    identity = legacy_session_id(old)
    frozen_path, frozen = jobs.datasets._load(dataset["id"])
    frozen_bytes = frozen_path.read_bytes()
    original_files = {p: p.read_bytes() for p in Path(run["output_dir"]).rglob("*") if p.is_file()}
    migrate_layout(tmp_path, "test")
    clear_storage_cache()
    assert legacy_session_id(storage_path(old)) == identity
    restored = SimulationManager._public_from_persisted(
        storage_path(Path(run["output_dir"])), run, {})
    assert restored["trajectory"] == str(storage_path(old))
    assert jobs.datasets.get(dataset["id"])["integrity_status"] == "HEALTHY"
    episode, _ = reconstruct_episode(jobs, dataset, frozen["members"][0])
    assert Path(episode["trajectory_path"]).is_file()
    for source in ("sparse", "stage", "rynnvalue"):
        assert jobs.defaults(dataset["id"], source)["reward_availability"]["ready"]
    config = tmp_path / "export.yaml"
    config.write_text(yaml.safe_dump(jobs._load_base_config()))
    exported = tmp_path / "bundle"
    export_dataset(jobs.datasets.root.parent, dataset["id"], exported, config,
                   jobs.ui_config.offline_rl_root, required_rewards=("rynnvalue", "stage"))
    assert {"stage", "rynnvalue"} <= read_bundle(exported, verify=True)["rewards"].keys()
    assert frozen_path.read_bytes() == frozen_bytes
    assert all(storage_path(path).read_bytes() == value for path, value in original_files.items())
    # Materializing global bindings may legitimately add sidecars; rollback only
    # allows unchanged recordings and must never discard those newer files.


def test_one_click_repair_refreshes_and_restarts_queue(tmp_path, monkeypatch):
    jobs, _, run = recording(tmp_path, monkeypatch)
    events = []
    jobs.close = lambda: events.append("pause")
    jobs.start_training_queue = lambda: events.append("resume")
    worker = SimpleNamespace(active_session_id=None, draft=None, lock=threading.RLock(),
                             sessions={"old": object()}, _trajectory_cache={},
                             refresh_history=lambda: events.append("refresh"))
    config = SimpleNamespace(dataset_root=tmp_path, project_id="test",
                             offline_rl_root=jobs.ui_config.offline_rl_root)
    gate = StorageMaintenance(config)
    app = SimpleNamespace(state=SimpleNamespace(manager=worker, offline_job_service=jobs))
    async def check():
        with gate:
            result = await gate.repair(app)
            assert result["moved"] == 1 and result["status"] == "COMPLETED"
            assert result["layout"] == "dated" and result["removed_date_dirs"] == 1
            assert not Path(run["output_dir"]).exists()
            assert events == ["pause", "refresh", "resume"]
            assert not worker.sessions and not gate.busy
            assert (await gate.repair(app))["status"] == "UNCHANGED"
            assert events == ["pause", "refresh", "resume", "pause", "resume"]
    asyncio.run(check())


def test_cleanup_only_and_noop_report_layout_without_refreshing_history(tmp_path, monkeypatch):
    jobs, _, run = recording(tmp_path, monkeypatch)
    migrate_layout(tmp_path, "test")
    current = storage_path(Path(run["output_dir"]))
    empty = current.parent / "2026-01-02"
    empty.mkdir()
    retained = current.parent / "2026-01-03"
    retained.mkdir()
    (retained / "notes.txt").write_text("keep me")
    events = []
    jobs.close = lambda: events.append("pause")
    jobs.start_training_queue = lambda: events.append("resume")
    worker = SimpleNamespace(active_session_id=None, draft=None, lock=threading.RLock(),
                             sessions={"current": object()}, _trajectory_cache={},
                             refresh_history=lambda: pytest.fail("no recordings were moved"))
    gate = StorageMaintenance(SimpleNamespace(dataset_root=tmp_path, project_id="test",
                                              offline_rl_root=jobs.ui_config.offline_rl_root))
    app = SimpleNamespace(state=SimpleNamespace(manager=worker, offline_job_service=jobs))
    async def check():
        with gate:
            result = await gate.repair(app)
            assert result["status"] == "COMPLETED" and result["moved"] == 0
            assert result["layout"] == "mixed" and result["removed_date_dirs"] == 1
            assert result["already_current"] == 1
            assert result["retained_date_dirs"] == [str(retained.relative_to(tmp_path))]
            assert "已清理" in result["message"] and not empty.exists()
            result = await gate.repair(app)
            assert result["status"] == "UNCHANGED" and result["removed_date_dirs"] == 0
            assert "非空旧目录已保留" in result["message"]
            assert (retained / "notes.txt").read_text() == "keep me"
            assert worker.sessions and not gate.busy
            assert events == ["pause", "resume", "pause", "resume"]
    asyncio.run(check())


def test_active_simulation_repair_refused_before_pause(tmp_path):
    config = SimpleNamespace(dataset_root=tmp_path)
    gate = StorageMaintenance(config)
    app = SimpleNamespace(state=SimpleNamespace(manager=SimpleNamespace(active_session_id="run"),
                                                offline_job_service=object()))
    async def check():
        with gate, pytest.raises(RuntimeError, match="停止仿真"):
            await gate.repair(app)
        assert not gate.busy
    asyncio.run(check())


def test_gate_rejects_new_requests_and_waits_for_existing_ones(tmp_path, monkeypatch):
    gate = StorageMaintenance(SimpleNamespace(dataset_root=tmp_path))
    entered = []
    monkeypatch.setattr(gate, "_repair", lambda app: entered.append("migrate") or {"status": "UNCHANGED"})
    async def app(scope, receive, send):
        pytest.fail("new request must not reach services during repair")
    async def check():
        messages = []
        async def send(message): messages.append(message)
        async with gate.request():
            task = asyncio.create_task(gate.repair(None))
            await asyncio.sleep(0)
            assert gate.busy and not entered
            await StorageRequestGate(app, gate)({"type": "http", "path": "/api/datasets/runs"}, None, send)
            assert messages[0]["status"] == 503
        await task
        assert entered == ["migrate"] and not gate.busy
    asyncio.run(check())


def test_other_storage_consumer_blocks_one_click(tmp_path, monkeypatch):
    jobs, _, run = recording(tmp_path, monkeypatch)
    jobs.close = lambda: None
    jobs.start_training_queue = lambda: None
    worker = SimpleNamespace(active_session_id=None, draft=None, lock=threading.RLock())
    gate = StorageMaintenance(SimpleNamespace(dataset_root=tmp_path, project_id="test",
                                              offline_rl_root=jobs.ui_config.offline_rl_root))
    app = SimpleNamespace(state=SimpleNamespace(manager=worker, offline_job_service=jobs))
    async def check():
        with gate, storage_lease(tmp_path), pytest.raises(RuntimeError, match="busy"):
            await gate.repair(app)
        assert Path(run["output_dir"]).is_dir() and not gate.error
    asyncio.run(check())


def test_cancelled_repair_failure_releases_gate_only_after_worker_finishes(tmp_path, monkeypatch):
    gate = StorageMaintenance(SimpleNamespace(dataset_root=tmp_path))
    started, release = threading.Event(), threading.Event()
    def fail(app):
        started.set()
        assert release.wait(5)
        raise RuntimeError("consumer busy")
    monkeypatch.setattr(gate, "_repair", fail)
    async def check():
        task = asyncio.create_task(gate.repair(None))
        while not started.is_set():
            await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()  # repeated cancellation cannot abandon the rename worker
        await asyncio.sleep(.01)
        assert gate.busy and not task.done()
        release.set()
        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            await task
        assert not gate.busy
    asyncio.run(check())
