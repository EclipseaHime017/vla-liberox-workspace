import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api import datasets, training_datasets
from backend.app.services import dataset_export_service as module
from backend.app.services.storage_maintenance import StorageMaintenance


def service(tmp_path):
    datasets = SimpleNamespace(get=lambda identifier, **_: {"id": identifier, "member_count": 2})
    return module.DatasetExportService(datasets, SimpleNamespace(
        offline_rl_root=tmp_path / "offline", project_root=tmp_path / "data/projects/test"))


def test_background_api_remains_responsive_and_blocks_delete_repair(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def copy(*args, **kwargs):
        kwargs["progress"](stage="复制轨迹", completed_runs=1, current_file="observations.npz")
        started.set()
        assert release.wait(10)
        return args[2]
    monkeypatch.setattr(module, "export_dataset", copy)
    exports = service(tmp_path)
    app = FastAPI()
    app.state.dataset_export_service = exports
    app.include_router(training_datasets.router)
    app.include_router(datasets.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await client.post("/api/training-datasets/ds_test/export")
            assert result.status_code == 202
            await asyncio.to_thread(started.wait, 5)
            progress = await client.get("/api/training-datasets/ds_test/export")
            assert progress.json()["completed_runs"] == 1
            assert progress.json()["status"] == "RUNNING"
            assert (await client.post("/api/training-datasets/ds_other/export")).status_code == 409
            deleted = await client.request("DELETE", "/api/training-datasets/ds_test",
                                          json={"confirm_dataset_id": "ds_test", "force": True})
            assert deleted.status_code == 409
            with pytest.raises(RuntimeError, match="正在导出"):
                StorageMaintenance(SimpleNamespace())._repair(app)
            assert (await client.get("/api/datasets/export?task_id=task")).status_code == 404
    try:
        asyncio.run(check())
    finally:
        release.set()
        exports.close()
    state = exports.get("ds_test")
    assert state["status"] == "COMPLETED" and state["output_path"].endswith("/runs")
    resumed = service(tmp_path)
    assert resumed.get("ds_test") == state
    resumed.close()


@pytest.mark.parametrize("failure", ["persist", "submit", "copy"])
def test_failures_do_not_lock_export_service(tmp_path, monkeypatch, failure):
    exports = service(tmp_path)
    def fail(*_, **__):
        raise OSError("disk full")
    if failure == "persist":
        monkeypatch.setattr(module, "atomic_write_json", fail)
    elif failure == "submit":
        monkeypatch.setattr(exports.worker, "submit", fail)
    else:
        monkeypatch.setattr(module, "export_dataset", fail)
    if failure in {"persist", "submit"}:
        with pytest.raises(OSError):
            exports.start("ds_test")
    else:
        exports.start("ds_test")
    exports.close()
    assert exports.active is False
    assert exports.get("ds_test") is None or exports.get("ds_test")["status"] == "FAILED"
    with exports.mutation_guard():
        pass


def test_restart_marks_interrupted_and_restores_latest_by_creation_time(tmp_path):
    root = tmp_path / "dataset-exports/.jobs"
    root.mkdir(parents=True)
    for name, time in (("zz-old", "2026-01-01T01:01:01.01"), ("aa-new", "2026-01-01T01:01:01.02")):
        (root / f"{name}.json").write_text(json.dumps({"id": name, "dataset_id": "ds_test",
            "created_at": time, "status": "RUNNING"}))
    exports = service(tmp_path)
    try:
        state = exports.get("ds_test")
        assert state["id"] == "aa-new" and state["status"] == "FAILED"
        assert "重启" in state["error"]
    finally:
        exports.close()
