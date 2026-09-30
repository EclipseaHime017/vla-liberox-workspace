"""Drain requests before a UI-initiated storage repair; never move live runs."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from starlette.responses import JSONResponse

from ..storage.paths import clear_storage_cache, storage_lease
from .inherited_reward_inputs import offline_module

REPAIR_PATH = "/api/datasets/storage/repair"


async def finish_before_cancelling(task):
    """Cancellation must not release a storage gate before its worker finishes."""
    try:
        return await asyncio.shield(task)
    finally:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue


async def storage_thread(function, *args):
    return await finish_before_cancelling(asyncio.create_task(asyncio.to_thread(function, *args)))


class StorageMaintenance:
    def __init__(self, config):
        self.config = config
        self.busy = False
        self.error = None
        self.requests = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.lease = None

    def __enter__(self):
        self.lease = storage_lease(getattr(self.config, "dataset_root", None))
        self.lease.__enter__()
        return self

    def __exit__(self, *exc):
        if self.lease is not None:
            self.lease.__exit__(*exc)
            self.lease = None

    @asynccontextmanager
    async def request(self):
        if self.busy or self.error:
            raise RuntimeError(self.error or "正在修复存储目录，请稍后再试")
        self.requests += 1
        self.idle.clear()
        try:
            yield
        finally:
            self.requests -= 1
            if not self.requests:
                self.idle.set()

    def _repair(self, app):
        worker, jobs = app.state.manager, app.state.offline_job_service
        if worker.active_session_id is not None or worker.draft is not None:
            raise RuntimeError("请先停止仿真并取消草稿，再修复存储目录")
        # Stop dispatch and wait for in-flight sidecar publication before moving.
        try:
            jobs.close()
            jobs._binding_executor = None
            with worker.lock:
                if worker.active_session_id is not None:
                    raise RuntimeError("仿真仍在运行，无法修复存储目录")
                self.__exit__(None, None, None)
                try:
                    module = offline_module(self.config.offline_rl_root, "run_layout")
                    result = module.migrate_layout(self.config.dataset_root, self.config.project_id)
                finally:
                    try:
                        self.__enter__()
                    except Exception as exc:
                        self.error = f"存储修复未完成，请停止服务并使用 --rollback 恢复：{exc}"
                        raise
                if result["moves"]:
                    clear_storage_cache()
                    # Completed in-memory sessions may still contain old absolute paths.
                    worker.sessions.clear()
                    worker._trajectory_cache.clear()
                    worker.refresh_history()
            removed = result.get("removed_date_directories", [])
            retained = result.get("retained_date_directories", [])
            message = ("存储目录已更新，数据与评价保持不变" if result["moves"] else
                       "已清理遗留空日期目录，数据无需迁移" if removed else
                       "未发现可自动修复的目录，非空旧目录已保留" if retained else
                       "未发现数据记录，无需修改" if result["layout"] == "empty" else
                       "已是无日期目录，无需修改")
            return {"status": result["status"], "moved": len(result["moves"]),
                    "layout": result["layout"], "removed_date_dirs": len(removed),
                    "retained_date_dirs": retained,
                    "already_current": result.get("already_current", 0),
                    "skipped": result.get("skipped", []),
                    "message": message}
        finally:
            if not self.error:
                jobs.start_training_queue()

    async def repair(self, app):
        if self.busy or self.error:
            raise RuntimeError(self.error or "已有存储修复正在运行")
        self.busy = True
        try:
            try:
                await asyncio.wait_for(self.idle.wait(), timeout=10)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("仍有文件下载或数据请求，请完成后重试") from exc
            return await storage_thread(self._repair, app)
        finally:
            self.busy = False


class StorageRequestGate:
    def __init__(self, app, maintenance: StorageMaintenance):
        self.app, self.maintenance = app, maintenance

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == REPAIR_PATH:
            await self.app(scope, receive, send)
            return
        if self.maintenance.busy or self.maintenance.error:
            await JSONResponse({"detail": self.maintenance.error or "正在修复存储目录，请稍后再试"},
                               status_code=503)(scope, receive, send)
            return
        async with self.maintenance.request():
            await finish_before_cancelling(asyncio.create_task(self.app(scope, receive, send)))
