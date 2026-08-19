"""FastAPI composition root for the local LIBERO-X data studio."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import eval_pickplace_direct as direct

from .api import controller, datasets, drafts, runs, websocket
from .api.models import (
    CreateBranchRequest,
    DeleteSessionRequest,
    DraftRequest,
    UpdateDraftRequest,
)
from .core.config import DEFAULT_UI_CONFIG, UIConfig, load_ui_config
from .core.frontend import frontend_build_info
from .services.run_service import RunService
from .services.dataset_service import DatasetService
from .workers.simulation_worker import SimulationManager


FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
FRONTEND_DIST = FRONTEND_ROOT / "dist"


def create_app(
    ui_config: UIConfig | None = None,
    eval_config: direct.EvalConfig | None = None,
    manager: SimulationManager | None = None,
) -> FastAPI:
    """Compose adapters while retaining injectable workers for tests."""
    ui_config = ui_config or load_ui_config(DEFAULT_UI_CONFIG)
    eval_config = eval_config or direct.load_config(direct.DEFAULT_CONFIG_PATH.resolve())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = manager is None
        worker = manager or SimulationManager(ui_config, eval_config)
        app.state.manager = worker  # compatibility for local diagnostics
        app.state.run_service = RunService(worker)
        app.state.dataset_service = DatasetService(app.state.run_service)
        try:
            yield
        finally:
            if owned:
                await asyncio.to_thread(app.state.run_service.close)

    app = FastAPI(
        title="LIBERO-X Local Data Studio",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(runs.router)
    app.include_router(drafts.router)
    app.include_router(controller.router)
    app.include_router(datasets.router)
    app.include_router(websocket.router)

    @app.get("/api/build-info", tags=["diagnostics"])
    async def build_info():
        return frontend_build_info(FRONTEND_ROOT)

    if (FRONTEND_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        if path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-store"})
        return HTMLResponse(
            "<h1>LIBERO-X frontend is not built</h1>"
            "<p>Run npm install && npm run build in frontend/.</p>",
            status_code=503,
        )

    return app


__all__ = [
    "CreateBranchRequest",
    "DeleteSessionRequest",
    "DraftRequest",
    "UpdateDraftRequest",
    "create_app",
]
