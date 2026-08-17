from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .dependencies import dataset_service, http_error, service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("/summary")
async def summary(request: Request): return service(request).dataset_summary()


@router.get("/runs")
async def runs(request: Request, task_id: str | None = Query(default=None)):
    return dataset_service(request).list_runs(task_id)


@router.get("/export")
async def export(request: Request, task_id: str = Query(min_length=1)):
    try:
        path, filename = dataset_service(request).export_task(task_id)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(Path(path).unlink, missing_ok=True),
        )
    except Exception as exc:
        raise http_error(exc) from exc
