from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .dependencies import (
    dataset_service, http_error, offline_job_service, service,
    training_dataset_service, trajectory_evaluation_service,
    robometer_evaluation_service,
)
from .models import TrajectoryEvaluationRequest

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("/summary")
async def summary(request: Request): return service(request).dataset_summary()


@router.get("/runs")
async def runs(
    request: Request,
    task_id: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    eligible: bool | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=50),
):
    try:
        return training_dataset_service(request).list_runs_page(
            task_id, source_type, outcome, eligible,
            page=page, page_size=page_size,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/runs/{run_id}")
async def run_detail(run_id: str, request: Request):
    try:
        return trajectory_evaluation_service(request).detail(
            run_id, robometer_evaluation_service(request)
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/evaluations", status_code=201)
async def evaluate_trajectories(body: TrajectoryEvaluationRequest, request: Request):
    try:
        return offline_job_service(request).start_trajectory_evaluation(
            task_id=body.task_id,
            run_ids=None if body.run_ids is None else list(body.run_ids),
            overwrite=(body.run_ids is not None) if body.overwrite is None else body.overwrite,
            evaluators=list(body.evaluators),
        )
    except Exception as exc:
        raise http_error(exc) from exc


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
