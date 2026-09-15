from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .dependencies import (
    dataset_service, http_error, offline_job_service, service,
    training_dataset_service, trajectory_evaluation_service,
    robometer_evaluation_service,
    stage_annotation_service,
)
from .models import RunTestLabelRequest, TrajectoryEvaluationRequest, StageAnnotationRequest
from ..services.dataset_evaluation_detail import attach_dataset_context, attach_inherited_rewards

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
    task_ids: list[str] | None = Query(default=None),
):
    try:
        return await run_in_threadpool(
            training_dataset_service(request).list_runs_page,
            task_id, source_type, outcome, eligible,
            page=page, page_size=page_size,
            **({"task_ids": task_ids} if task_ids is not None else {}),
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/runs/{run_id}")
async def run_detail(run_id: str, request: Request,
                     dataset_id: str | None = None, version_id: str | None = None):
    try:
        result = await run_in_threadpool(
            trajectory_evaluation_service(request).detail,
            run_id, robometer_evaluation_service(request),
            include_global_evaluations=True,
        )
        datasets = training_dataset_service(request)
        result["run"]["is_test"] = datasets.is_test(run_id)
        result = await run_in_threadpool(attach_dataset_context, result, datasets, dataset_id, version_id)
        if dataset_id is not None:
            result = await run_in_threadpool(attach_inherited_rewards, result,
                                            offline_job_service(request), dataset_id)
        if result.get("global_evaluation_pending"):
            offline_job_service(request).schedule_first_reward_snapshot(result["run"])
        return result
    except Exception as exc:
        raise http_error(exc) from exc


@router.patch("/runs/{run_id}/labels")
async def update_run_labels(
    run_id: str, body: RunTestLabelRequest, request: Request,
):
    try:
        return await run_in_threadpool(
            training_dataset_service(request).set_test, run_id, body.is_test,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/runs/{run_id}/stage-annotation")
async def get_stage_annotation(run_id: str, request: Request):
    try:
        return await run_in_threadpool(stage_annotation_service(request).detail, run_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.put("/runs/{run_id}/stage-annotation")
async def save_stage_annotation(run_id: str, body: StageAnnotationRequest, request: Request):
    try:
        return await run_in_threadpool(
            stage_annotation_service(request).save, run_id,
            [frame.model_dump() for frame in body.keyframes], body.exponent, body.revision,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/evaluations", status_code=201)
async def evaluate_trajectories(body: TrajectoryEvaluationRequest, request: Request):
    try:
        return offline_job_service(request).start_trajectory_evaluation(
            task_id=body.task_id,
            run_ids=None if body.run_ids is None else list(body.run_ids),
            overwrite=False if body.overwrite is None else body.overwrite,
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
