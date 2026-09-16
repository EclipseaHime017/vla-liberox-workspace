"""Persistent offline job, training, and TensorBoard endpoints."""

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error, offline_job_service
from .models import TrainingRunRequest


router = APIRouter(prefix="/api", tags=["offline-jobs"])


@router.get("/jobs")
async def jobs(request: Request):
    return offline_job_service(request).list()


@router.get("/jobs/{job_id}")
async def job(job_id: str, request: Request):
    try:
        return offline_job_service(request).get(job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/jobs/{job_id}/logs")
async def logs(
    job_id: str, request: Request, offset: int = Query(default=0, ge=0),
    limit: int = Query(default=256_000, ge=1, le=1_000_000),
):
    try:
        return offline_job_service(request).logs(job_id, offset, limit)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/jobs/{job_id}/stop")
async def stop(job_id: str, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).stop, job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/training/defaults")
async def defaults(
    request: Request, dataset_id: str | None = Query(default=None), reward_source: str | None = Query(default=None)
):
    try:
        return await run_in_threadpool(offline_job_service(request).defaults, dataset_id, reward_source)
    except Exception as exc:
        raise http_error(exc, key_error_context="Training configuration could not be loaded") from exc


@router.post("/training-runs", status_code=201)
async def train(body: TrainingRunRequest, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).start_training,
            body.dataset_id, body.parameters
        )
    except Exception as exc:
        raise http_error(exc, key_error_context="Training could not be started") from exc


@router.get("/tensorboard")
async def tensorboard(request: Request):
    return offline_job_service(request).tensorboard_status()


@router.get("/training-queue")
async def training_queue(request: Request):
    return await run_in_threadpool(offline_job_service(request).training_queue)


@router.post("/training-queue", status_code=201)
async def enqueue_training(body: TrainingRunRequest, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).enqueue_training,
                                      body.dataset_id, body.parameters)
    except Exception as exc:
        raise http_error(exc, key_error_context="Training could not be queued") from exc


@router.post("/tensorboard/start")
async def start_tensorboard(request: Request):
    try:
        return offline_job_service(request).start_tensorboard()
    except Exception as exc:
        raise http_error(exc) from exc
