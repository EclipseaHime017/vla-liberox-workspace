"""Persistent offline job, training, and TensorBoard endpoints."""

from fastapi import APIRouter, Query, Request

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
        return offline_job_service(request).stop(job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/training/defaults")
async def defaults(
    request: Request, dataset_id: str | None = Query(default=None)
):
    try:
        return offline_job_service(request).defaults(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/training-runs", status_code=201)
async def train(body: TrainingRunRequest, request: Request):
    try:
        return offline_job_service(request).start_training(
            body.dataset_id, body.parameters
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/tensorboard")
async def tensorboard(request: Request):
    return offline_job_service(request).tensorboard_status()


@router.post("/tensorboard/start")
async def start_tensorboard(request: Request):
    try:
        return offline_job_service(request).start_tensorboard()
    except Exception as exc:
        raise http_error(exc) from exc
