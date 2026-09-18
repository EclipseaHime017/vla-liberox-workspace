"""Lightweight batch policy evaluation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error, offline_job_service
from .models import DeleteEvaluationRequest, EvaluationQueueRequest, EvaluationRequest


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("/preview")
async def preview(body: EvaluationRequest, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).preview_evaluation, body.model_dump())
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("", status_code=201)
async def create(body: EvaluationRequest, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).start_evaluation, body.model_dump())
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("")
async def history(
    request: Request,
    task_id: str | None = Query(default=None),
    task_ids: list[str] | None = Query(default=None),
    policy_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    try:
        return await run_in_threadpool(offline_job_service(request).list_evaluations,
            task_id=task_id,
            **({"task_ids": task_ids} if task_ids is not None else {}),
            policy_id=policy_id,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/queue")
async def queue(request: Request):
    return await run_in_threadpool(offline_job_service(request).evaluation_queue)


@router.post("/queue", status_code=201)
async def enqueue(body: EvaluationQueueRequest, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).enqueue_evaluation, body.model_dump())
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{evaluation_id}")
async def detail(evaluation_id: str, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).get_evaluation, evaluation_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{evaluation_id}/stop")
async def stop(evaluation_id: str, request: Request):
    try:
        return await run_in_threadpool(offline_job_service(request).stop_evaluation, evaluation_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/{evaluation_id}")
async def delete(
    evaluation_id: str, body: DeleteEvaluationRequest, request: Request
):
    try:
        return await run_in_threadpool(offline_job_service(request).delete_evaluation,
            evaluation_id, body.confirm_evaluation_id
        )
    except Exception as exc:
        raise http_error(exc) from exc
