"""Independent analysis tools backed by the platform work queue."""
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field, StrictStr
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error
from .models import StrictModel
from ..domain.critic_diagnostic import CriticDiagnosticRequest


class DeleteDiagnosticRequest(StrictModel):
    confirm_job_id: StrictStr = Field(min_length=1)


router = APIRouter(prefix="/api/tools/critic", tags=["tools"])


def service(request: Request):
    from fastapi import HTTPException
    current = getattr(request.app.state, "critic_diagnostic_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Critic diagnostics are unavailable")
    return current


@router.get("/defaults")
async def defaults(request: Request, dataset_id: str | None = Query(default=None)):
    try:
        return await run_in_threadpool(service(request).defaults, dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("")
async def jobs(request: Request):
    try:
        return await run_in_threadpool(service(request).list)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("", status_code=201)
async def create(body: CriticDiagnosticRequest, request: Request):
    try:
        return await run_in_threadpool(service(request).start, body.model_dump())
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{job_id}")
async def detail(job_id: str, request: Request):
    try:
        return await run_in_threadpool(service(request).get, job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{job_id}/stop")
async def stop(job_id: str, request: Request):
    try:
        return await run_in_threadpool(service(request).stop, job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/{job_id}")
async def delete(job_id: str, body: DeleteDiagnosticRequest, request: Request):
    try:
        return await run_in_threadpool(service(request).delete, job_id, body.confirm_job_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{job_id}/artifacts/{filename}")
async def artifact(job_id: str, filename: str, request: Request):
    try:
        path = await run_in_threadpool(service(request).artifact, job_id, filename)
        return FileResponse(path, filename=filename)
    except Exception as exc:
        raise http_error(exc) from exc
