"""Independent analysis tools backed by the platform work queue."""
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field, StrictStr
from starlette.concurrency import run_in_threadpool

from studio.api.dependencies import http_error
from studio.api.models import StrictModel
from studio.domain.critic_diagnostic import CriticDiagnosticRequest
from studio.domain.return_distribution import ReturnDistributionRequest


class DeleteToolJobRequest(StrictModel):
    confirm_job_id: StrictStr = Field(min_length=1)


def _service_of(state: str, label: str):
    def service(request: Request):
        current = getattr(request.app.state, state, None)
        if current is None:
            raise HTTPException(status_code=503, detail=f"{label} are unavailable")
        return current
    return service


async def _call(function, *args):
    try:
        return await run_in_threadpool(function, *args)
    except Exception as exc:
        raise http_error(exc) from exc


def _job_routes(router: APIRouter, service, request_model) -> None:
    """List, create, inspect, stop, delete and download one tool's queue jobs."""

    @router.get("")
    async def jobs(request: Request):
        return await _call(service(request).list)

    @router.post("", status_code=201)
    async def create(body: request_model, request: Request):
        return await _call(service(request).start, body.model_dump())

    @router.get("/{job_id}")
    async def detail(job_id: str, request: Request):
        return await _call(service(request).get, job_id)

    @router.post("/{job_id}/stop")
    async def stop(job_id: str, request: Request):
        return await _call(service(request).stop, job_id)

    @router.delete("/{job_id}")
    async def delete(job_id: str, body: DeleteToolJobRequest, request: Request):
        return await _call(service(request).delete, job_id, body.confirm_job_id)

    @router.get("/{job_id}/artifacts/{filename}")
    async def artifact(job_id: str, filename: str, request: Request):
        path = await _call(service(request).artifact, job_id, filename)
        return FileResponse(path, filename=filename)


router = APIRouter(prefix="/api/tools/critic", tags=["tools"])
critic_service = _service_of("critic_diagnostic_service", "Critic diagnostics")


@router.get("/defaults")
async def critic_defaults(request: Request, dataset_id: str | None = Query(default=None)):
    return await _call(critic_service(request).defaults, dataset_id)


_job_routes(router, critic_service, CriticDiagnosticRequest)

returns_router = APIRouter(prefix="/api/tools/returns", tags=["tools"])
returns_service = _service_of("return_distribution_service", "Return distributions")


@returns_router.get("/defaults")
async def returns_defaults(request: Request):
    return await _call(returns_service(request).defaults)


_job_routes(returns_router, returns_service, ReturnDistributionRequest)
