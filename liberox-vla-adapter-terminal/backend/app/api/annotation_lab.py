from fastapi import APIRouter, Request
from typing import Literal
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error

router = APIRouter(prefix="/api/annotation-lab", tags=["annotation-lab"])


class LabOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["plan_only", "localize"] | None = None
    coarse_fps: float | None = Field(default=None, ge=0.25, le=10, allow_inf_nan=False)
    window_seconds: float | None = Field(default=None, ge=0.5, le=4, allow_inf_nan=False)
    cameras: list[str] | None = None


class LabRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(min_length=1, max_length=100)
    options: LabOptions = Field(default_factory=LabOptions)


async def call(request, method, *args):
    try:
        service = request.app.state.annotation_lab_service
        if service is None:
            raise RuntimeError("自动标注实验服务未配置")
        return await run_in_threadpool(getattr(service, method), *args)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/config")
async def config(request: Request):
    return await call(request, "defaults")


@router.get("/experiments")
async def experiments(request: Request):
    return await call(request, "list")


@router.post("/experiments", status_code=202)
async def start(request: Request, body: LabRequest):
    return await call(request, "start", body.run_ids, body.options.model_dump(exclude_none=True))


@router.get("/experiments/{identifier}")
async def status(identifier: str, request: Request):
    return await call(request, "get", identifier)


@router.post("/experiments/{identifier}/stop")
async def stop(identifier: str, request: Request):
    return await call(request, "stop", identifier)


@router.get("/experiments/{identifier}/runs/{run_id}")
async def result(identifier: str, run_id: str, request: Request):
    return await call(request, "result", identifier, run_id)


@router.get("/experiments/{identifier}/runs/{run_id}/evidence/{camera}/{step}")
async def evidence(identifier: str, run_id: str, camera: str, step: int, request: Request):
    path = await call(request, "evidence", identifier, run_id, camera, step)
    return FileResponse(path, media_type="image/jpeg")
