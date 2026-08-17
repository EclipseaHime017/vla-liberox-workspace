from fastapi import APIRouter, Request

from .dependencies import http_error, service

router = APIRouter(prefix="/api/controller", tags=["controller"])

@router.get("")
async def status(request: Request): return service(request).controller_status()

@router.post("/calibrate", status_code=202)
async def calibrate(request: Request):
    try: return service(request).calibrate_controller()
    except Exception as exc: raise http_error(exc) from exc
