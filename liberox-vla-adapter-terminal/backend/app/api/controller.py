from fastapi import APIRouter, Request

from .dependencies import http_error, service
from .models import ControllerCalibrationRequest, ControllerGravityRequest
from typing import Literal

router = APIRouter(prefix="/api", tags=["controller"])

@router.get("/controller")
async def status(request: Request, controller_id: Literal["spacemouse", "factr"] = "spacemouse"):
    return service(request).controller_status(controller_id)

@router.get("/controllers")
async def controllers(request: Request):
    return service(request).controller_catalog()

@router.post("/controller/calibrate", status_code=202)
async def calibrate(request: Request, body: ControllerCalibrationRequest | None = None,
                    controller_id: Literal["spacemouse", "factr"] = "spacemouse"):
    try: return service(request).calibrate_controller(controller_id, "reference" if body is None else body.phase)
    except Exception as exc: raise http_error(exc) from exc


@router.post("/controller/gravity")
def gravity(request: Request, body: ControllerGravityRequest,
            controller_id: Literal["factr"] = "factr"):
    # Physical enable/disable waits for ACK; run in FastAPI's thread pool,
    # never block the event loop that serves videos/controller status.
    try: return service(request).set_controller_gravity(controller_id, body.enabled)
    except Exception as exc: raise http_error(exc) from exc
