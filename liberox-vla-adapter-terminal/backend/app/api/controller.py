from fastapi import APIRouter, Request

from .dependencies import http_error, service
from .models import ControllerCalibrationRequest, ControllerGravityRequest
from typing import Literal
from ipaddress import ip_address
from urllib.parse import urlsplit

router = APIRouter(prefix="/api", tags=["controller"])


def local_usb_authorization(request: Request) -> bool:
    """Only a same-origin local UI may open the deployment desktop's polkit dialog."""
    if request.headers.get("X-FACTR-USB-Repair") != "1" or request.client is None:
        return False
    try:
        if not ip_address(request.client.host).is_loopback:
            return False
        host = request.url.hostname
        if host != "localhost" and not ip_address(host).is_loopback:
            return False
        origin = urlsplit(request.headers.get("origin", ""))
        return (origin.scheme == request.url.scheme and origin.netloc == request.url.netloc and
                origin.path in ("", "/") and not origin.query and not origin.fragment and
                request.headers.get("sec-fetch-site", "same-origin") == "same-origin")
    except (ValueError, TypeError):
        return False

@router.get("/controller")
async def status(request: Request, controller_id: Literal["spacemouse", "factr"] = "spacemouse"):
    return service(request).controller_status(controller_id)

@router.get("/controllers")
async def controllers(request: Request):
    return service(request).controller_catalog()

@router.post("/controller/calibrate", status_code=202)
async def calibrate(request: Request, body: ControllerCalibrationRequest | None = None,
                    controller_id: Literal["spacemouse", "factr"] = "spacemouse"):
    try:
        kwargs = {"allow_usb_authorization": True} if controller_id == "factr" and local_usb_authorization(request) else {}
        return service(request).calibrate_controller(controller_id, "reference" if body is None else body.phase, **kwargs)
    except Exception as exc: raise http_error(exc) from exc


@router.post("/controller/gravity")
def gravity(request: Request, body: ControllerGravityRequest,
            controller_id: Literal["factr"] = "factr"):
    # Physical enable/disable waits for ACK; run in FastAPI's thread pool,
    # never block the event loop that serves videos/controller status.
    try: return service(request).set_controller_gravity(controller_id, body.enabled)
    except Exception as exc: raise http_error(exc) from exc
