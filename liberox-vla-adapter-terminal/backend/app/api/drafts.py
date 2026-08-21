from fastapi import APIRouter, Request, Response

from .dependencies import http_error, service
from .models import DraftRequest, UpdateDraftRequest

router = APIRouter(prefix="/api/draft", tags=["draft"])

@router.get("")
async def get_draft(request: Request): return service(request).get_draft()

@router.post("", status_code=201)
async def create_draft(body: DraftRequest, request: Request):
    try: return service(request).create_draft(
        body.task_id,
        body.max_steps,
        body.open_loop_steps,
        body.policy_id,
        body.seed,
        body.disabled_policy_cameras,
    )
    except Exception as exc: raise http_error(exc) from exc

@router.patch("")
async def update_draft(body: UpdateDraftRequest, request: Request):
    try: return service(request).update_draft(
        task_id=body.task_id,
        policy_id=body.policy_id,
        max_steps=body.max_steps,
        open_loop_steps=body.open_loop_steps,
        seed=body.seed,
        disabled_policy_cameras=body.disabled_policy_cameras,
    )
    except Exception as exc: raise http_error(exc) from exc

@router.delete("")
async def discard_draft(request: Request): return {"discarded": service(request).discard_draft()}

@router.get("/preview.jpg")
async def draft_preview(request: Request):
    try: return Response(service(request).draft_preview(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as exc: raise http_error(exc) from exc

@router.post("/start", status_code=201)
async def start_draft(request: Request):
    try: return service(request).start_draft()
    except Exception as exc: raise http_error(exc) from exc
