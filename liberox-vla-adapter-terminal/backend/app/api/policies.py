"""Policy overlay catalog and local model management endpoints."""

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error
from .models import CopyPolicyRequest, DeletePolicyRequest, RenamePolicyRequest


router = APIRouter(prefix="/api/models", tags=["models"])


def _service(request: Request):
    current = getattr(request.app.state, "policy_management_service", None)
    if current is None:
        from ..services.policy_management_service import PolicyManagementService

        current = PolicyManagementService(
            request.app.state.manager,
            getattr(request.app.state, "offline_job_service", None),
        )
        request.app.state.policy_management_service = current
    return current


@router.get("")
async def list_models(request: Request):
    try:
        return await run_in_threadpool(_service(request).list)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{policy_id}")
async def model_detail(policy_id: str, request: Request):
    try:
        return await run_in_threadpool(_service(request).detail, policy_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.patch("/{policy_id}")
async def rename_model(policy_id: str, body: RenamePolicyRequest, request: Request):
    try:
        return _service(request).rename(policy_id, body.label)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{policy_id}/copy", status_code=201)
async def copy_model(policy_id: str, body: CopyPolicyRequest, request: Request):
    try:
        return _service(request).copy(policy_id, body.label)
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/{policy_id}")
async def delete_model(policy_id: str, body: DeletePolicyRequest, request: Request):
    try:
        return _service(request).delete(policy_id, body.confirm_policy_id)
    except Exception as exc:
        raise http_error(exc) from exc
