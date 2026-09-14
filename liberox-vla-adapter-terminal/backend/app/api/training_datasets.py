"""Immutable training-dataset and annotation endpoints."""

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from .dependencies import http_error, offline_job_service, training_dataset_service
from .models import (
    CreateTrainingDatasetRequest,
    DatasetAnnotationRequest,
    DatasetPreviewRequest,
    DeleteTrainingDatasetRequest,
    DeriveTrainingDatasetRequest,
)


router = APIRouter(prefix="/api/training-datasets", tags=["training-datasets"])


@router.get("")
async def list_datasets(request: Request, task_id: str | None = Query(default=None)):
    return training_dataset_service(request).list(task_id)


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, request: Request):
    try:
        return training_dataset_service(request).get(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/preview")
async def preview(body: DatasetPreviewRequest, request: Request):
    try:
        return training_dataset_service(request).preview(
            body.task_id, body.selection.model_dump()
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("", status_code=201)
async def create(body: CreateTrainingDatasetRequest, request: Request):
    try:
        dataset = await run_in_threadpool(training_dataset_service(request).create,
            name=body.name, task_id=body.task_id,
            selection=body.selection.model_dump(),
            validation_fraction=body.validation_fraction,
            split_seed=body.split_seed,
            success_consecutive_steps=body.success_consecutive_steps,
        )
        return dataset
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/derive", status_code=201)
async def derive(dataset_id: str, body: DeriveTrainingDatasetRequest, request: Request):
    try:
        dataset = await run_in_threadpool(training_dataset_service(request).derive,
            dataset_id, name=body.name, selection=body.selection.model_dump(),
            validation_fraction=body.validation_fraction,
            split_seed=body.split_seed,
            success_consecutive_steps=body.success_consecutive_steps,
        )
        return dataset
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str, body: DeleteTrainingDatasetRequest, request: Request
):
    try:
        return offline_job_service(request).delete_dataset(
            dataset_id, body.confirm_dataset_id, force=body.force
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/verify")
async def verify(dataset_id: str, request: Request):
    try:
        return await run_in_threadpool(training_dataset_service(request).verify, dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/annotations", status_code=201)
async def annotate(
    dataset_id: str, request: Request,
    body: DatasetAnnotationRequest | None = None,
):
    try:
        return await run_in_threadpool(offline_job_service(request).start_annotation,
            dataset_id, **({} if body is None else body.model_dump(exclude_none=True)),
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{dataset_id}/versions")
async def versions(dataset_id: str, request: Request):
    try:
        return training_dataset_service(request).versions(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/versions/{version_id}/activate")
async def activate_version(dataset_id: str, version_id: str, request: Request):
    try:
        return await run_in_threadpool(training_dataset_service(request).activate_version, dataset_id, version_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{dataset_id}/reward-config")
async def reward_config(dataset_id: str, request: Request):
    try:
        return offline_job_service(request).reward_configuration(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/{dataset_id}/members")
async def members(dataset_id: str, request: Request, page: int = Query(default=1, ge=1),
                  page_size: int = Query(default=5, ge=1, le=50)):
    try:
        return training_dataset_service(request).members_page(dataset_id, page=page, page_size=page_size)
    except Exception as exc:
        raise http_error(exc) from exc
