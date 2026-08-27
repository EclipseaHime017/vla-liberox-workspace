"""Immutable training-dataset and annotation endpoints."""

from fastapi import APIRouter, Query, Request

from .dependencies import http_error, offline_job_service, training_dataset_service
from .models import (
    CreateTrainingDatasetRequest,
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
        dataset = training_dataset_service(request).create(
            name=body.name, task_id=body.task_id,
            selection=body.selection.model_dump(),
            validation_fraction=body.validation_fraction,
            split_seed=body.split_seed,
            success_consecutive_steps=body.success_consecutive_steps,
        )
        try:
            job = offline_job_service(request).start_annotation(dataset["id"])
            dataset = training_dataset_service(request).get(dataset["id"])
            dataset["automatic_evaluation_job_id"] = job["id"]
        except Exception as exc:
            # Freezing remains successful if another GPU job is active.  The
            # same non-overwriting evaluation can be resumed from the UI.
            dataset["automatic_evaluation_error"] = str(exc)
        return dataset
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/derive", status_code=201)
async def derive(dataset_id: str, body: DeriveTrainingDatasetRequest, request: Request):
    try:
        dataset = training_dataset_service(request).derive(
            dataset_id, name=body.name, selection=body.selection.model_dump(),
            validation_fraction=body.validation_fraction,
            split_seed=body.split_seed,
            success_consecutive_steps=body.success_consecutive_steps,
        )
        try:
            job = offline_job_service(request).start_annotation(dataset["id"])
            dataset = training_dataset_service(request).get(dataset["id"])
            dataset["automatic_evaluation_job_id"] = job["id"]
        except Exception as exc:
            dataset["automatic_evaluation_error"] = str(exc)
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
        return training_dataset_service(request).verify(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/{dataset_id}/annotations", status_code=201)
async def annotate(dataset_id: str, request: Request):
    try:
        return offline_job_service(request).start_annotation(dataset_id)
    except Exception as exc:
        raise http_error(exc) from exc
