"""FastAPI dependencies and exception translation."""

from fastapi import HTTPException, Request

from ..services.run_service import RunService
from ..services.dataset_service import DatasetService
from ..services.offline_job_service import OfflineJobService
from ..services.training_dataset_service import TrainingDatasetService
from ..services.trajectory_evaluation_service import TrajectoryEvaluationService
from ..services.robometer_evaluation_service import RobometerEvaluationService
from ..services.stage_annotation_service import StageAnnotationService


def service(request: Request) -> RunService:
    current = getattr(request.app.state, "run_service", None)
    if current is None:
        current = RunService(request.app.state.manager)
        request.app.state.run_service = current
    return current


def dataset_service(request: Request) -> DatasetService:
    current = getattr(request.app.state, "dataset_service", None)
    if current is None:
        current = DatasetService(service(request))
        request.app.state.dataset_service = current
    return current


def training_dataset_service(request: Request) -> TrainingDatasetService:
    current = getattr(request.app.state, "training_dataset_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Offline dataset service is unavailable")
    return current


def offline_job_service(request: Request) -> OfflineJobService:
    current = getattr(request.app.state, "offline_job_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Offline job service is unavailable")
    return current


def trajectory_evaluation_service(request: Request) -> TrajectoryEvaluationService:
    current = getattr(request.app.state, "trajectory_evaluation_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Trajectory evaluation service is unavailable")
    return current


def robometer_evaluation_service(request: Request) -> RobometerEvaluationService:
    current = getattr(request.app.state, "robometer_evaluation_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Robometer evaluation service is unavailable")
    return current


def stage_annotation_service(request: Request) -> StageAnnotationService:
    current = getattr(request.app.state, "stage_annotation_service", None)
    if current is None:
        raise HTTPException(status_code=503, detail="Stage annotation service is unavailable")
    return current


def http_error(exc: Exception) -> HTTPException:
    from ..core.exceptions import ConflictError

    if isinstance(exc, ConflictError):
        return HTTPException(status_code=409, detail=exc.detail())
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Run not found")
    if isinstance(exc, (ValueError, IndexError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
