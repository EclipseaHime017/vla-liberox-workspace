"""FastAPI dependencies and exception translation."""

from fastapi import HTTPException, Request

from ..services.run_service import RunService


def service(request: Request) -> RunService:
    current = getattr(request.app.state, "run_service", None)
    if current is None:
        current = RunService(request.app.state.manager)
        request.app.state.run_service = current
    return current


def http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Run not found")
    if isinstance(exc, (ValueError, IndexError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
