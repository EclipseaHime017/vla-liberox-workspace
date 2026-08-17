from fastapi import APIRouter, Request

from .dependencies import service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("/summary")
async def summary(request: Request): return service(request).dataset_summary()
