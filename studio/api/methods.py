"""Training-method catalog: methods, their extra parameters and compatible model families."""
from fastapi import APIRouter

from core.methods import DEFAULT_METHOD, method_catalog
from core.registry import family_capabilities

router = APIRouter(prefix="/api/methods", tags=["training"])


@router.get("")
def list_methods():
    return {"methods": method_catalog(family_capabilities()), "default": DEFAULT_METHOD}
