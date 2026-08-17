"""Application package for the local LIBERO-X studio."""

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    """Import the composition root only when the web application is created."""
    from .main import create_app as factory

    return factory(*args, **kwargs)

__all__ = ["create_app"]
