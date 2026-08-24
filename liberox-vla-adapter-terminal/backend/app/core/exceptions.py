"""Typed application errors that preserve useful API conflict details."""

from __future__ import annotations

from typing import Any


class ConflictError(RuntimeError):
    def __init__(self, message: str, *, code: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}

    def detail(self) -> dict[str, Any]:
        return {"message": str(self), "code": self.code, **self.context}
