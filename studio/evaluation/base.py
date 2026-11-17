"""Evaluation boundary for simulator outcomes."""

from __future__ import annotations

from typing import Protocol


class Evaluator(Protocol):
    def success(self, done: bool) -> bool: ...
