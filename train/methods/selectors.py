"""Inference-time action selection for methods that export a critic.

A selector receives ``candidates`` (N sampled action chunks for one observation) and their
critic ``scores`` and returns the index of the chunk to execute. Methods whose actor is
used directly (BC, IQL) export no selector; sampling-based methods such as IDQL register
one here and declare ``exports_critic`` in ``core.methods``.
"""
from __future__ import annotations

from typing import Protocol

import torch


class Selector(Protocol):
    def __call__(self, candidates: torch.Tensor, scores: torch.Tensor) -> int: ...


SELECTORS: dict[str, Selector] = {}
