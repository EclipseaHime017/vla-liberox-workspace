"""Per-chunk reward formulas: sparse success base reward and potential-based shaping."""
from __future__ import annotations

import numpy as np


def sparse_macro_reward(done: np.ndarray, start: int, length: int) -> float:
    """Return the paper sparse reward for one action-chunk transition."""
    terminal = start + length - 1
    if length < 1 or start < 0 or terminal >= len(done):
        raise ValueError(
            f"Invalid macro-action interval [{start}, {start + length}) for {len(done)} actions"
        )
    return 0.0 if bool(done[terminal]) else -1.0


def sparse_primitive_return(
    done: np.ndarray, start: int, length: int, gamma: float,
) -> float:
    """Return the discounted primitive-step costs inside one chunk."""
    end = start + length
    if length < 1 or start < 0 or end > len(done):
        raise ValueError(
            f"Invalid action-chunk interval [{start}, {end}) for {len(done)} actions"
        )
    return float(sum(
        (gamma ** offset) * (0.0 if bool(done[start + offset]) else -1.0)
        for offset in range(length)
    ))


def chunk_reward_components(
    done: np.ndarray,
    start: int,
    length: int,
    value_start: float,
    value_end: float,
    gamma: float,
    shaping_weight: float,
    accumulate_primitive_steps: bool = False,
    rynnvalue: bool = True,
    zero_success_potential: bool = False,
) -> tuple[float, float, float]:
    """Return sparse, raw PBRS shape, and final rewards for one chunk."""
    sparse = (
        sparse_primitive_return(done, start, length, gamma)
        if accumulate_primitive_steps
        else sparse_macro_reward(done, start, length)
    )
    phi_start, phi_end = -float(value_start), -float(value_end)
    if zero_success_potential:
        # done marks actions; observation k is after action k-1.
        if start > 0 and bool(done[start - 1]):
            phi_start = 0.0
        if bool(done[start + length - 1]):
            phi_end = 0.0
    duration = length if accumulate_primitive_steps else 1
    pbrs_shaping = (gamma ** duration) * phi_end - phi_start
    dense = shaping_weight * pbrs_shaping if rynnvalue else 0.0
    return sparse, pbrs_shaping, sparse + dense


def shaped_chunk_reward(
    done: np.ndarray,
    start: int,
    length: int,
    value_start: float,
    value_end: float,
    gamma: float,
    shaping_weight: float,
    accumulate_primitive_steps: bool = False,
) -> float:
    return chunk_reward_components(
        done, start, length, value_start, value_end, gamma, shaping_weight,
        accumulate_primitive_steps,
    )[2]
