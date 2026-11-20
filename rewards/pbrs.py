"""Reward formulas over an action interval: per-step sparse cost and potential-based shaping.

Every interval accumulates its primitive steps with the per-step discount, so a transition
of L steps bootstraps with gamma^L.
"""
from __future__ import annotations

import numpy as np


def sparse_primitive_return(
    done: np.ndarray, start: int, length: int, gamma: float,
) -> float:
    """Return the discounted primitive-step costs inside one interval."""
    end = start + length
    if length < 1 or start < 0 or end > len(done):
        raise ValueError(
            f"Invalid action interval [{start}, {end}) for {len(done)} actions"
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
    rynnvalue: bool = True,
    zero_success_potential: bool = False,
) -> tuple[float, float, float]:
    """Return sparse, raw PBRS shape, and final rewards for one interval.

    Per-step shaping gamma*phi(k+1) - phi(k) telescopes over the interval, so only the
    potentials at its two ends are needed.
    """
    sparse = sparse_primitive_return(done, start, length, gamma)
    phi_start, phi_end = -float(value_start), -float(value_end)
    if zero_success_potential:
        # done marks actions; observation k is after action k-1.
        if start > 0 and bool(done[start - 1]):
            phi_start = 0.0
        if bool(done[start + length - 1]):
            phi_end = 0.0
    pbrs_shaping = (gamma ** length) * phi_end - phi_start
    dense = shaping_weight * pbrs_shaping if rynnvalue else 0.0
    return sparse, pbrs_shaping, sparse + dense
