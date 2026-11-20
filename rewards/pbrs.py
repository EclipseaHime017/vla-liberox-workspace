"""Per-step reward formulas: sparse success cost and potential-based shaping.

Rewards are defined for every action step t of a recording. A training transition of L
steps sums them with the per-step discount and bootstraps with gamma^L.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def success_mask(episode: dict[str, Any]) -> np.ndarray:
    """done[t] for every action t: true from the confirmed terminal action onward."""
    done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
    if episode["terminal_step"] is not None:
        done[int(episode["terminal_step"]):] = True
    return done


def sparse_rewards(done: np.ndarray) -> np.ndarray:
    """-1 per action before success, 0 from the confirming action on."""
    return np.where(np.asarray(done, dtype=bool), 0.0, -1.0)


def shaping_rewards(potential: np.ndarray, done: np.ndarray, gamma: float,
                    zero_success_potential: bool = False) -> np.ndarray:
    """gamma * phi(t+1) - phi(t) per action, from one potential per observation."""
    done = np.asarray(done, dtype=bool)
    phi = np.asarray(potential, dtype=np.float64).copy()
    if phi.shape != (len(done) + 1,):
        raise ValueError(f"Potential needs {len(done) + 1} observations, got {phi.shape}")
    if zero_success_potential:
        # done marks actions; observation t + 1 follows action t.
        phi[1:][done] = 0.0
    return gamma * phi[1:] - phi[:-1]


def discounted_sum(rewards: np.ndarray, start: int, end: int, gamma: float) -> float:
    """sum_k gamma^k * rewards[start + k] over the actions [start, end)."""
    if not 0 <= start < end <= len(rewards):
        raise ValueError(f"Invalid action interval [{start}, {end}) for {len(rewards)} actions")
    return float(np.dot(np.power(gamma, np.arange(end - start)), rewards[start:end]))
