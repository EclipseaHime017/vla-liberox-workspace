"""Gain-independent, mutually exclusive translation / rotation selection."""
from __future__ import annotations

import numpy as np


class SpaceMouseIntent:
    def __init__(self, mode: str, switch_ratio: float):
        if mode not in {"exclusive", "combined"}:
            raise ValueError("SpaceMouse motion_mode must be exclusive or combined")
        if not np.isfinite(switch_ratio) or switch_ratio < 1:
            raise ValueError("SpaceMouse intent_switch_ratio must be finite and >= 1")
        self.mode = mode
        self.switch_ratio = switch_ratio
        self.reset()

    def reset(self) -> None:
        self.selected = "idle"
        self.translation_strength = 0.0
        self.rotation_strength = 0.0

    def apply(self, axes: np.ndarray) -> np.ndarray:
        """Input is deadzoned/mapped normalized axes, before gain or EMA.

        Idle requires a clear winner to start. During motion the existing group
        is retained until the challenger exceeds it by switch_ratio. This is
        amplitude hysteresis, not a dwell timer or an extra smoothing window.
        """
        translation = float(np.linalg.norm(axes[:3]))
        rotation = float(np.linalg.norm(axes[3:]))
        self.translation_strength, self.rotation_strength = translation, rotation
        if translation == rotation == 0:
            self.selected = "idle"
        elif self.mode == "combined":
            self.selected = "combined"
        elif translation > self.switch_ratio * rotation:
            self.selected = "translation"
        elif rotation > self.switch_ratio * translation:
            self.selected = "rotation"
        # Otherwise keep the previous group, or stay idle when still ambiguous.
        result = axes.copy()
        if self.selected == "idle":
            result.fill(0.0)
        elif self.selected == "translation":
            result[3:] = 0.0
        elif self.selected == "rotation":
            result[:3] = 0.0
        return result
