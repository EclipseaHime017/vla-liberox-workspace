"""Policy boundary consumed by simulation workers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class PolicyProvider(Protocol):
    def load(self, open_loop_steps: int, policy_id: str = "base") -> None: ...
    def unload(self) -> None: ...
    def predict(
        self,
        observation: dict[str, Any],
        prompt: str,
        disabled_policy_cameras: tuple[str, ...] = (),
    ) -> Sequence[np.ndarray]: ...
    def process_action(self, action: np.ndarray) -> np.ndarray: ...
    def metadata(self) -> dict[str, Any]: ...
