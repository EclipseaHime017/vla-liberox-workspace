"""VLA-Adapter policy provider."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

import eval_pickplace_direct as direct


class VLAAdapterPolicyProvider:
    """Lazy, reusable provider for the checkpoint selected by config.yaml."""

    def __init__(self, runtime: SimpleNamespace, eval_config: direct.EvalConfig):
        self.runtime = runtime
        self.eval_config = eval_config
        self.cfg = None
        self.components = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.components is not None

    def load(self, open_loop_steps: int) -> None:
        with self._lock:
            if self.loaded:
                self.cfg.num_open_loop_steps = open_loop_steps
                return
            direct.load_policy_runtime(self.runtime)
            load_config = replace(
                self.eval_config,
                open_loop_steps=open_loop_steps,
                trials=1,
                headless=True,
            )
            self.cfg, self.components = direct.build_model(self.runtime, load_config)

    def unload(self) -> None:
        with self._lock:
            self.cfg = None
            self.components = None
            torch = getattr(self.runtime, "torch", None)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(
        self,
        observation: dict[str, Any],
        prompt: str,
    ) -> Sequence[np.ndarray]:
        with self._lock:
            if self.cfg is None or self.components is None:
                raise RuntimeError("Policy provider is not loaded")
            return direct.checked_action_chunk(
                self.runtime,
                self.cfg,
                self.components,
                observation,
                prompt,
            )

    def process_action(self, action: np.ndarray) -> np.ndarray:
        with self._lock:
            if self.cfg is None:
                raise RuntimeError("Policy provider is not loaded")
            return self.runtime.process_action(action, self.cfg.model_family)

    def metadata(self) -> dict[str, Any]:
        torch = getattr(self.runtime, "torch", None)
        gpu = "CPU"
        if torch is not None and torch.cuda.is_available():
            index = torch.cuda.current_device()
            gpu = f"cuda:{index} ({torch.cuda.get_device_name(index)})"
        model_device = None
        if self.components is not None:
            try:
                model_device = str(next(self.components.model.parameters()).device)
            except (AttributeError, StopIteration):
                model_device = "unknown"
        return {
            "provider": "vla_adapter",
            "checkpoint": str(self.eval_config.checkpoint),
            "loaded": self.loaded,
            "gpu": gpu,
            "model_device": model_device,
            "model_switching": False,
            "action_schema": {
                "size": 7,
                "components": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
                "range": [-1.0, 1.0],
                "units": "normalized OSC_POSE command",
                "predicted_chunk_size": 8,
            },
        }

