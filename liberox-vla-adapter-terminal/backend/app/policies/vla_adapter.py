"""VLA-Adapter policy provider."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

import eval_pickplace_direct as direct

from .catalog import PolicyCatalog, PolicyEntry


class VLAAdapterPolicyProvider:
    """Lazy, reusable provider for the checkpoint selected by config.yaml."""

    def __init__(
        self,
        runtime: SimpleNamespace,
        eval_config: direct.EvalConfig,
        catalog: PolicyCatalog,
    ):
        self.runtime = runtime
        self.eval_config = eval_config
        self.cfg = None
        self.components = None
        self.catalog = catalog
        self.current_policy_id: str | None = None
        self.current_policy_entry: PolicyEntry | None = None
        self._base_action_head: OrderedDict[str, Any] | None = None
        self._base_proprio_projector: OrderedDict[str, Any] | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.components is not None

    @staticmethod
    def _cpu_state(module: Any) -> OrderedDict[str, Any]:
        return OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in module.state_dict().items()
        )

    def _apply_policy(self, entry: PolicyEntry) -> None:
        if self.components is None:
            raise RuntimeError("Cannot apply a policy before loading the base model")
        torch = self.runtime.torch
        if entry.is_base:
            if self._base_action_head is None or self._base_proprio_projector is None:
                raise RuntimeError("Base component snapshot is unavailable")
            action_state = self._base_action_head
            proprio_state = self._base_proprio_projector
        else:
            assert entry.action_head is not None and entry.proprio_projector is not None
            action_state = torch.load(
                entry.action_head, map_location="cpu", weights_only=True
            )
            proprio_state = torch.load(
                entry.proprio_projector, map_location="cpu", weights_only=True
            )
        self.components.action_head.load_state_dict(action_state, strict=True)
        self.components.proprio_projector.load_state_dict(proprio_state, strict=True)
        self.components.action_head.eval()
        self.components.proprio_projector.eval()
        self.current_policy_id = entry.policy_id
        self.current_policy_entry = entry

    def load(self, open_loop_steps: int, policy_id: str = "base") -> None:
        with self._lock:
            # Re-validate manifests and component hashes at the actual load
            # boundary. A file removed or modified after draft creation must
            # fail explicitly instead of silently retaining the old overlay.
            if hasattr(self, "catalog"):
                self.catalog.refresh()
            entry = self.catalog.entry(policy_id) if hasattr(self, "catalog") else None
            if self.loaded:
                self.cfg.num_open_loop_steps = open_loop_steps
                if policy_id != getattr(self, "current_policy_id", "base"):
                    if entry is None:
                        raise ValueError(f"Unknown policy_id: {policy_id}")
                    self._apply_policy(entry)
                return
            direct.load_policy_runtime(self.runtime)
            load_config = replace(
                self.eval_config,
                open_loop_steps=open_loop_steps,
                trials=1,
                headless=True,
            )
            self.cfg, self.components = direct.build_model(self.runtime, load_config)
            self._base_action_head = self._cpu_state(self.components.action_head)
            self._base_proprio_projector = self._cpu_state(
                self.components.proprio_projector
            )
            self.current_policy_id = "base"
            self.current_policy_entry = self.catalog.entry("base")
            if policy_id != "base":
                if entry is None:
                    raise ValueError(f"Unknown policy_id: {policy_id}")
                self._apply_policy(entry)

    def unload(self) -> None:
        with self._lock:
            self.cfg = None
            self.components = None
            self.current_policy_id = None
            self.current_policy_entry = None
            self._base_action_head = None
            self._base_proprio_projector = None
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
        policy_id = self.current_policy_id or "base"
        entry = getattr(self, "current_policy_entry", None)
        if entry is None and hasattr(self, "catalog"):
            entry = self.catalog.entry(policy_id)
        return {
            "provider": "vla_adapter",
            "checkpoint": str(self.eval_config.checkpoint),
            "loaded": self.loaded,
            "gpu": gpu,
            "model_device": model_device,
            "policy_id": policy_id,
            "policy_label": entry.label if entry is not None else "VLA-Adapter · Object-Pro（基础模型）",
            "overlay": None if entry is None or entry.manifest is None else str(entry.manifest),
            "model_switching": True,
            "action_schema": {
                "size": 7,
                "components": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
                "range": [-1.0, 1.0],
                "units": "normalized OSC_POSE command",
                "predicted_chunk_size": 8,
            },
        }
