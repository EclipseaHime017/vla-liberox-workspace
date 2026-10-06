"""VLA-Adapter policy provider."""

from __future__ import annotations

import threading
from pathlib import Path
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

import eval_pickplace_direct as direct

from .catalog import PolicyCatalog, PolicyEntry
from .checkpoint_view import checkpoint_view


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
        self._checkpoint_view = None
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
            if entry.backbone is not None:
                self.components.model.load_state_dict(
                    torch.load(entry.backbone, map_location="cpu", weights_only=True), strict=True
                )
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
        if hasattr(self.components, "model"):
            self.components.model.eval()
        self.catalog.refresh()
        if self.catalog.entry(entry.policy_id).content_sha256 != entry.content_sha256:
            raise ValueError("Policy weights changed during loading; select the model again")
        self.current_policy_id = entry.policy_id
        self.current_policy_entry = entry

    def load(self, open_loop_steps: int, policy_id: str = "base", *,
             expected_content_sha256: str | None = None) -> None:
        with self._lock:
            # Re-validate manifests and component hashes at the actual load
            # boundary. A file removed or modified after draft creation must
            # fail explicitly instead of silently retaining the old overlay.
            if hasattr(self, "catalog"):
                self.catalog.refresh()
            entry = self.catalog.entry(policy_id) if hasattr(self, "catalog") else None
            if entry is not None and Path(entry.base_checkpoint).expanduser().is_dir():
                entry = self.catalog.select(policy_id)
            if expected_content_sha256 is not None and (
                entry is None or entry.content_sha256 != expected_content_sha256
            ):
                raise ValueError("Selected policy weights have changed; select the model again")
            previous = getattr(self, "current_policy_entry", None)
            changed = (entry is not None and (previous is None or
                       previous.content_sha256 != entry.content_sha256))
            if self.loaded and changed and (
                (previous is not None and previous.backbone is not None)
                or (entry is not None and entry.backbone is not None)
                or previous is None
                or (previous.base_checkpoint, previous.base_revision, previous.stats_key)
                   != (entry.base_checkpoint, entry.base_revision, entry.stats_key)
            ):
                # No full base snapshot on GPU/CPU: reload across adapted backbones.
                # Frozen-backbone overlays keep the existing lightweight switch.
                self.unload()
            if self.loaded:
                self.cfg.num_open_loop_steps = open_loop_steps
                if changed:
                    if entry is None:
                        raise ValueError(f"Unknown policy_id: {policy_id}")
                    try:
                        self._apply_policy(entry)
                    except Exception:
                        self.unload()
                        raise
                self.current_policy_id = policy_id
                self.current_policy_entry = entry
                return
            direct.load_policy_runtime(self.runtime)
            self._checkpoint_view = checkpoint_view(entry)
            load_config = replace(
                self.eval_config,
                checkpoint=self._checkpoint_view.name,
                stats_key=entry.stats_key,
                use_pro_version=("Pro" in entry.base_checkpoint
                                 if self.eval_config.use_pro_version is None else self.eval_config.use_pro_version),
                open_loop_steps=open_loop_steps,
                trials=1,
                headless=True,
            )
            try:
                self.cfg, self.components = direct.build_model(self.runtime, load_config)
            except BaseException:
                self.unload()
                raise
            self._base_action_head = self._cpu_state(self.components.action_head)
            self._base_proprio_projector = self._cpu_state(
                self.components.proprio_projector
            )
            self.current_policy_id = "base"
            self.current_policy_entry = entry if entry.is_base else self.catalog.entry("base")
            if policy_id != "base":
                if entry is None:
                    raise ValueError(f"Unknown policy_id: {policy_id}")
                try:
                    self._apply_policy(entry)
                except Exception:
                    self.unload()
                    raise

    def unload(self) -> None:
        with self._lock:
            self.cfg = None
            self.components = None
            self.current_policy_id = None
            self.current_policy_entry = None
            self._base_action_head = None
            self._base_proprio_projector = None
            view = getattr(self, "_checkpoint_view", None)
            if view is not None:
                view.cleanup()
                self._checkpoint_view = None
            torch = getattr(self.runtime, "torch", None)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(
        self,
        observation: dict[str, Any],
        prompt: str,
        disabled_policy_cameras: tuple[str, ...] = (),
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
                disabled_policy_cameras,
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
