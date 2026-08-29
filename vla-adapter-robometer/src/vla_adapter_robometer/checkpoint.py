"""Compatibility loader for self-contained Robometer PEFT checkpoints."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def remap_checkpoint_state_dict(
    checkpoint: dict[str, Any], model_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Map saved keys only when name structure and tensor shape agree."""
    model_keys = set(model_state)
    suffixes: dict[tuple[str, tuple[int, ...]], list[str]] = defaultdict(list)
    for key, value in model_state.items():
        parts = key.split(".")
        for start in range(max(1, len(parts) - 2)):
            suffixes[(".".join(parts[start:]), _shape(value))].append(key)

    result: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    used: set[str] = set()
    for key, value in checkpoint.items():
        candidates = [key]
        if key.startswith("model.model."):
            tail = key[len("model.model."):]
            candidates.extend((
                f"model.base_model.model.{tail}",
                f"model.base_model.model.model.{tail}",
                f"model.{tail}",
            ))
        if key.startswith("model."):
            tail = key[len("model."):]
            candidates.extend((tail, f"model.base_model.{tail}"))
        target = next((
            candidate for candidate in candidates
            if candidate in model_keys and candidate not in used
            and _shape(model_state[candidate]) == _shape(value)
        ), None)
        if target is None:
            parts = key.split(".")
            for start in range(min(5, len(parts) - 1)):
                suffix = ".".join(parts[start:])
                matches = [
                    item for item in suffixes.get((suffix, _shape(value)), [])
                    if item not in used
                ]
                if len(matches) == 1:
                    target = matches[0]
                    break
        if target is None:
            unmatched.append(key)
            continue
        result[target] = value
        mapping[key] = target
        used.add(target)
    return result, mapping, unmatched


def install_robometer_checkpoint_loader(setup_utils: Any) -> None:
    """Replace the pinned upstream loader without modifying its checkout."""
    if getattr(setup_utils, "_vla_adapter_checkpoint_patch", False):
        return

    def load_checkpoint(
        model: Any, checkpoint_path: str, cfg: Any,
        load_adapters: bool = True, prefer_model_shards: bool = False,
    ) -> None:
        from safetensors.torch import load_file

        root = Path(checkpoint_path)
        files = setup_utils._get_checkpoint_safetensors_files(
            root, prefer_model_shards=prefer_model_shards,
        )
        if not files:
            raise ValueError(f"No safetensors files found in checkpoint: {root}")
        checkpoint_state: dict[str, Any] = {}
        for path in files:
            checkpoint_state.update(load_file(str(path)))
        if not load_adapters:
            checkpoint_state = {
                key: value for key, value in checkpoint_state.items()
                if "lora_A" not in key and "lora_B" not in key
            }

        model_state = model.state_dict()
        remapped, mapping, unmatched = remap_checkpoint_state_dict(
            checkpoint_state, model_state,
        )
        checkpoint_adapters = {
            key for key in checkpoint_state if "lora_A" in key or "lora_B" in key
        }
        loaded_adapters = checkpoint_adapters.intersection(mapping)
        if load_adapters and checkpoint_adapters != loaded_adapters:
            missing = sorted(checkpoint_adapters - loaded_adapters)
            raise RuntimeError(
                "Robometer adapter key mapping is incomplete: loaded "
                f"{len(loaded_adapters)}/{len(checkpoint_adapters)}; "
                f"first unmatched keys: {missing[:5]}"
            )
        required_heads = {
            key for key in checkpoint_state
            if key.startswith(("progress_head.", "success_head."))
        }
        missing_heads = sorted(required_heads - mapping.keys())
        if missing_heads:
            raise RuntimeError(
                f"Robometer head key mapping is incomplete: {missing_heads[:5]}"
            )
        coverage = len(mapping) / max(1, len(checkpoint_state))
        if coverage < 0.95:
            raise RuntimeError(
                "Robometer checkpoint mapping coverage is too low: "
                f"{len(mapping)}/{len(checkpoint_state)} ({coverage:.1%}); "
                f"first unmatched keys: {unmatched[:5]}"
            )
        missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Mapped Robometer keys remain unexpected: {unexpected_keys[:5]}"
            )
        setup_utils.logger.info(
            f"Loaded {len(mapping)}/{len(checkpoint_state)} checkpoint tensors, "
            f"including {len(loaded_adapters)}/{len(checkpoint_adapters)} adapter "
            f"tensors; {len(unmatched)} source tensors were structurally incompatible"
        )
        if missing_keys:
            setup_utils.logger.info(
                f"Model retains {len(missing_keys)} initialized/base tensors not present "
                "in checkpoint"
            )

    setup_utils._load_checkpoint_weights_from_safetensors = load_checkpoint
    setup_utils._vla_adapter_checkpoint_patch = True
