"""Trainability and backbone persistence; no dependency on BC or IQL."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def configure_components(components: Any, settings: dict, *, training: bool) -> None:
    model = components.model
    mode = settings["backbone"]
    model.requires_grad_(False)
    if training and mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA requires peft==0.11.1 in the VLA training environment") from exc
        lora = settings["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules="all-linear", init_lora_weights="gaussian", bias="none",
        ))
        # Match VLA-Adapter finetune.py: queries are learned along with LoRA.
        model.action_queries.requires_grad_(True)
        components.model = model
    elif training and mode == "full":
        model.requires_grad_(True)
    model.train(training and mode != "frozen")
    for name in ("action_head", "proprio_projector"):
        module = getattr(components, name)
        enabled = training and settings[name] == "train"
        module.requires_grad_(enabled)
        module.train(enabled)


def trainable_parameters(components: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for name in ("model", "action_head", "proprio_projector")
                  for parameter in getattr(components, name).parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable actor parameters")
    return parameters


def parameter_counts(components: Any) -> dict[str, dict[str, int]]:
    return {name: {"total": sum(p.numel() for p in module.parameters()),
                   "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
            for name, module in (("backbone", components.model), ("action_head", components.action_head),
                                 ("proprio_projector", components.proprio_projector))}


def save_backbone(components: Any, settings: dict, directory: Path) -> None:
    mode = settings["backbone"]
    if mode == "frozen":
        return
    model = components.model
    if mode == "lora":
        # All trainable deltas, including the unwrapped action-query embedding.
        state = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
    else:
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, directory / "backbone.pt")


def restore_backbone(components: Any, settings: dict, directory: Path) -> None:
    if settings["backbone"] == "frozen":
        return
    state = torch.load(directory / "backbone.pt", map_location="cpu", weights_only=True)
    if settings["backbone"] == "lora":
        parameters = dict(components.model.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError("LoRA checkpoint keys do not match trainable backbone parameters")
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != parameters[name].shape:
                    raise ValueError(f"LoRA checkpoint shape mismatch: {name}")
                parameters[name].copy_(value)
    else:
        components.model.load_state_dict(state, strict=True)


def export_backbone(components: Any, settings: dict, directory: Path) -> bool:
    """Final export only: merge LoRA in-place after resumable checkpoint is saved.

    Inference receives ordinary VLA weights and does not import PEFT or trainers.
    This intentionally stores a full backbone for adapted policies.
    """
    if settings["backbone"] == "frozen":
        return False
    if settings["backbone"] == "lora":
        components.model = components.model.merge_and_unload(safe_merge=True)
    torch.save({name: value.detach().cpu() for name, value in components.model.state_dict().items()},
               directory / "backbone.pt")
    return True
