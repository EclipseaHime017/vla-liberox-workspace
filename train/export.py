"""Export a saved actor without resuming its optimizer or publishing registry/latest."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
import tempfile

import yaml

from core.registry import model_contract
from core.config import LoadedConfig, validate_train_config
from core.config_sources import read_source
from core.io import sha256_file, stable_hash
from core.artifacts import parent_identity, validate_parent, validate_policy
from core.model_config import model_backend, model_config, model_signature
from train.methods.base import critic_artifacts


def _source_hashes(checkpoint: Path) -> dict[str, str]:
    names = {"checkpoint.json", "effective_config.yaml"}
    names.update(path.name for path in checkpoint.glob("*.pt") if path.name != "trainer.pt")
    names.update(path.name for path in checkpoint.glob("*identity.json"))
    result = {}
    for name in sorted(names):
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Checkpoint input must be a regular file: {path}")
        result[name] = sha256_file(path)
    return result


def export_checkpoint(checkpoint: Path, destination: Path) -> Path:
    """Restore the exact actor, validate its parent, then atomically export a new policy.

    Uses the saved model/runtime settings. No optimizer, replay data or rewards are
    loaded, and no training step is executed. LoRA is merged by its family backend.
    """
    checkpoint = checkpoint.expanduser().resolve()
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export destination already exists: {destination}")
    if destination == checkpoint or checkpoint in destination.parents:
        raise ValueError("Export destination must be outside the source checkpoint")
    source_hashes = _source_hashes(checkpoint)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    step = metadata.get("step")
    if metadata.get("schema_version") != 2 or type(step) is not int or step < 1:
        raise ValueError("Checkpoint requires schema 2 and a positive integer step")
    match = re.fullmatch(r"step_(\d+)", checkpoint.name)
    if match is None or int(match[1]) != step:
        raise ValueError("Checkpoint directory step does not match checkpoint.json")
    raw = read_source(checkpoint / "effective_config.yaml")
    if LoadedConfig(checkpoint / "effective_config.yaml", raw).digest != metadata.get("config_sha256"):
        raise ValueError("Checkpoint effective configuration hash mismatch")
    config = validate_train_config(copy.deepcopy(raw), checkpoint / "effective_config.yaml")
    settings = model_config(config.raw)
    if model_signature(config.raw) != metadata.get("model_config"):
        raise ValueError("Checkpoint model settings do not match saved configuration")
    method = config.section("training")["method"]
    if method != metadata.get("algorithm"):
        raise ValueError("Checkpoint algorithm does not match saved configuration")
    parent = metadata.get("model_parent")
    if not isinstance(parent, dict):
        raise ValueError("Checkpoint lacks a frozen parent identity; refusing inferred parent")
    if metadata.get("base_checkpoint") != settings["base_checkpoint"]:
        raise ValueError("Checkpoint base path does not match saved configuration")
    if metadata.get("stats_key") != parent.get("stats_key"):
        raise ValueError("Checkpoint normalization does not match its parent")
    if model_contract(settings) != metadata.get("model_contract"):
        raise ValueError("Saved configuration input/output contract differs from checkpoint")
    settings.update(base_revision=parent["revision"], stats_key=parent["stats_key"],
                    contract=parent["contract"])
    validate_parent(parent, settings)
    if model_contract(settings) != metadata.get("model_contract"):
        raise ValueError("Checkpoint input/output contract does not match its parent")
    config.raw["model"] = settings
    backend = model_backend(config.raw)
    # training=True creates the same trainable parameter layout, including LoRA;
    # inference-only loading would freeze every pi05 parameter and export no actor.
    components = backend.load_components(config, training=True)
    try:
        if parent_identity(getattr(components, "parent", None)) != parent_identity(parent):
            raise ValueError("Loaded base model differs from checkpoint parent")
        if getattr(components, "identity", None) != metadata.get("model_identity"):
            raise ValueError("Loaded model identity differs from checkpoint")
        backend.restore_actor(components, settings, checkpoint)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".checkpoint-export-", dir=destination.parent) as temporary:
            staging = Path(temporary) / destination.name
            staging.mkdir()
            artifacts = backend.export_actor(components, settings, staging, checkpoint=checkpoint)
            critic = critic_artifacts(config, checkpoint, staging)
            if critic:
                artifacts = {**artifacts, **critic, "component_sha256": {
                    **artifacts["component_sha256"], **critic["component_sha256"]}}
            io = parent["contract"]["io"]
            compatibility = {"base_checkpoint": parent["checkpoint"], "stats_key": parent["stats_key"],
                             "action_horizon": io["replay_horizon"], "action_dim": io["action_dim"],
                             "proprio_dim": io["proprio_dim"]}
            payload = {**artifacts, **compatibility, "schema_version": 5,
                       "family": settings["family"], "parent": parent, "model_config": settings,
                       "policy_id": destination.name, "label": f"{method.upper()} · step {step}",
                       "algorithm": method, "training_step": step,
                       "dataset_sha256": metadata["dataset_sha256"],
                       "reward_sha256": metadata["reward_sha256"],
                       "compatibility_sha256": stable_hash(compatibility)}
            manifest = staging / "policy.yaml"
            manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
            validate_policy(manifest, payload)
            if _source_hashes(checkpoint) != source_hashes:
                raise ValueError("Checkpoint changed while exporting; retry after checkpoint save completes")
            (staging / "checkpoint_export.json").write_text(json.dumps({
                "source_checkpoint": str(checkpoint), "step": step, "source_sha256": source_hashes,
                "component_sha256": payload["component_sha256"],
            }, indent=2) + "\n")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Export destination already exists: {destination}")
            staging.rename(destination)
    finally:
        holder = getattr(components, "checkpoint_holder", None)
        if holder is not None:
            holder.cleanup()
    return destination / "policy.yaml"
