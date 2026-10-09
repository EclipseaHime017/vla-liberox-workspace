"""Dependency-free identity contract for the pinned OpenPI LIBERO conversion."""
from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file, stable_hash
from .base_models import BASE_MODELS, base_model
from .model_storage import local_model_directory

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
IDENTITY_FILE = "pi05_identity.json"


def checkpoint_identity(directory: Path, hash_file=sha256_file, *, base_id: str | None = None,
                        parent: dict | None = None, contract: dict | None = None) -> dict:
    directory = local_model_directory(directory).resolve()
    metadata = json.loads((directory / IDENTITY_FILE).read_text())
    base = (base_model(base_id) if base_id is not None else
            next((entry for entry in BASE_MODELS.values() if entry.source == metadata.get("source")), None))
    if base is None or base.family != "pi05":
        raise ValueError("Unregistered π₀.₅ checkpoint source")
    contract = parent["contract"] if parent else (contract or base.contract)
    io = contract["io"]
    source, revision, stats_key = ((parent["source"], parent["source_revision"], parent["stats_key"])
                                   if parent else (base.source, base.revision, base.stats_key))
    norm_file = f"assets/{stats_key}/norm_stats.json"
    expected = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                "source": source, "config_name": contract["architecture"]["config_name"],
                "native_action_horizon": io["native_action_horizon"], "native_action_dim": io["padded_action_dim"],
                "discrete_state_input": contract["architecture"]["discrete_state_input"]}
    if revision is not None:
        expected["source_revision"] = revision
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"π₀.₅ checkpoint identity does not match {base.id}; use prepare_pi05.py --base-model {base.id}")
    if metadata.get("conversion_precision", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("Unsupported π₀.₅ checkpoint conversion precision")
    hashes = metadata.get("files", {})
    if set(hashes) != {"model.safetensors", norm_file}:
        raise ValueError("π₀.₅ checkpoint must bind both model weights and LIBERO normalization assets")
    for name, digest in hashes.items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"π₀.₅ checkpoint hash mismatch: {name}")
    return metadata


def identity_digest(identity: dict) -> str:
    return stable_hash(identity)
