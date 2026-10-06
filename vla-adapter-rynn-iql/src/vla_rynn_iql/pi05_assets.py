"""Dependency-free identity contract for the pinned OpenPI LIBERO conversion."""
from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file, stable_hash

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OFFICIAL_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
NORM_FILE = "assets/physical-intelligence/libero/norm_stats.json"
IDENTITY_FILE = "pi05_identity.json"


def checkpoint_identity(directory: Path, hash_file=sha256_file) -> dict:
    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / IDENTITY_FILE).read_text())
    expected = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                "source": OFFICIAL_CHECKPOINT, "config_name": "pi05_libero",
                "native_action_horizon": 10, "native_action_dim": 32,
                "discrete_state_input": False}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported π₀.₅ checkpoint identity; use prepare_pi05.py to convert the official LIBERO checkpoint")
    if metadata.get("conversion_precision", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("Unsupported π₀.₅ checkpoint conversion precision")
    hashes = metadata.get("files", {})
    if set(hashes) != {"model.safetensors", NORM_FILE}:
        raise ValueError("π₀.₅ checkpoint must bind both model weights and LIBERO normalization assets")
    for name, digest in hashes.items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"π₀.₅ checkpoint hash mismatch: {name}")
    return metadata


def identity_digest(identity: dict) -> str:
    return stable_hash(identity)
