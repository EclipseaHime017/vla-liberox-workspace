#!/usr/bin/env python3
"""Download the official LIBERO checkpoint, convert it, and bind normalization identity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT.parent / "weights/pi05_libero_torch")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Official converter output precision; float32 requires substantially more host RAM")
    args = parser.parse_args()
    # JAX is used for conversion only, not allowed to preallocate the inference GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from vla_rynn_iql.pi05 import verify_runtime
    from vla_rynn_iql.pi05_assets import OPENPI_COMMIT, OFFICIAL_CHECKPOINT, NORM_FILE, IDENTITY_FILE
    from vla_rynn_iql.io import atomic_json, sha256_file
    verify_runtime()
    import openpi
    from openpi.shared.download import maybe_download
    source = Path(maybe_download(OFFICIAL_CHECKPOINT))
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05-convert-", dir=output.parent))
    root = Path(openpi.__file__).resolve().parents[2]
    print(f"Converting {source} → {output}; this requires substantial CPU RAM and disk space.", flush=True)
    try:
        subprocess.run([sys.executable, str(root / "examples/convert_jax_model_to_pytorch.py"),
                        "--checkpoint-dir", str(source), "--config-name", "pi05_libero",
                        "--output-path", str(temporary), "--precision", args.precision], check=True)
        # The upstream converter searches the parent for assets; pretrained releases own their assets.
        norm = temporary / NORM_FILE
        norm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / NORM_FILE, norm)
        identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                    "source": OFFICIAL_CHECKPOINT, "config_name": "pi05_libero",
                    "conversion_precision": args.precision,
                    "native_action_horizon": 10, "native_action_dim": 32, "discrete_state_input": False,
                    "files": {name: sha256_file(temporary / name) for name in ("model.safetensors", NORM_FILE)}}
        atomic_json(temporary / IDENTITY_FILE, identity)
        temporary.rename(output)
    except BaseException:
        print(f"Conversion failed; partial files retained for diagnosis at {temporary}", file=sys.stderr)
        raise
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
