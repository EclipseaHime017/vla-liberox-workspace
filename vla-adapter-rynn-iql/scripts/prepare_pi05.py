#!/usr/bin/env python3
"""Convert a registered π₀.₅ base checkpoint and bind its normalization identity."""
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
from vla_rynn_iql.base_models import BASE_MODELS, base_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", choices=tuple(key for key, base in BASE_MODELS.items() if base.family == "pi05"), default="pi05-libero-base")
    parser.add_argument("--output", type=Path, help="Defaults to the selected base model's weights directory")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Official converter output precision; float32 requires substantially more host RAM")
    args = parser.parse_args()
    # JAX is used for conversion only, not allowed to preallocate the inference GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from vla_rynn_iql.pi05 import verify_runtime
    from vla_rynn_iql.pi05_assets import OPENPI_COMMIT, IDENTITY_FILE
    from vla_rynn_iql.io import atomic_json, sha256_file
    from vla_rynn_iql.model_storage import download_repository
    verify_runtime()
    import openpi
    from openpi.shared.download import maybe_download
    base = base_model(args.base_model)
    output = (args.output or Path(base.checkpoint)).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing checkpoint: {output}")
    if base.revision is None:
        source = Path(maybe_download(base.source))
    else:
        source = download_repository(base.source, base.revision,
                                     allow_patterns=["params/**", base.norm_file])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05-convert-", dir=output.parent))
    root = Path(openpi.__file__).resolve().parents[2]
    print(f"Converting {source} → {output}; this requires substantial CPU RAM and disk space.", flush=True)
    try:
        subprocess.run([sys.executable, str(root / "examples/convert_jax_model_to_pytorch.py"),
                        "--checkpoint-dir", str(source), "--config-name", base.config_name,
                        "--output-path", str(temporary), "--precision", args.precision], check=True)
        # The upstream converter searches the parent for assets; pretrained releases own their assets.
        norm = temporary / base.norm_file
        norm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / base.norm_file, norm)
        identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                    "source": base.source, "config_name": base.config_name,
                    "conversion_precision": args.precision,
                    "native_action_horizon": base.contract["io"]["native_action_horizon"],
                    "native_action_dim": base.contract["io"]["padded_action_dim"],
                    "discrete_state_input": base.contract["architecture"]["discrete_state_input"],
                    "files": {name: sha256_file(temporary / name) for name in ("model.safetensors", base.norm_file)}}
        if base.revision is not None:
            identity["source_revision"] = base.revision
        atomic_json(temporary / IDENTITY_FILE, identity)
        temporary.rename(output)
    except BaseException:
        print(f"Conversion failed; partial files retained for diagnosis at {temporary}", file=sys.stderr)
        raise
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
