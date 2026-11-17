#!/usr/bin/env python3
"""Serve one checkpoint on one GPU through an SSH-forwarded loopback port."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from core.paths import THIRD_PARTY  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vla-root", type=Path, default=THIRD_PARTY / "VLA-Adapter")
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index; use only allocated GPUs")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 30 <= args.idle_seconds <= 86400 or args.gpu < 0:
        parser.error("Invalid port, GPU index or idle limit")
    gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allocated is not None and str(args.gpu) not in allocated.split(",") and gpu not in allocated.split(","):
        parser.error("Requested GPU is outside CUDA_VISIBLE_DEVICES allocation")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cpu", TOKENIZERS_PARALLELISM="false")
    from core.resources import TrainingLease
    from cluster.remote_inference import read_token, serve
    with TrainingLease(devices=[gpu]):
        serve(args.policy.resolve(), args.vla_root.resolve(), read_token(args.token_file), args.port, idle_seconds=args.idle_seconds)


if __name__ == "__main__":
    main()
