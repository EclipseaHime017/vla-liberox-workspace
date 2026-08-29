#!/usr/bin/env python3
"""Verify Robometer's pinned native runtime before loading the 4B model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_adapter_robometer.runtime import validate_runtime_versions


def main() -> int:
    versions = validate_runtime_versions()
    import torch
    import torchao
    import transformers
    import xformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in robometer-reward")
    left = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
    right = left @ left
    torch.cuda.synchronize()
    if not torch.isfinite(right).all():
        raise RuntimeError("CUDA BF16 smoke test produced a non-finite result")
    print("Robometer runtime is compatible")
    for name, version in versions.items():
        print(f"  {name}: {version}")
    print(f"  cuda: {torch.version.cuda}")
    print(f"  gpu: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
