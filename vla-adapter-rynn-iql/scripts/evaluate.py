#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_INFERENCE_CONFIG, load_inference_config
from vla_rynn_iql.evaluation import evaluate
from vla_rynn_iql.runtime import run_cuda_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a VLA-Adapter IQL policy overlay")
    parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_inference_config(args.config)
    print(run_cuda_stage("LIBERO-X overlay evaluation", lambda: evaluate(config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
