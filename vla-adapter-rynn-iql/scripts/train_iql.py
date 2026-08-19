#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
from vla_rynn_iql.training import train
from vla_rynn_iql.runtime import run_cuda_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-train VLA-Adapter with RynnValue-shaped IQL")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(args.config)
    print(run_cuda_stage("VLA-Adapter IQL post-training", lambda: train(config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
