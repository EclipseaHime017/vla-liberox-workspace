#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config, reward_source, needs_rynnvalue
from vla_rynn_iql.rewards import annotate_manifest
from vla_rynn_iql.runtime import run_cuda_stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache official frozen-RynnValue outputs for prepared trajectories"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute matching trajectory annotations instead of reusing the cache",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(args.config)
    source = reward_source(config.section("reward"))
    if not needs_rynnvalue(config.section("reward")):
        print(f"Skipped RynnValue annotation: reward.source={source}; "
              "use materialize_rewards.py to derive the selected rewards.")
        return 0
    print(run_cuda_stage(
        "RynnValue trajectory annotation",
        lambda: annotate_manifest(config, overwrite=args.overwrite),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
