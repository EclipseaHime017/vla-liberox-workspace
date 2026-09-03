#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
from vla_rynn_iql.rewards import materialize_reward_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive cached sparse/PBRS/final rewards from RynnValue annotations"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild matching deterministic reward artifacts",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(args.config)
    print(materialize_reward_manifest(config, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
