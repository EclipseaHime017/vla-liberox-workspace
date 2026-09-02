#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.distributed_training import (
    DistributedTrainingCancelled,
    train_distributed,
)
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.server_config import load_server_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Server-only DDP + ZeRO-1 IQL worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rank = int(os.environ.get("RANK", "0"))
    try:
        policy = train_distributed(
            load_train_config(args.config), load_server_config(args.server_config)
        )
    except DistributedTrainingCancelled as exc:
        if rank == 0:
            print(str(exc), file=sys.stderr, flush=True)
        return 130
    if rank == 0:
        if policy is None:
            raise RuntimeError("Rank zero completed without producing a policy overlay")
        result = {"status": "completed", "policy_overlay": str(policy)}
        if args.result_file is not None:
            atomic_json(args.result_file, result)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
