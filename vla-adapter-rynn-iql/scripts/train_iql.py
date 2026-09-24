#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
from vla_rynn_iql.training import TrainingCancelled, request_training_stop, train
from vla_rynn_iql.runtime import run_cuda_stage
from vla_rynn_iql.methods import training_method


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-train VLA-Adapter with IQL or behavior cloning")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--result-file", type=Path,
        help="Optionally write the completed policy overlay path as an atomic JSON result",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(args.config)
    signal.signal(signal.SIGTERM, lambda *_: request_training_stop())
    signal.signal(signal.SIGINT, lambda *_: request_training_stop())
    try:
        policy = run_cuda_stage(f"VLA-Adapter {training_method(config.raw).name.upper()} post-training", lambda: train(config))
    except TrainingCancelled:
        return 130
    print(policy)
    if args.result_file is not None:
        result = args.result_file.expanduser().resolve()
        result.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{result.name}.", dir=result.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"policy_overlay": str(Path(policy).resolve())}, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, result)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
