#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.monitoring import convert_run_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an existing IQL metrics.jsonl file to TensorBoard events"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing metrics.jsonl",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Destination (default: <run-dir>/tensorboard-imported)",
    )
    args = parser.parse_args()
    target, count = convert_run_metrics(args.run_dir, args.log_dir)
    print(f"Converted {count} metric records to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
