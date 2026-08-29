#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_adapter_robometer.config import DEFAULT_CONFIG, load_config
from vla_adapter_robometer.evaluation import evaluate_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_selection(load_config(args.config), overwrite=args.overwrite)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
