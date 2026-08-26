#!/usr/bin/env python3
"""Run one backend-generated, metrics-only LIBERO-X evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from backend.app.evaluation.batch import run_from_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stopped = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    result = run_from_path(args.config, stop_requested=stopped.is_set)
    return 0 if result["status"] == "COMPLETED" else 130 if result["status"] == "CANCELED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

