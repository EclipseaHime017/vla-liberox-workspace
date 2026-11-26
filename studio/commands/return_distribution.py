#!/usr/bin/env python3
"""Execute one queue-owned phase of the Value Flows return distribution tool."""
import argparse
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from studio.tools.return_distribution_runner import analyze, load_spec, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("rollout", "analyze"), required=True)
    args = parser.parse_args()
    stopped = threading.Event()
    def stop(_signum, _frame):
        stopped.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    spec = load_spec(args.config)
    print(f"Return distribution phase: {args.phase}", flush=True)
    try:
        (rollout if args.phase == "rollout" else analyze)(spec, args.config.resolve().parent,
                                                         stop_requested=stopped.is_set)
    except InterruptedError:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
