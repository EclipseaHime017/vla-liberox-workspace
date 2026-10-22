#!/usr/bin/env python3
"""Execute one queue-owned critic diagnostic phase without training an actor."""
import argparse
from pathlib import Path
import signal
import sys
import threading

PLATFORM = Path(__file__).resolve().parents[1]
for entry in (PLATFORM, PLATFORM / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from backend.app.tools.critic_runner import fit, load_spec, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("rollout", "fit"), required=True)
    args = parser.parse_args()
    stopped = threading.Event()
    def stop(_signum, _frame):
        stopped.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    spec = load_spec(args.config)
    print(f"Critic diagnostic phase: {args.phase}", flush=True)
    try:
        (rollout if args.phase == "rollout" else fit)(spec, args.config.resolve().parent,
                                                     stop_requested=stopped.is_set)
    except InterruptedError:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
