#!/usr/bin/env python3
"""Persistent FIFO server training; background workers survive SSH disconnects."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.server_queue import DEFAULT_QUEUE, TrainingQueue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE, help="Persistent queue directory")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Validate and freeze one training request")
    add.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/server_pipeline.yaml")
    add.add_argument("--task")
    add.add_argument("--runs-root", type=Path)
    add.add_argument("--model")
    add.add_argument("--method", choices=("bc", "iql"))
    add.add_argument("--name")
    commands.add_parser("start", help="Start an independent background worker")
    run = commands.add_parser("run", help="Run the worker in this terminal")
    run.add_argument("--once", action="store_true", help="Exit after draining currently pending jobs")
    commands.add_parser("list", help="Show jobs without loading models or checking dataset hashes")
    commands.add_parser("stop", help="Finish the active job, then stop; keep pending jobs")
    for name in ("cancel", "retry"):
        item = commands.add_parser(name)
        item.add_argument("job_id")
    args = parser.parse_args(argv)
    queue = TrainingQueue(args.queue)
    try:
        if args.command == "add":
            from vla_rynn_iql.server_config import configure_server
            server = configure_server(args.config, task=args.task, runs_root=args.runs_root,
                                      model=args.model, method=args.method)
            result = queue.add(server, name=args.name)
        elif args.command == "start":
            result = queue.start()
        elif args.command == "run":
            queue.run(once=args.once)
            result = queue.list()
        elif args.command == "list":
            result = queue.list()
        elif args.command == "stop":
            queue.stop()
            result = {"status": "STOP_REQUESTED", "message": "Pending jobs stay queued; active training finishes"}
        elif args.command == "cancel":
            queue.cancel(args.job_id)
            result = queue.list()
        else:
            result = queue.retry(args.job_id)
    except (ValueError, OSError, StopIteration, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
