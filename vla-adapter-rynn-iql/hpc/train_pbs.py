#!/usr/bin/env python3
"""Prepare or submit parallel server experiments in one single-node PBS job."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from hpc.pbs import run_allocation, stage_submission, submit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "submit"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path,
                             default=Path(__file__).resolve().parent / "configs/pbs_pipeline.yaml")
        if name == "submit":
            command.add_argument("--afterok", help="Start only after this complete PBS job ID succeeds")
    run = commands.add_parser("run", help="Internal foreground entrypoint, invoked inside PBS")
    run.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_allocation(args.directory)
        directory, manifest = stage_submission(args.config)
        result = {"directory": str(directory), "experiments": manifest["experiments"]}
        if args.command == "submit":
            result["job_id"] = submit(directory, afterok=args.afterok)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
