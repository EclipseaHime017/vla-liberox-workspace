#!/usr/bin/env python3
"""Evaluate exported checkpoints on LIBERO-X without the UI/backend."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Batch evaluation YAML")
    parser.add_argument("--check-render", action="store_true", help="Check real LIBERO-X rendering without loading a model")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from sim.runtime import run_worker
        return run_worker(args.worker)
    if args.config is None:
        parser.error("--config is required")
    from cluster.evaluate import load_evaluation, run_evaluations
    return run_evaluations(load_evaluation(args.config), check_render=args.check_render)


if __name__ == "__main__":
    raise SystemExit(main())
