#!/usr/bin/env python3
"""Copy a trained model for deployment with an explicit base-weight location."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.model_artifacts import export_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    args = parser.parse_args()
    print(export_policy(args.policy, args.output, args.base_checkpoint))


if __name__ == "__main__":
    main()
