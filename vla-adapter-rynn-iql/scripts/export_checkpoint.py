#!/usr/bin/env python3
"""Export a completed training checkpoint as a new evaluable policy directory."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.checkpoint_export import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
