#!/usr/bin/env python3
"""Read-only T2-D final critic census; run once per final seed checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rynn_iql.checkpoint_critic import export_critic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared-manifest", "reward-manifest", "replay-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    result = export_critic(**vars(parser.parse_args()))
    print(f"Final step {result['step']}; {result['count']} unique train chunks; {result['groups']}")


if __name__ == "__main__":
    main()
