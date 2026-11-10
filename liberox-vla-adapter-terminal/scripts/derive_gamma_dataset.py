#!/usr/bin/env python3
"""Publish a NEW P4 gamma-only dataset using existing sealed rewards; never run a VLM."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "liberox-vla-adapter-terminal"), str(ROOT / "vla-adapter-rynn-iql/src")]
from backend.app.services.derived_reward_dataset import derive_gamma_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gamma", type=float, default=.99)
    print(json.dumps(derive_gamma_dataset(**vars(parser.parse_args())), indent=2, ensure_ascii=False))
