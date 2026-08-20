#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(environment: str, script: str, config: Path) -> None:
    subprocess.run([
        "conda", "run", "--no-capture-output", "-n", environment,
        "python", str(ROOT / "scripts" / script), "--config", str(config),
    ], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the two-environment offline IQL pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "liberox_iql.yaml")
    parser.add_argument("--inference-config", type=Path, default=ROOT / "configs" / "inference.yaml")
    parser.add_argument("--reward-env", default="rynnvalue-reward")
    parser.add_argument("--train-env", default="vla-liberox")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    run(args.train_env, "prepare_dataset.py", args.config.resolve())
    run(args.reward_env, "annotate_rewards.py", args.config.resolve())
    run(args.train_env, "train_iql.py", args.config.resolve())
    if not args.skip_evaluation:
        run(args.train_env, "evaluate.py", args.inference_config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
