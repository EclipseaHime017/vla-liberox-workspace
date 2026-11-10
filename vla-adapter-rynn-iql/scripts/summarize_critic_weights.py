"""Summarize a completed diagnostic without fitting or changing any inputs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.critic_weight_summary import summarize_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--probe-config", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize_files(load_train_config(args.training_config), load_train_config(args.probe_config),
                    args.result, args.output)


if __name__ == "__main__":
    main()
