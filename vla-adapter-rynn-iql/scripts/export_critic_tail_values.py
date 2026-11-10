"""Export failure endpoint V from an existing frozen diagnostic; never train."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.critic_tail_values import export_tail_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--critic-sha256", required=True, help="Independently recorded original critic.pt checksum")
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_tail_values(load_train_config(args.training_config), args.result, args.output,
                      critic_sha256=args.critic_sha256, split=args.split, device=args.device)


if __name__ == "__main__":
    main()
