#!/usr/bin/env python3
"""torchrun worker for BC and IQL; legacy filename retained for deployment scripts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.server_config import distributed_config
from vla_rynn_iql.distributed_training import train_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    train_distributed(load_train_config(args.config), distributed_config(execution["distributed"]),
                      Path(execution["cache_root"]), Path(execution["run_dir"]))


if __name__ == "__main__":
    main()
