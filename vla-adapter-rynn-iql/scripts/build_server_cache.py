#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.server_config import load_server_config
from vla_rynn_iql.server_replay import build_replay_cache, validate_replay_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the server-only mmap replay cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = load_server_config(args.server_config)
    config = load_train_config(args.config)
    cache, reused = build_replay_cache(config, server.replay_cache, rebuild=args.rebuild)
    metadata = validate_replay_cache(cache, config)
    result = {
        "path": str(cache),
        "reused": reused,
        "item_count": int(metadata["item_count"]),
        "fingerprint": str(metadata["fingerprint"]),
    }
    if args.result_file is not None:
        atomic_json(args.result_file, result)
    print(f"SERVER CACHE {'reused' if reused else 'built'}: {cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
