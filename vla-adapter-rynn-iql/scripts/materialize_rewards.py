#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
from vla_rynn_iql.rewards import materialize_reward_manifest, load_annotation_index
from vla_rynn_iql.fusion_rewards import import_saved_model_inputs
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.data import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive independent Sparse, RynnValue or Stage-based IQL rewards"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--annotation-manifest", type=Path,
                        help="Use already evaluated official outputs; never runs a model")
    parser.add_argument("--model-inputs", type=Path, help="Dataset/global saved evaluation references")
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild matching deterministic reward artifacts",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(args.config)
    if args.annotation_manifest and args.model_inputs:
        raise ValueError("Choose only one model input reference")
    if args.model_inputs:
        import_saved_model_inputs(config, args.model_inputs)
    if args.annotation_manifest:
        incoming = json.loads(args.annotation_manifest.read_text())
        if incoming.get("dataset_sha256") != load_manifest(config)["dataset_sha256"]:
            raise ValueError("RynnValue inputs belong to a different prepared dataset")
        atomic_json(Path(config.section("paths")["work_dir"]) / "annotations" / "annotation_manifest.json", incoming)
        load_annotation_index(config)  # An All task must consume exactly its new model outputs.
    print(materialize_reward_manifest(config, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
