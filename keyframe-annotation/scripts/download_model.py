#!/usr/bin/env python3
"""Explicit download only; normal annotation is local-files-only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from keyframe_annotation.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen3_vl.yaml")
args = parser.parse_args()
config = load_config(args.config)
from huggingface_hub import snapshot_download
print(snapshot_download(config.model_id, revision=config.revision, cache_dir=config.cache_dir,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*", "README.md"]))
