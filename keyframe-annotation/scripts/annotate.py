#!/usr/bin/env python3
"""Standalone read-only proposal worker; no UI/robot/reward dependencies."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from keyframe_annotation.config import load_config
from keyframe_annotation.pipeline import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt("Annotation stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from keyframe_annotation.runtime import QwenAnnotator
    result = Experiment(load_config(args.config), json.loads(args.inputs.read_text()), args.output, QwenAnnotator).run()
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
