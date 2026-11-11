#!/usr/bin/env python3
"""Register P7 Rynn .99/W5-on inputs; queue Q/V-only 10k/b64/seed7 using the existing API."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    args = vars(parser.parse_args())
    runtime = args.pop("runtime_root").absolute()
    sys.path[:0] = [str(runtime / "liberox-vla-adapter-terminal"),
                   str(runtime / "liberox-vla-adapter-terminal/scripts"),
                   str(runtime / "vla-adapter-rynn-iql/src")]
    spec = importlib.util.spec_from_file_location("backend.app.services.p7_reward_dataset",
        ROOT / "liberox-vla-adapter-terminal/backend/app/services/p7_reward_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(module.derive_p7_dataset(**args), ensure_ascii=False, indent=2))
