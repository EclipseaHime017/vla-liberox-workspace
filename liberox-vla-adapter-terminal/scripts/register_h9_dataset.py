#!/usr/bin/env python3
"""Register existing H9 numerical outputs, optionally export through the official exporter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-version-id", required=True)
    parser.add_argument("--config-path", required=True, type=Path)
    for key in ("config-sha256", "prepared-sha256", "reward-sha256", "name"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--arm", required=True, choices=("T3-2", "T3-3", "T3-4"))
    parser.add_argument("--export-to", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=ROOT,
                        help="Existing compatible runtime; allows isolated new-script deployment")
    args = vars(parser.parse_args())
    destination = args.pop("export_to")
    runtime = args.pop("runtime_root").absolute()
    sys.path[:0] = [str(runtime / "liberox-vla-adapter-terminal"),
                   str(runtime / "liberox-vla-adapter-terminal/scripts"),
                   str(runtime / "vla-adapter-rynn-iql/src")]
    spec = importlib.util.spec_from_file_location("backend.app.services.h9_reward_dataset",
        ROOT / "liberox-vla-adapter-terminal/backend/app/services/h9_reward_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from backend.app.services.dataset_transfer import export_dataset
    if destination is not None and (destination.exists() or destination.is_symlink()):
        parser.error("Export destination must be new")
    result = module.register_h9_dataset(**args)
    # Print registration receipt before export, so export errors never lose the ID.
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if destination is not None:
        output = export_dataset(Path(result["dataset_manifest"]).parents[2], result["dataset_id"],
            destination, Path(result["version"]["config_path"]), runtime / "vla-adapter-rynn-iql")
        print(json.dumps({"export_path": str(output)}, ensure_ascii=False))
