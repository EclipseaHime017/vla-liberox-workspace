#!/usr/bin/env python3
"""Export old/current PC datasets to portable task-organized server folders."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG
from vla_rynn_iql.portable_dataset import read_bundle, task_slug


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Verify a copied bundle without loading models")
    check.add_argument("directory", type=Path)
    for command in ("export", "migrate"):
        item = commands.add_parser(command, help="Copy a frozen dataset" if command == "export" else
                                    "Copy all legacy/current frozen datasets; retain original files and catalogs")
        item.add_argument("--project-root", type=Path,
                          default=WORKSPACE / "dataset-root/projects/libero_x_vla")
        if command == "export":
            item.add_argument("--dataset", required=True, help="ds_... ID, as displayed in the UI")
        item.add_argument("--output", type=Path, default=WORKSPACE / "training-datasets")
        item.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
        item.add_argument("--require-reward", action="append", choices=["sparse", "stage", "rynnvalue", "final"], default=[])
        item.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "verify":
        bundle = read_bundle(args.directory, verify=True)
        print(f"Verified {bundle['dataset']['id']}: rewards={list(bundle['rewards'])}")
        return 0
    source_root = args.project_root.expanduser().resolve()
    if args.command == "export" and (Path(args.dataset).name != args.dataset or args.dataset in {".", ".."}):
        parser.error("--dataset must be an ID, not a path")
    paths = ([source_root / "datasets" / args.dataset / "dataset.json"] if args.command == "export"
             else sorted((source_root / "datasets").glob("*/dataset.json")))
    if not paths:
        parser.error("No frozen datasets found")
    for selection in paths:
        frozen = json.loads(selection.read_text())
        target = args.output.expanduser().resolve() / task_slug(frozen["task_id"]) / frozen["id"]
        print(f"{selection} -> {target}", flush=True)
        if target.exists():
            raise FileExistsError(f"Destination already exists; verify it or choose a new --output: {target}")
        if args.dry_run:
            continue
        # PC-only adapter imports storage/services, not FastAPI or simulation.
        sys.path[:0] = [str(WORKSPACE / "liberox-vla-adapter-terminal"),
                        str(WORKSPACE / "liberox-vla-adapter-terminal/scripts")]
        from backend.app.services.dataset_transfer import export_dataset
        export_dataset(source_root, frozen["id"], target, args.config.resolve(), PROJECT,
                       required_rewards=tuple(args.require_reward))
        print(f"Ready to copy: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
