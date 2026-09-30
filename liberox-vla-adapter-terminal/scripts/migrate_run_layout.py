#!/usr/bin/env python3
"""Preview, apply or roll back the undated recording layout, without starting UI."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent / "vla-adapter-rynn-iql/src")]

from backend.app.core.config import DEFAULT_UI_CONFIG, load_ui_config
from vla_rynn_iql.run_layout import migrate_layout, plan_migration, rollback_migration


def assert_stopped(config):
    host = "127.0.0.1" if config.host == "0.0.0.0" else config.host
    try:
        with socket.create_connection((host, config.port), timeout=0.5):
            pass
    except ConnectionRefusedError:
        pass
    else:
        raise RuntimeError(f"UI port {config.port} is still listening; stop UI before migration")
    # Compatibility check for old CLI processes that do not hold storage leases.
    scripts = {"run_ui.py", "train_terminal.py", "prepare_dataset.py", "annotate_rewards.py",
               "train_iql.py", "evaluate_trajectories.py", "transfer_dataset.py"}
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not args or "python" not in Path(args[0]).name:
                continue
            if any(Path(arg).name in scripts for arg in args[1:] if arg):
                raise RuntimeError(f"Stop data consumer PID {proc.name} before migration")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Managed detached jobs are additionally checked by their recorded PIDs.
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-config", type=Path, default=DEFAULT_UI_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Move recordings and update catalog paths")
    mode.add_argument("--rollback", action="store_true", help="Roll back the latest migration if recordings are unchanged")
    mode.add_argument("--dry-run", action="store_true", help="Read-only plan (default)")
    args = parser.parse_args()
    try:
        config = load_ui_config(args.ui_config)
        if args.apply or args.rollback:
            assert_stopped(config)
        result = (rollback_migration(config.dataset_root) if args.rollback else
                  migrate_layout(config.dataset_root, config.project_id) if args.apply else
                  plan_migration(config.dataset_root, config.project_id))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"{result.get('status', 'DRY RUN')}: {len(result['moves'])} recordings; "
              f"{result.get('already_current', 0)} already undated; "
              f"{len(result.get('empty_date_directories', []))} initially empty date directories; "
              f"{len(result.get('removed_date_directories', []))} date directories removed; "
              f"{len(result.get('skipped', []))} skipped")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Migration refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
