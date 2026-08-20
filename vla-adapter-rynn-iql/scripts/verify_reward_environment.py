#!/usr/bin/env python3
"""Verify the local official RynnValue checkout before model execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the pinned local RynnValue checkout and model revision"
    )
    parser.add_argument(
        "--checkout", type=Path, default=ROOT.parent / "RynnValue",
        help="Official RynnValue source checkout (default: workspace/RynnValue)",
    )
    args = parser.parse_args()
    lock = yaml.safe_load(
        (ROOT / "configs" / "dependency-lock.yaml").read_text(encoding="utf-8")
    )["rynnvalue"]
    checkout = args.checkout.expanduser().resolve()
    package_init = checkout / "rynn_value" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"Invalid RynnValue checkout; missing {package_init}")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != lock["git_commit"]:
        raise RuntimeError(
            f"RynnValue checkout mismatch: expected {lock['git_commit']}, got {commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"RynnValue checkout has local modifications:\n{dirty}")
    sys.path.insert(0, str(checkout))
    import rynn_value

    imported_checkout = Path(rynn_value.__file__).resolve().parents[1]
    if imported_checkout != checkout:
        raise RuntimeError(
            f"Imported RynnValue from {imported_checkout}, expected {checkout}"
        )
    print(f"RynnValue checkout verified: {commit}")
    print(f"RynnValue import path: {rynn_value.__file__}")
    print(f"Pinned model revision: {lock['hf_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
