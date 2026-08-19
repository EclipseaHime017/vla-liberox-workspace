#!/usr/bin/env python3
"""Verify the local official RynnValue checkout before model execution."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the pinned local RynnValue checkout and model revision"
    )
    parser.parse_args()
    lock = yaml.safe_load(
        (ROOT / "configs" / "dependency-lock.yaml").read_text(encoding="utf-8")
    )["rynnvalue"]
    import rynn_value

    checkout = Path(rynn_value.__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != lock["git_commit"]:
        raise RuntimeError(
            f"RynnValue checkout mismatch: expected {lock['git_commit']}, got {commit}"
        )
    print(f"RynnValue checkout verified: {commit}")
    print(f"Pinned model revision: {lock['hf_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
