#!/usr/bin/env python3
"""Validate the public LIBERO-X LeRobot v2.1 metadata and one Parquet shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SHAPES = {"image": [256, 256, 3], "wrist_image": [256, 256, 3], "state": [8], "actions": [7]}
EXPECTED_COLUMNS = {
    "image",
    "wrist_image",
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if info.get("codebase_version") != "v2.1":
        errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v2.1'")
    if info.get("robot_type") != "panda":
        errors.append(f"robot_type={info.get('robot_type')!r}, expected 'panda'")
    if info.get("fps") != 10:
        errors.append(f"fps={info.get('fps')!r}, expected 10")
    features = info.get("features", {})
    for name, shape in EXPECTED_SHAPES.items():
        actual = features.get(name, {}).get("shape")
        if actual != shape:
            errors.append(f"feature {name!r} shape={actual!r}, expected {shape!r}")

    parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        errors.append("no Parquet episode found under data/chunk-*/episode_*.parquet")
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Install pyarrow to inspect Parquet: pip install pyarrow") from exc
        schema = pq.read_schema(parquet_files[0])
        missing = sorted(EXPECTED_COLUMNS - set(schema.names))
        if missing:
            errors.append(f"Parquet schema missing columns: {missing}")

    print(f"dataset_root: {root}")
    print(f"episodes: {info.get('total_episodes')}")
    print(f"frames: {info.get('total_frames')}")
    print(f"tasks: {info.get('total_tasks')}")
    print(f"parquet_files_found: {len(parquet_files)}")
    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED: LIBERO-X LeRobot schema matches the VLA adapter contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

