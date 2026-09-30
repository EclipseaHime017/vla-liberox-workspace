"""Resolve explicitly relocated recordings without changing historical manifests.

This module is stdlib-only and is also loaded by the UI and Robometer. Mapping
discovery is cached; restart consumers after performing storage maintenance.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from contextlib import contextmanager, ExitStack
from functools import lru_cache, wraps
from pathlib import Path

REGISTRY = ".run-layout.json"
JOURNAL = ".run-layout-migration.json"
LOCK = ".storage-maintenance.lock"


@lru_cache(maxsize=8192)
def _registry_for(directory: Path) -> Path | None:
    return next((parent / REGISTRY for parent in (directory, *directory.parents)
                 if (parent / REGISTRY).is_file()), None)


@lru_cache(maxsize=32)
def _mappings(registry: Path, signature: tuple) -> tuple:
    if registry.is_symlink():
        raise ValueError(f"Storage registry must not be a symlink: {registry}")
    payload = json.loads(registry.read_text())
    pairs = validate_registry(registry.parent, payload)
    return dict(pairs), {new: old for old, new in pairs}


def validate_registry(root: Path, payload: dict) -> tuple:
    registry = root / REGISTRY
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported storage mapping: {registry}")
    pairs = []
    for item in payload["moves"]:
        pair = []
        for key in ("old", "new"):
            relative = Path(item[key])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"Unsafe storage mapping: {registry}")
            path = root / relative
            if path.resolve() != path:
                raise ValueError(f"Storage mapping traverses a symlink: {path}")
            pair.append(path)
        pairs.append(tuple(pair))
    old, new = zip(*pairs) if pairs else ((), ())
    if len(set(old)) != len(old) or len(set(new)) != len(new):
        raise ValueError(f"Ambiguous storage mapping: {registry}")
    for left in old:
        if any(left == right or left in right.parents or right in left.parents for right in new):
            raise ValueError(f"Chained storage mapping: {registry}")
    return tuple(pairs)


def storage_path(value: str | Path, *, original: bool = False) -> Path:
    """Resolve at the IO boundary only; never rewrite hash-bearing metadata."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    registry = _registry_for(path.parent)
    if registry is None:
        return path
    stat = registry.stat()
    index = _mappings(registry, (stat.st_ino, stat.st_size, stat.st_mtime_ns))[int(original)]
    for source in (path, *path.parents):
        if source in index:
            return index[source] / path.relative_to(source)
    return path


def clear_storage_cache() -> None:
    # Lightweight consumers can import this file under an isolated module name.
    for module in tuple(sys.modules.values()):
        if getattr(module, "__file__", None) == __file__:
            module._registry_for.cache_clear()
            module._mappings.cache_clear()


def with_dataset_lease(function):
    """Protect standalone data consumers as well as UI-spawned jobs."""
    @wraps(function)
    def guarded(config, *args, **kwargs):
        raw = getattr(config, "raw", config)
        paths = raw.get("paths", {})
        candidates = [*paths.get("dataset_sources", []), paths.get("work_dir"),
                      paths.get("selection_manifest"), raw.get("data", {}).get("selection_manifest")]
        roots = set()
        for value in candidates:
            if not value:
                continue
            path = Path(value).expanduser().absolute()
            for parent in (path, *path.parents):
                if (parent / "projects").is_dir() or (parent / REGISTRY).is_file():
                    roots.add(parent)
                    break
        with ExitStack() as stack:
            for root in sorted(roots):
                stack.enter_context(storage_lease(root))
            clear_storage_cache()
            return function(config, *args, **kwargs)
    return guarded


@contextmanager
def storage_lease(root: Path | None, *, exclusive: bool = False):
    """Nonblocking lifetime lease; maintenance must not race readers or writers."""
    if root is None:
        yield
        return
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in (LOCK, JOURNAL, REGISTRY, "catalog.sqlite3", ".layout-history"):
        if (root / name).is_symlink():
            raise ValueError(f"Storage control path must not be a symlink: {root / name}")
    with (root / LOCK).open("a+") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Storage is busy; stop the UI and all dataset/training jobs before migration") from exc
        try:
            journal = root / JOURNAL
            if not exclusive and journal.is_file():
                state = json.loads(journal.read_text())["status"]
                if state not in {"COMPLETED", "ROLLED_BACK"}:
                    raise RuntimeError("Incomplete storage migration; run migrate_run_layout.py --rollback first")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
