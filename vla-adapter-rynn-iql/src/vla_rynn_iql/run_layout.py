"""Offline, reversible migration of managed dated run directories.

Only directory entries and catalog path columns change. Historical manifests,
model inputs, sidecars and arrays are intentionally never rewritten.
"""
from __future__ import annotations

import json
import fcntl
import errno
import os
import re
import sqlite3
import uuid
from datetime import date, datetime, timezone
from contextlib import contextmanager, ExitStack
from pathlib import Path

from .io import atomic_json as _atomic_json, stable_hash
from .storage_paths import JOURNAL, REGISTRY, clear_storage_cache, storage_lease, validate_registry

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_COLUMNS = {"runs": ("run_path",), "training_datasets": ("manifest_path",),
            "offline_jobs": ("job_path",), "annotation_runs": ("manifest_path",),
            "training_runs": ("output_path", "overlay_path"), "evaluation_runs": ("result_path",)}


def _legacy_processes():
    """Conservative compatibility guard for consumers predating storage leases."""
    consumers = {"run_ui.py", "train.py", "train_iql.py", "train_terminal.py", "prepare_dataset.py",
                 "materialize_rewards.py", "annotate_rewards.py",
                 "evaluate_trajectories.py"}
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if args and "python" in Path(args[0]).name and any(
                Path(arg).name in consumers for arg in args[1:] if arg
            ):
                yield proc.name
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


@contextmanager
def _legacy_idle(root: Path):
    """Also protect deployments whose already-running workers predate the lease."""
    for pid in _legacy_processes():
        raise RuntimeError(f"Stop standalone dataset/training process PID {pid} before migration")
    with ExitStack() as stack:
        for project in (root / "projects").glob("*"):
            for name in (".gpu-task.lock", ".training-queue.lock"):
                path = project / name
                if not path.exists():
                    continue
                handle = stack.enter_context(path.open("a+"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"Active platform job: {path}") from exc
            for path in (project / "jobs").glob("*/job.json"):
                job = _read(path)
                if job.get("status") not in {"STARTING", "RUNNING", "STOPPING"}:
                    continue
                launcher = _read(path.with_name("launcher.json"), {})
                pids = {job.get("pid"), job.get("launcher_pid"), launcher.get("pid")} - {None}
                if not pids:
                    raise RuntimeError(f"Unresolved active job; stop/reconcile it before migration: {path}")
                for pid in pids:
                    if type(pid) is not int or pid <= 0:
                        raise RuntimeError(f"Invalid active job PID: {path}")
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        continue
                    except PermissionError as exc:
                        raise RuntimeError(f"Cannot verify active job PID {pid}") from exc
                    raise RuntimeError(f"Stop active job PID {pid} before migration: {path}")
        yield


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def atomic_json(path: Path, value):
    _atomic_json(path, value)
    _sync(path.parent)


def _inventory(directory: Path) -> str:
    """Cheap identity check: rename preserves file inodes, sizes and mtimes."""
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Run contains a symlink, resolve it before migration: {path}")
        if path.is_file():
            stat = path.stat()
            entries.append((str(path.relative_to(directory)), stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return stable_hash(entries)


def _catalog_changes(root: Path, moves: list[dict]) -> list[dict]:
    database = root / "catalog.sqlite3"
    if not database.is_file():
        return []
    changes = []
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, columns in _COLUMNS.items():
            if table not in tables:
                continue
            available = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
            for column in set(columns) & available:
                for row_id, value in db.execute(f'SELECT id, "{column}" FROM "{table}"'):
                    if not value:
                        continue
                    path = Path(value)
                    for move in moves:
                        old, new = root / move["old"], root / move["new"]
                        if path == old or old in path.parents:
                            changes.append({"table": table, "column": column, "id": row_id,
                                            "old": value, "new": str(new / path.relative_to(old))})
                            break
    return changes


def plan_migration(root: Path, project_id: str) -> dict:
    root = root.expanduser().resolve()
    if not project_id or Path(project_id).name != project_id or project_id in {".", ".."}:
        raise ValueError("project_id must be a single directory name")
    runs = root / "projects" / project_id / "runs"
    if not runs.is_dir():
        raise FileNotFoundError(runs)
    if runs.resolve() != runs:
        raise ValueError("Managed run root must not traverse symlinks")
    previous = _read(root / JOURNAL)
    if previous and previous["status"] not in {"COMPLETED", "ROLLED_BACK"}:
        raise RuntimeError("Unfinished migration; use --rollback before preparing a new plan")
    registry = root / REGISTRY
    if registry.exists():
        if registry.is_symlink():
            raise ValueError("Storage registry must not be a symlink")
        validate_registry(root, _read(registry))
    moves, skipped, current = [], [], 0
    date_directories, empty_date_directories = [], []
    for task in sorted(runs.iterdir()):
        if not task.is_dir() or task.is_symlink():
            continue
        for directory in sorted(task.iterdir()):
            if directory.is_symlink():
                raise ValueError(f"Unexpected symlink: {directory}")
            if not directory.is_dir():
                continue
            if not _DATE.fullmatch(directory.name):
                current += int((directory / "run.json").is_file())
                continue
            date.fromisoformat(directory.name)
            relative = str(directory.relative_to(root))
            date_directories.append(relative)
            children = sorted(directory.iterdir())
            if not children:
                empty_date_directories.append(relative)
            for old in children:
                if old.is_symlink():
                    raise ValueError(f"Unexpected symlink: {old}")
                manifest = old / "run.json"
                if not old.is_dir() or not manifest.is_file():
                    skipped.append(str(old.relative_to(root)))
                    continue
                payload = _read(manifest)
                if not isinstance(payload.get("id"), str) or not payload["id"]:
                    raise ValueError(f"Missing managed run identity: {manifest}")
                new = task / old.name
                if new.exists() or new.is_symlink():
                    raise FileExistsError(f"Migration target already exists; nothing was changed: {new}")
                moves.append({"old": str(old.relative_to(root)), "new": str(new.relative_to(root)),
                              "run_id": payload["id"], "inventory": _inventory(old)})
    targets = [move["new"] for move in moves]
    if len(set(targets)) != len(targets):
        raise ValueError("Two dated recordings have the same target directory; refusing to merge")
    return {"schema_version": 1, "root": str(root), "project_id": project_id,
            "moves": moves, "already_current": current, "skipped": skipped,
            "layout": ("mixed" if current else "dated") if date_directories else ("undated" if current else "empty"),
            "date_directories": date_directories, "empty_date_directories": empty_date_directories,
            "catalog_changes": _catalog_changes(root, moves)}


def _catalog_apply(root: Path, changes: list[dict], *, reverse: bool = False):
    if not changes:
        return
    database = root / "catalog.sqlite3"
    with sqlite3.connect(f"{database.as_uri()}?mode=rw", uri=True) as db:
        db.execute("BEGIN IMMEDIATE")
        for change in changes:
            table, column = change["table"], change["column"]
            if column not in _COLUMNS.get(table, ()):
                raise ValueError("Invalid catalog change in migration journal")
            source, target = (change["new"], change["old"]) if reverse else (change["old"], change["new"])
            row = db.execute(f'SELECT "{column}" FROM "{table}" WHERE id=?', (change["id"],)).fetchone()
            if row is None or row[0] not in {source, target}:
                raise RuntimeError(f"Catalog entry changed since migration: {table}/{change['id']}")
            db.execute(f'UPDATE "{table}" SET "{column}"=? WHERE id=?', (target, change["id"]))


def _sync(directory: Path):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _move(source: Path, destination: Path):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    source.rename(destination)
    _sync(source.parent)
    _sync(destination.parent)


def _journal_paths(root: Path, journal: dict):
    if journal.get("root") != str(root) or journal.get("schema_version") != 1:
        raise ValueError("Migration journal root/schema mismatch")
    project = journal["project_id"]
    if not project or Path(project).name != project or project in {".", ".."}:
        raise ValueError("Unsafe migration journal project")
    runs = root / "projects" / project / "runs"
    for move in journal["moves"]:
        if any(Path(move[key]).is_absolute() or ".." in Path(move[key]).parts for key in ("old", "new")):
            raise ValueError("Unsafe migration journal path")
        old, new = root / move["old"], root / move["new"]
        if (old.resolve() != old or new.resolve() != new or not old.is_relative_to(runs)
                or not new.is_relative_to(runs) or old.parent.parent != new.parent
                or old.name != new.name or not _DATE.fullmatch(old.parent.name)):
            raise ValueError("Unsafe migration journal path")
        yield old, new, move


def _journal_date_directories(root: Path, journal: dict) -> list[Path]:
    """Validate the narrow cleanup scope before deleting or restoring directories."""
    runs = root / "projects" / journal["project_id"] / "runs"
    directories = []
    for name in journal.get("date_directories", []):
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.resolve() != path
                or not path.is_relative_to(runs) or len(path.relative_to(runs).parts) != 2
                or not _DATE.fullmatch(path.name) or (path.exists() and not path.is_dir())):
            raise ValueError("Unsafe migration journal date directory")
        date.fromisoformat(path.name)
        directories.append(path)
    return directories


def _rollback(root: Path, journal: dict):
    paths = list(_journal_paths(root, journal))
    date_directories = _journal_date_directories(root, journal)
    # Validate every entry before reversing anything. Do not overwrite later work.
    for old, new, move in paths:
        if old.exists() == new.exists():
            raise RuntimeError(f"Ambiguous/missing recording during rollback: {old}")
        actual = old if old.exists() else new
        if _inventory(actual) != move["inventory"]:
            raise RuntimeError(f"Recording changed after migration; rollback refused: {actual}")
    journal["status"] = "ROLLING_BACK"
    atomic_json(root / JOURNAL, journal)
    _catalog_apply(root, journal["catalog_changes"], reverse=True)
    for directory in date_directories:
        if not directory.exists():
            directory.mkdir()
            _sync(directory.parent)
    for old, new, _ in reversed(paths):
        if new.exists():
            old.parent.mkdir(exist_ok=True)
            _move(new, old)
    if journal["previous_registry"] is None:
        (root / REGISTRY).unlink(missing_ok=True)
        _sync(root)
    else:
        atomic_json(root / REGISTRY, journal["previous_registry"])
    journal["status"] = "ROLLED_BACK"
    atomic_json(root / JOURNAL, journal)
    clear_storage_cache()


def rollback_migration(root: Path) -> dict:
    root = root.expanduser().resolve()
    with storage_lease(root, exclusive=True), _legacy_idle(root):
        journal = _read(root / JOURNAL)
        if not journal:
            raise ValueError("No migration journal found")
        if journal["status"] != "ROLLED_BACK":
            _rollback(root, journal)
        return journal


def migrate_layout(root: Path, project_id: str) -> dict:
    root = root.expanduser().resolve()
    with storage_lease(root, exclusive=True), _legacy_idle(root):
        plan = plan_migration(root, project_id)
        if not plan["moves"] and not plan["empty_date_directories"]:
            return {**plan, "status": "UNCHANGED", "removed_date_directories": [],
                    "retained_date_directories": plan["date_directories"]}
        journal = {**plan, "id": uuid.uuid4().hex, "status": "APPLYING",
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "removed_date_directories": [], "retained_date_directories": [],
                   "previous_registry": _read(root / REGISTRY)}
        prior = journal["previous_registry"] or {"schema_version": 1, "moves": []}
        registry = {"schema_version": 1, "moves": [*prior["moves"], *plan["moves"]]}
        validate_registry(root, registry)
        # Keep previous completed journals for diagnostics; rollback targets latest only.
        previous = _read(root / JOURNAL)
        if previous:
            if not re.fullmatch(r"[0-9a-f]{32}", previous.get("id", "")):
                raise ValueError("Invalid previous migration journal ID")
            atomic_json(root / ".layout-history" / f"{previous['id']}.json", previous)
        atomic_json(root / JOURNAL, journal)
        try:
            for old, new, _ in _journal_paths(root, journal):
                _move(old, new)
            if plan["moves"]:
                atomic_json(root / REGISTRY, registry)
            _catalog_apply(root, plan["catalog_changes"])
            for directory in _journal_date_directories(root, journal):
                try:
                    directory.rmdir()
                except OSError as exc:
                    if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                        raise
                    journal["retained_date_directories"].append(str(directory.relative_to(root)))
                else:
                    journal["removed_date_directories"].append(str(directory.relative_to(root)))
                    _sync(directory.parent)
            journal["status"] = "COMPLETED"
            atomic_json(root / JOURNAL, journal)
            clear_storage_cache()
            return journal
        except BaseException:
            # Failed rollback deliberately leaves a non-final journal blocking consumers.
            _rollback(root, journal)
            raise
