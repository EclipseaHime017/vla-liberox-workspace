import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from vla_rynn_iql import run_layout
from vla_rynn_iql.data import load_manifest, prepare_dataset, iter_episode_arrays
from vla_rynn_iql.io import atomic_json, sha256_file, stable_hash
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index, reward_manifest_digest
from vla_rynn_iql.storage_paths import (
    JOURNAL, REGISTRY, clear_storage_cache, storage_lease, storage_path,
)
from test_replay import FakeAnnotator, _stats


@pytest.fixture(autouse=True)
def paths_cache():
    clear_storage_cache()
    yield
    clear_storage_cache()


def source(configured):
    return Path(configured.raw["paths"]["dataset_sources"][0])


@pytest.mark.parametrize("reward_source", ["rynnvalue", "final"])
def test_migration_preserves_bytes_replay_and_legacy_reward_identity(configured, reward_source):
    configured.raw["reward"].update(source=reward_source, alpha=0.0)
    root = source(configured)
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    prepared = load_manifest(configured)
    index = load_reward_index(configured)
    digest = reward_manifest_digest(index)
    before = {p: p.read_bytes() for p in (root / "projects").rglob("*") if p.is_file()}
    arrays = list(iter_episode_arrays(prepared))
    replay = ReplayDataset(configured, _stats(7), _stats(8), reward_index=index)
    sample = replay[0]
    # A negative discovery cache must not hide the mapping after apply.
    assert storage_path(next(iter(before))) == next(iter(before))
    result = run_layout.migrate_layout(root, "libero_x_vla")
    assert result["status"] == "COMPLETED" and len(result["moves"]) == 2
    for old, content in before.items():
        assert not old.exists()
        new = storage_path(old)
        assert new.read_bytes() == content and not new.is_symlink()
        assert storage_path(new, original=True) == old
    assert reward_manifest_digest(index) == digest
    assert load_manifest(configured) == prepared
    assert reward_manifest_digest(load_reward_index(configured)) == digest
    assert all(not (root / move["old"]).parent.exists() for move in result["moves"])
    after_arrays = list(iter_episode_arrays(prepared))
    for (_, old, _), (_, new, _) in zip(arrays, after_arrays):
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
    after = ReplayDataset(configured, _stats(7), _stats(8), reward_index=index)[0]
    for key in sample:
        if hasattr(sample[key], "shape"):
            np.testing.assert_array_equal(sample[key], after[key])
    assert run_layout.migrate_layout(root, "libero_x_vla")["status"] == "UNCHANGED"
    assert run_layout.rollback_migration(root)["status"] == "ROLLED_BACK"
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (root / REGISTRY).exists()


def test_dry_run_and_collision_are_read_only(configured):
    root = source(configured)
    before = {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*")}
    plan = run_layout.plan_migration(root, "libero_x_vla")
    assert len(plan["moves"]) == 2
    assert before == {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*")}
    (root / plan["moves"][0]["new"]).mkdir()
    with pytest.raises(FileExistsError):
        run_layout.migrate_layout(root, "libero_x_vla")
    assert all((root / item["old"]).exists() for item in plan["moves"])
    assert not (root / JOURNAL).exists()


def test_empty_date_cleanup_without_moves_preserves_registry_and_can_roll_back(configured):
    root = source(configured)
    first = run_layout.migrate_layout(root, "libero_x_vla")
    task = (root / first["moves"][0]["new"]).parent
    empty = task / "2026-01-02"
    empty.mkdir()
    registry_bytes = (root / REGISTRY).read_bytes()
    plan = run_layout.plan_migration(root, "libero_x_vla")
    assert not plan["moves"] and plan["layout"] == "mixed"
    assert plan["empty_date_directories"] == [str(empty.relative_to(root))]
    assert empty.exists()  # detection itself is read-only
    result = run_layout.migrate_layout(root, "libero_x_vla")
    assert result["status"] == "COMPLETED" and not result["moves"]
    assert result["removed_date_directories"] == plan["empty_date_directories"]
    assert not empty.exists() and (root / REGISTRY).read_bytes() == registry_bytes
    journal_bytes = (root / JOURNAL).read_bytes()
    noop = run_layout.migrate_layout(root, "libero_x_vla")
    assert noop["status"] == "UNCHANGED" and noop["layout"] == "undated"
    assert (root / JOURNAL).read_bytes() == journal_bytes
    run_layout.rollback_migration(root)
    assert empty.is_dir() and not list(empty.iterdir())
    assert all((root / move["new"]).is_dir() for move in first["moves"])
    assert (root / REGISTRY).read_bytes() == registry_bytes


def test_migration_cleans_preexisting_empty_dates_but_retains_unknown_contents(configured):
    root = source(configured)
    plan = run_layout.plan_migration(root, "libero_x_vla")
    assert plan["layout"] == "dated"
    dated = (root / plan["moves"][0]["old"]).parent
    extra = dated.with_name("2026-01-03")
    extra.mkdir()
    unknown = dated / "notes.txt"
    unknown.write_text("keep me")
    result = run_layout.migrate_layout(root, "libero_x_vla")
    assert str(extra.relative_to(root)) in result["removed_date_directories"]
    assert result["retained_date_directories"] == [str(dated.relative_to(root))]
    assert result["skipped"] == [str(unknown.relative_to(root))]
    assert unknown.read_text() == "keep me"
    journal_bytes = (root / JOURNAL).read_bytes()
    repeated = run_layout.migrate_layout(root, "libero_x_vla")
    assert repeated["status"] == "UNCHANGED"
    assert repeated["retained_date_directories"] == result["retained_date_directories"]
    assert (root / JOURNAL).read_bytes() == journal_bytes
    run_layout.rollback_migration(root)
    assert extra.is_dir() and unknown.read_text() == "keep me"
    assert all((root / move["old"]).is_dir() for move in result["moves"])


def test_cleanup_failure_restores_already_removed_empty_dates(tmp_path, monkeypatch):
    runs = tmp_path / "projects/test/runs/task"
    empty = [runs / "2026-01-01", runs / "2026-01-02"]
    for directory in empty:
        directory.mkdir(parents=True)
    original = Path.rmdir
    def fail_second(path):
        if path == empty[1]:
            raise PermissionError("injected cleanup failure")
        return original(path)
    monkeypatch.setattr(Path, "rmdir", fail_second)
    with pytest.raises(PermissionError, match="injected"):
        run_layout.migrate_layout(tmp_path, "test")
    assert all(directory.is_dir() for directory in empty)
    assert json.loads((tmp_path / JOURNAL).read_text())["status"] == "ROLLED_BACK"
    assert not (tmp_path / REGISTRY).exists()


@pytest.mark.parametrize("invalid", ["../2026-01-01", "/tmp/2026-01-01",
                                    "projects/test/runs/2026-01-01",
                                    "projects/test/runs/task/2026-01-01/nested/2026-01-02"])
def test_unsafe_cleanup_journal_rejected_before_rollback(tmp_path, invalid):
    journal = {"schema_version": 1, "root": str(tmp_path), "project_id": "test",
               "status": "APPLYING", "moves": [], "catalog_changes": [],
               "date_directories": [invalid], "previous_registry": None}
    atomic_json(tmp_path / JOURNAL, journal)
    before = (tmp_path / JOURNAL).read_bytes()
    with pytest.raises(ValueError, match="Unsafe"):
        run_layout.rollback_migration(tmp_path)
    assert (tmp_path / JOURNAL).read_bytes() == before


def test_empty_layout_is_a_noop(tmp_path):
    (tmp_path / "projects/test/runs").mkdir(parents=True)
    result = run_layout.migrate_layout(tmp_path, "test")
    assert result["status"] == "UNCHANGED" and result["layout"] == "empty"
    assert not (tmp_path / JOURNAL).exists()


def test_catalog_paths_change_transactionally_and_restore(configured):
    root = source(configured)
    old = next(root.rglob("run.json")).parent
    db_path = root / "catalog.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE runs(id TEXT PRIMARY KEY, run_path TEXT NOT NULL)")
        db.execute("INSERT INTO runs VALUES (?,?)", ("id", str(old)))
    run_layout.migrate_layout(root, "libero_x_vla")
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT run_path FROM runs").fetchone()[0] == str(storage_path(old))
    run_layout.rollback_migration(root)
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT run_path FROM runs").fetchone()[0] == str(old)


def test_rename_failure_rolls_back_and_does_not_publish_partial_mapping(configured, monkeypatch):
    root = source(configured)
    move = run_layout._move
    count = 0
    def fail_once(old, new):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected rename failure")
        move(old, new)
    monkeypatch.setattr(run_layout, "_move", fail_once)
    with pytest.raises(OSError, match="injected"):
        run_layout.migrate_layout(root, "libero_x_vla")
    assert json.loads((root / JOURNAL).read_text())["status"] == "ROLLED_BACK"
    assert len(list(root.glob("projects/*/runs/*/*/*/run.json"))) == 2
    assert not (root / REGISTRY).exists()


def test_storage_lock_pending_journal_and_changed_recording_block_operations(configured):
    root = source(configured)
    with storage_lease(root):
        with pytest.raises(RuntimeError, match="busy"):
            run_layout.migrate_layout(root, "libero_x_vla")
    result = run_layout.migrate_layout(root, "libero_x_vla")
    current = root / result["moves"][0]["new"] / "run.json"
    current.write_text(current.read_text() + " ")
    with pytest.raises(RuntimeError, match="changed"):
        run_layout.rollback_migration(root)
    atomic_json(root / JOURNAL, {**result, "status": "APPLYING"})
    with pytest.raises(RuntimeError, match="Incomplete"):
        with storage_lease(root):
            pytest.fail("incomplete migration must not be read")


def test_active_legacy_cpu_job_blocks_migration(configured):
    root = source(configured)
    atomic_json(root / "projects/libero_x_vla/jobs/job/job.json",
                {"status": "RUNNING", "requires_gpu": False, "pid": os.getpid()})
    with pytest.raises(RuntimeError, match="active job PID"):
        run_layout.migrate_layout(root, "libero_x_vla")
    assert not (root / JOURNAL).exists()


def test_legacy_standalone_process_blocks_migration(configured, monkeypatch):
    monkeypatch.setattr(run_layout, "_legacy_processes", lambda: iter(["12345"]))
    with pytest.raises(RuntimeError, match="PID 12345"):
        run_layout.migrate_layout(source(configured), "libero_x_vla")
    assert not (source(configured) / JOURNAL).exists()


def test_unsafe_registry_and_journal_rejected(configured):
    root = source(configured)
    old = next(root.rglob("run.json"))
    atomic_json(root / REGISTRY, {"schema_version": 1, "moves": [{"old": "../outside", "new": "foo"}]})
    with pytest.raises(ValueError, match="Unsafe"):
        storage_path(old)
    atomic_json(root / JOURNAL, {"schema_version": 1, "root": str(root), "project_id": "/tmp",
                              "status": "APPLYING", "moves": [], "catalog_changes": []})
    with pytest.raises(ValueError, match="Unsafe"):
        run_layout.rollback_migration(root)
