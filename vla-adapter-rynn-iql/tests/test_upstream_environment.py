import hashlib
import importlib.metadata
import subprocess

import pytest
import yaml

from vla_rynn_iql.upstream_environment import (
    DIFF_ARGS, verify_tensorflow_version, verify_vla_adapter_environment,
)


def versions(monkeypatch, installed):
    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(importlib.metadata, "version", version)


@pytest.mark.parametrize("installed", [
    {"tensorflow": "2.15.0"}, {"tensorflow-cpu": "2.15.0"}, {"tensorflow-intel": "2.15.0"},
    {"tensorflow": "2.15.0", "tensorflow-cpu": "2.15.0"},
])
def test_tensorflow_distribution_names_are_supported(monkeypatch, installed):
    versions(monkeypatch, installed)
    assert verify_tensorflow_version("2.15.0") == installed


@pytest.mark.parametrize("installed,message", [
    ({}, "not installed"), ({"tensorflow-cpu": "2.16.0"}, "version mismatch"),
    ({"tensorflow": "2.15.0", "tensorflow-cpu": "2.16.0"}, "version mismatch"),
])
def test_tensorflow_errors_explain_missing_or_wrong_versions(monkeypatch, installed, message):
    versions(monkeypatch, installed)
    with pytest.raises(RuntimeError, match=message):
        verify_tensorflow_version("2.15.0")


def git(root, *args):
    return subprocess.run(["git", "-c", "core.filemode=true", "-c", "core.quotePath=true",
                           "-C", str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    versions(monkeypatch, {"tensorflow-cpu": "2.15.0"})
    root = tmp_path / "upstream"
    root.mkdir()
    git(root, "init", "-q")
    source = root / "loader.py"
    source.write_text("loader\n")
    (root / "existing.py").write_text("preprocessing\n")
    generated = root / "pkg.egg-info/PKG-INFO"
    generated.parent.mkdir()
    generated.write_text("installed metadata\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "base")
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    source.write_text("registered local loader\n")
    patch = tmp_path / "loader.patch"
    patch.write_bytes(git(root, *DIFF_ARGS))
    lock = tmp_path / "dependency-lock.yaml"
    lock.write_text(yaml.safe_dump({"vla_adapter": {
        "tensorflow": "2.15.0", "git_commit": commit,
        "patch": {"path": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()},
    }}))
    return root, lock, patch


def test_recorded_patch_is_read_only_and_generated_files_are_ignored(checkout):
    root, lock, _ = checkout
    (root / "pkg.egg-info/PKG-INFO").write_text("regenerated metadata\n")
    generated = root / "__pycache__/loader.pyc"
    generated.parent.mkdir()
    generated.write_bytes(b"compiled")
    before = git(root, "status", "--porcelain", "--untracked-files=all")
    verify_vla_adapter_environment(root, lock_path=lock)
    assert git(root, "status", "--porcelain", "--untracked-files=all") == before


@pytest.mark.parametrize("changed,message", [
    ("clean", "source differs"), ("source", "source differs"), ("other_source", "source differs"),
    ("untracked", "unrecorded files"), ("patch", "patch hash mismatch"), ("commit", "commit mismatch"),
    ("staged", "source differs"),
    ("cache_source", "unrecorded files"), ("assume_unchanged", "index hides source"),
    ("skip_worktree", "index hides source"),
])
def test_unrecorded_checkout_changes_are_rejected(checkout, changed, message):
    root, lock, patch = checkout
    if changed == "clean":
        (root / "loader.py").write_text("loader\n")
    elif changed == "source":
        (root / "loader.py").write_text("unrecorded loader\n")
    elif changed == "other_source":
        (root / "existing.py").write_text("modified preprocessing\n")
    elif changed == "untracked":
        (root / "extra.py").write_text("extra code\n")
    elif changed == "patch":
        patch.write_bytes(patch.read_bytes() + b"\n")
    elif changed == "commit":
        raw = yaml.safe_load(lock.read_text())
        raw["vla_adapter"]["git_commit"] = "0" * 40
        lock.write_text(yaml.safe_dump(raw))
    elif changed == "staged":
        (root / "existing.py").write_text("staged preprocessing\n")
        git(root, "add", "existing.py")
    elif changed == "cache_source":
        directory = root / "__pycache__"
        directory.mkdir()
        (directory / "extra.py").write_text("unrecorded code\n")
    elif changed in ("assume_unchanged", "skip_worktree"):
        git(root, "update-index", "--" + changed.replace("_", "-"), "existing.py")
        (root / "existing.py").write_text("hidden preprocessing change\n")
    with pytest.raises(RuntimeError, match=message):
        verify_vla_adapter_environment(root, lock_path=lock)
