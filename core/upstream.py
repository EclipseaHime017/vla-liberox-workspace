"""Pinned upstreams (third_party/lock.yaml) and read-only checks of their local checkouts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess

import yaml

from core.paths import THIRD_PARTY

LOCK_PATH = THIRD_PARTY / "lock.yaml"
TF_DISTRIBUTIONS = ("tensorflow", "tensorflow-cpu", "tensorflow-intel", "tensorflow-macos")
SOURCE_PATHS = (".", ":(glob,exclude)**/*.pyc",
                ":(glob,exclude)**/*.pyo", ":(glob,exclude)**/*.egg-info/**")
DIFF_ARGS = ("diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-renames",
             "--unified=3", "--diff-algorithm=myers", "--no-indent-heuristic", "--no-color",
             "--inter-hunk-context=0", "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--", *SOURCE_PATHS)


def upstream_lock(name: str, *, lock_path: Path = LOCK_PATH) -> dict:
    """One upstream's pin: repository, commit, checkout and optional patch, weights, packages."""
    return yaml.safe_load(lock_path.read_text(encoding="utf-8"))[name]


def checkout_path(name: str) -> Path:
    """Default local checkout of an upstream, third_party/<checkout>/."""
    return THIRD_PARTY / upstream_lock(name)["checkout"]


def verify_tensorflow_version(expected: str) -> dict[str, str]:
    installed = {}
    for name in TF_DISTRIBUTIONS:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if not installed:
        raise RuntimeError(f"TensorFlow is not installed; install tensorflow=={expected} or its CPU equivalent")
    if any(version != expected for version in installed.values()):
        raise RuntimeError(f"TensorFlow version mismatch: expected {expected}, installed {installed}")
    return installed


def _git(checkout: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-pager", "-c", "core.filemode=true", "-c", "core.quotePath=true",
         "-c", "diff.suppressBlankEmpty=false",
         "-C", str(checkout), *args],
        check=True, capture_output=True, timeout=30,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return result.stdout


def verify_vla_adapter_environment(checkout: Path, *, lock_path: Path = LOCK_PATH) -> None:
    lock = upstream_lock("vla_adapter", lock_path=lock_path)
    verify_tensorflow_version(lock["packages"]["tensorflow"])
    commit = _git(checkout, "rev-parse", "HEAD").decode().strip()
    if commit != lock["commit"]:
        raise RuntimeError(f"VLA-Adapter commit mismatch: expected {lock['commit']}, found {commit}")
    patch = (lock_path.parent / lock["patch"]["path"]).resolve()
    content = patch.read_bytes()
    if hashlib.sha256(content).hexdigest() != lock["patch"]["sha256"]:
        raise RuntimeError(f"VLA-Adapter patch hash mismatch: {patch}")
    indexed = _git(checkout, "ls-files", "-v", "-z", "--", *SOURCE_PATHS).split(b"\0")
    hidden = [item[2:].decode() for item in indexed if item and (item[:1].islower() or item[:1] == b"S")]
    if hidden:
        raise RuntimeError(f"VLA-Adapter index hides source changes; clear assume-unchanged/skip-worktree: {hidden[:20]}")
    if _git(checkout, *DIFF_ARGS) != content:
        raise RuntimeError("VLA-Adapter source differs from the locked patch; apply the recorded patch "
                           "to the pinned commit and remove unrecorded source changes")
    untracked = _git(checkout, "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS)
    if untracked:
        paths = untracked.decode().strip("\0").split("\0")
        raise RuntimeError(f"VLA-Adapter has unrecorded files: {paths[:20]}")
