"""Fail-fast checks for the native Robometer PyTorch dependency matrix."""

from __future__ import annotations

from importlib import metadata

from core.upstream import upstream_lock

# The pinned matrix lives in third_party/lock.yaml (robometer.runtime).
EXPECTED = {name: str(upstream_lock("robometer")["runtime"][name])
            for name in ("torch", "torchvision", "torchao", "xformers", "transformers", "trl")}


def public_version(value: str) -> str:
    """Ignore CUDA local suffixes such as ``+cu128`` when comparing wheels."""
    return value.split("+", 1)[0]


def validate_runtime_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    problems: list[str] = []
    for package, expected in EXPECTED.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        found[package] = installed
        if public_version(installed) != expected:
            problems.append(f"{package}=={installed} (expected {expected})")
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "Incompatible Robometer runtime: " + details + ". Rebuild the "
            "robometer-reward environment (python envs/setup.py --role reward-robometer); "
            "downgrading torchao alone is not supported."
        )
    return found
