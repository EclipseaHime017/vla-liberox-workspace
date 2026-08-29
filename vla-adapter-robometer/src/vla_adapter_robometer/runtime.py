"""Fail-fast checks for the native Robometer PyTorch dependency matrix."""

from __future__ import annotations

from importlib import metadata


EXPECTED = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchao": "0.13.0",
    "xformers": "0.0.32.post2",
    "transformers": "4.57.1",
    "trl": "0.20.0",
}


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
            "robometer-reward environment using vla-adapter-robometer/README.md; "
            "downgrading torchao alone is not supported."
        )
    return found
