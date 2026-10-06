#!/usr/bin/env python3
"""Install pinned OpenPI only inside a dedicated Python 3.11 Conda environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from vla_rynn_iql.pi05_assets import OPENPI_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=PROJECT.parent / "OpenPI")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).name != "pi05":
        raise SystemExit("Run this installer in a NEW dedicated Conda environment named pi05, with Python 3.11")
    root = args.openpi_root.expanduser().resolve()
    if not root.exists():
        subprocess.run(["git", "clone", "https://github.com/Physical-Intelligence/openpi.git", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", OPENPI_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENPI_COMMIT or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise SystemExit("OpenPI checkout is not clean and pinned; existing user changes will not be overwritten")
    subprocess.run(["git", "-C", str(root), "submodule", "update", "--init", "--recursive"], check=True)
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": sys.prefix, "UV_LINK_MODE": "copy",
                   "GIT_LFS_SKIP_SMUDGE": "1"}
    lock = tomllib.loads((root / "uv.lock").read_text())
    torch_package = next(package for package in lock["package"] if package["name"] == "torch")
    cuda_packages = [item["name"] for item in torch_package["dependencies"]
                     if item["name"].startswith("nvidia-") or item["name"] == "triton"]
    skip_cuda = [argument for name in ("torch", "torchvision", *cuda_packages)
                 for argument in ("--no-install-package", name)]
    # Use the upstream lock, not a second manually maintained dependency set.
    subprocess.run([sys.executable, "-m", "uv", "sync", "--frozen", "--inexact", "--no-default-groups",
                    *skip_cuda, "--project", str(root)],
                   env=environment, check=True)
    # The upstream lock's PyPI CUDA build lacks Blackwell support. Keep its Torch
    # version, but select the official cu128 wheels for both model and vision ops.
    wheel_dir = PROJECT / "outputs/pi05-install/wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-cache-dir", "--no-deps",
                    "torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "--dest", str(wheel_dir),
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    wheels = []
    for package in ("torch-2.7.1+cu128", "torchvision-0.22.1+cu128"):
        candidates = list(wheel_dir.glob(f"{package}-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {package} wheel in {wheel_dir}")
        wheels.append(str(candidates[0]))
    # CUDA runtime packages use the user's normal package index, rather than
    # downloading every transitive dependency from the model-wheel endpoint.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", *wheels], check=True)
    import importlib.util
    target = Path(importlib.util.find_spec("transformers").origin).parent
    for source in (root / "src/openpi/models_pytorch/transformers_replace").iterdir():
        if source.is_dir():
            shutil.copytree(source, target / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target / source.name)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "-e", str(PROJECT)], check=True)
    protobuf = next(package["version"] for package in lock["package"] if package["name"] == "protobuf")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "PyYAML>=6,<7",
                    "tensorboard==2.20.0", f"protobuf=={protobuf}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("π₀.₅ environment ready. Next run scripts/prepare_pi05.py to download and convert official weights.")


if __name__ == "__main__":
    main()
