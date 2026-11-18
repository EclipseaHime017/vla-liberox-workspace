#!/usr/bin/env python3
"""Install what a device role needs: conda environments, pinned upstreams and weights.

    python envs/setup.py --list
    python envs/setup.py --role studio --role teleop --dry-run
    python envs/setup.py --role train --family pi05

Roles, units and steps are declared in envs/roles.yaml, upstream pins in
third_party/lock.yaml. Runs from any Python that has PyYAML or ruamel.yaml (conda's
base environment has the latter) and imports nothing from this repository, which
may not be installed yet. Existing checkouts are verified, never reset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
ROLES = ROOT / "envs" / "roles.yaml"
LOCK = THIRD_PARTY / "lock.yaml"
PACKAGES = ("core", "sim", "rewards", "train", "cluster", "studio")
DEFAULT_FAMILY = "vla_adapter"
# Installed environments run this with `python -I`, so neither the working directory nor
# PYTHONPATH can mask the install. From-root environments run it with `python -E` in the
# repository root, the way the platform starts them.
VERIFY_PACKAGES = (
    "import importlib.util, pathlib, sys\n"
    "root = pathlib.Path(sys.argv[1])\n"
    f"names = {PACKAGES!r}\n"
    "specs = {name: importlib.util.find_spec(name) for name in names}\n"
    "wrong = {name: spec and spec.origin for name, spec in specs.items()\n"
    "         if spec is None or pathlib.Path(spec.origin).resolve() != root / name / '__init__.py'}\n"
    "sys.exit(f'packages not imported from {root}: {wrong}' if wrong else 0)\n"
)


def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            from ruamel.yaml import YAML
        except ImportError:
            raise SystemExit("envs/setup.py needs PyYAML or ruamel.yaml; run it with conda's base Python") from None
        return YAML(typ="safe", pure=True).load(text)
    return yaml.safe_load(text)


def installs_repository(spec: dict) -> bool:
    """True when the repository is pip-installed into the environment, False when it is
    only started from the repository root (``repository: from-root`` in envs/roles.yaml)."""
    mode = spec.get("repository", "installed")
    if mode not in {"installed", "from-root"}:
        raise SystemExit(f"repository must be installed or from-root, not {mode!r}")
    return mode == "installed"


def fill(value, lock: dict):
    """Expand "{lock[name][key]}" references from third_party/lock.yaml."""
    if isinstance(value, str):
        return value.format(lock=lock)
    if isinstance(value, list):
        return [fill(item, lock) for item in value]
    return value


def select_units(config: dict, roles: list[str], families: list[str]) -> list[str]:
    """Units for the given roles and families, once each, in declaration order."""
    unknown = [role for role in roles if role not in config["roles"]]
    if unknown:
        raise SystemExit(f"Unknown role {unknown}; choose from {', '.join(config['roles'])}")
    unknown = [family for family in families if family not in config["families"]]
    if unknown:
        raise SystemExit(f"Unknown family {unknown}; choose from {', '.join(config['families'])}")
    selected = set()
    for role in roles:
        spec = config["roles"][role]
        selected.update(spec.get("units", []))
        for family in families:
            for part in spec.get("family", []):
                selected.update(config["families"][family][part])
    missing = selected - set(config["units"])
    if missing:
        raise SystemExit(f"envs/roles.yaml references undefined units {sorted(missing)}")
    return [name for name in config["units"] if name in selected]


def plan_steps(config: dict, lock: dict, units: list[str]) -> list[tuple[str, str, str, object]]:
    """(unit, environment, kind, value) for every step, with lock references filled."""
    steps = []
    for unit in units:
        spec = config["units"][unit]
        for step in spec["steps"]:
            (kind, value), = step.items()
            steps.append((unit, spec["environment"], kind, fill(value, lock)))
    return steps


class Installer:
    def __init__(self, config: dict, lock: dict, *, dry_run: bool):
        self.environments, self.lock, self.dry_run = config["environments"], lock, dry_run
        self.conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not self.conda and not dry_run:
            raise SystemExit("conda was not found; install Miniconda first")
        self.conda = self.conda or "conda"
        self.ready: set[str] = set()
        self.done: set[tuple[str, str]] = set()
        self.manual: list[str] = []

    # -- helpers ---------------------------------------------------------------------------
    def call(self, command: list[str], *, shown: str | None = None) -> None:
        print("  $ " + (shown or shlex.join(command)), flush=True)
        if not self.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    def inside(self, environment: str, *command: str) -> list[str]:
        return [self.conda, "run", "--no-capture-output", "-n", environment, *command]

    def existing_environments(self) -> set[str]:
        listing = json.loads(subprocess.run([self.conda, "env", "list", "--json"], check=True,
                                            capture_output=True, text=True).stdout)
        return {Path(prefix).name for prefix in listing["envs"]}

    def checkout(self, name: str) -> Path:
        return THIRD_PARTY / self.lock[name]["checkout"]

    # -- steps -----------------------------------------------------------------------------
    def environment(self, name: str) -> None:
        if name in self.ready:
            return
        self.ready.add(name)
        python = self.environments[name]["python"]
        if self.dry_run:
            print(f"  conda environment {name} (python {python}), created if missing")
        elif name not in self.existing_environments():
            self.call([self.conda, "create", "-y", "-n", name, f"python={python}", "pip"])

    def clone(self, name: str) -> None:
        pin, target = self.lock[name], self.checkout(name)
        relative = target.relative_to(ROOT)
        if self.dry_run:
            print(f"  clone {pin['repository']} -> {relative} @ {pin['commit'][:12]} (an existing checkout is verified)")
            return
        if not target.exists():
            self.call(["git", "clone", pin["repository"], str(target)])
            self.call(["git", "-C", str(target), "checkout", "--detach", pin["commit"]])
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        if head != pin["commit"]:
            raise SystemExit(f"{relative} is at {head}, the lock pins {pin['commit']}; "
                             "check out the pinned commit yourself (local work is never reset)")

    def patch(self, name: str) -> None:
        pin, target = self.lock[name]["patch"], self.checkout(name)
        patch = THIRD_PARTY / pin["path"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != pin["sha256"]:
            raise SystemExit(f"{patch.relative_to(ROOT)} differs from its sha256 in third_party/lock.yaml")
        if self.dry_run:
            print(f"  apply {patch.relative_to(ROOT)} to {target.relative_to(ROOT)} (skipped if already applied)")
            return
        applied = subprocess.run(["git", "-C", str(target), "apply", "--reverse", "--check", str(patch)],
                                 capture_output=True).returncode == 0
        if not applied:
            self.call(["git", "-C", str(target), "apply", str(patch)])

    def run_step(self, environment: str, kind: str, value) -> None:
        if (kind, json.dumps([environment, value])) in self.done:
            return
        self.done.add((kind, json.dumps([environment, value])))
        if kind in {"note", "manual"}:
            print(f"  {kind}: {value}")
            if kind == "manual":
                self.manual.append(value)
            return
        if kind == "clone":
            return self.clone(value)
        if kind == "patch":
            return self.patch(value)
        self.environment(environment)
        if kind == "pip":
            self.call(self.inside(environment, "python", "-m", "pip", "install", *value))
        elif kind == "run":
            self.call(self.inside(environment, "python", *value))
        elif kind == "conda":
            self.call([self.conda, "install", "-y", "-n", environment, *value])
        elif kind == "npm":
            self.call(self.inside(environment, "npm", "--prefix", "web", *value))
        elif kind == "hf":
            pin = self.lock[value]
            self.call(self.inside(environment, "hf", "download", pin["weights"], "--revision", pin["weights_revision"]))
        elif kind == "verify" and value == "packages":
            installed = installs_repository(self.environments[environment])
            where = "this checkout" if installed else "the repository root (not installed)"
            self.call(self.inside(environment, "python", "-I" if installed else "-E", "-c", VERIFY_PACKAGES, str(ROOT)),
                      shown=f"check in {environment}: {', '.join(PACKAGES)} import from {where}")
        else:
            raise SystemExit(f"Unknown step {kind}: {value}")


def summary(config: dict, lock: dict, units: list[str], steps) -> list[str]:
    checkouts = list(dict.fromkeys(lock[value]["checkout"] for _, _, kind, value in steps if kind == "clone"))
    return [
        f"environments: {', '.join(dict.fromkeys(env for _, env, _, _ in steps)) or '-'}",
        f"third_party:  {', '.join(checkouts) or '-'}",
        f"weights:      {', '.join(unit for unit in units if unit.endswith('-weights')) or '-'}",
        f"web build:    {'yes' if 'web' in units else 'no'}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", action="append", default=[], help="Device role; repeat to combine")
    parser.add_argument("--family", action="append", default=[],
                        help=f"Policy family for family-dependent roles (default {DEFAULT_FAMILY}); repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without installing anything")
    parser.add_argument("--list", action="store_true", help="List roles and exit")
    args = parser.parse_args(argv)
    config, lock = load_yaml(ROLES), load_yaml(LOCK)
    if args.list or not args.role:
        for name, spec in config["roles"].items():
            print(f"{name:18} {spec['summary']}")
        print(f"families: {', '.join(config['families'])}")
        return 0 if args.list else 2
    uses_family = any(config["roles"].get(role, {}).get("family") for role in args.role)
    families = args.family or ([DEFAULT_FAMILY] if uses_family else [])
    units = select_units(config, args.role, families)
    steps = plan_steps(config, lock, units)
    print(f"roles: {', '.join(args.role)}" + (f" · families: {', '.join(families)}" if uses_family else ""))
    for line in summary(config, lock, units, steps):
        print("  " + line)
    installer = Installer(config, lock, dry_run=args.dry_run)
    current = None
    for unit, environment, kind, value in steps:
        if unit != current:
            current = unit
            print(f"[{unit}] ({environment})")
        installer.run_step(environment, kind, value)
    if installer.manual:
        print("Manual steps left:" + "".join(f"\n  - {item}" for item in installer.manual))
    print("Dry run only; nothing was installed." if args.dry_run else "Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
