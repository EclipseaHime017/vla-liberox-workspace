#!/usr/bin/env python3
"""Read-only deployment check of this checkout: repository, conda environments, upstreams.

    python skills/deploy/scripts/preflight.py                         # whatever is present here
    python skills/deploy/scripts/preflight.py --role studio --role teleop
    python skills/deploy/scripts/preflight.py --role train --family pi05 --json

Never installs, fetches, pulls or modifies anything. Needs only the standard library and
the YAML loader of envs/setup.py (PyYAML or ruamel.yaml), so it runs before any project
environment exists. Exit code 0: no failures (warnings allowed); 1: at least one failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ("core", "sim", "rewards", "train", "cluster", "studio")
LEGACY_DIRECTORIES = ("liberox-vla-adapter-terminal", "vla-adapter-rynn-iql", "vla-adapter-robometer")
STALE_DISTRIBUTIONS = ("vla-rynn-iql", "vla-adapter-robometer")
# Runs inside each environment with `python -I`: the working directory cannot mask the install.
PROBE = (
    "import importlib.metadata, importlib.util, json, sys\n"
    f"packages = {{name: getattr(importlib.util.find_spec(name), 'origin', None) for name in {PACKAGES!r}}}\n"
    "stale = {}\n"
    f"for name in {STALE_DISTRIBUTIONS!r}:\n"
    "    try:\n"
    "        stale[name] = importlib.metadata.version(name)\n"
    "    except importlib.metadata.PackageNotFoundError:\n"
    "        pass\n"
    "print(json.dumps({'python': sys.version.split()[0], 'packages': packages, 'stale': stale}))\n"
)


def load_setup():
    spec = importlib.util.spec_from_file_location("envs_setup", ROOT / "envs" / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(area: str, item: str, status: str, detail: str, hint: str = "") -> dict:
    return {"area": area, "item": item, "status": status, "detail": detail, "hint": hint}


def git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=60)


def required(config: dict, setup, roles: list[str], families: list[str]) -> tuple[list[str], list[str], bool]:
    """Environments, upstreams and whether the web build belongs to the requested roles."""
    if not roles:
        return [], [], False
    units = setup.select_units(config, roles, families)
    lock = setup.load_yaml(setup.LOCK)
    steps = setup.plan_steps(config, lock, units)
    environments = list(dict.fromkeys(environment for _, environment, _, _ in steps))
    upstreams = list(dict.fromkeys(value for _, _, kind, value in steps if kind == "clone"))
    return environments, upstreams, "web" in units


def check_repository(root: Path) -> list[dict]:
    area = "repository"
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    head = git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    results = [result(area, "version", "ok", f"{version} · {branch} @ {head}")]
    changes = [line for line in git(root, "status", "--porcelain").stdout.splitlines() if line]
    if changes:
        results.append(result(area, "working tree", "warn", f"{len(changes)} uncommitted changes",
                              "commit or stash them with the user's consent before pulling; never discard them"))
    counts = git(root, "rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    if counts.returncode == 0:
        behind, ahead = (int(value) for value in counts.stdout.split())
        if behind and ahead:
            results.append(result(area, "upstream", "warn", f"diverged: {behind} behind, {ahead} ahead (as of the last fetch)",
                                  "do not pull or force-push; ask the user how to reconcile"))
        elif behind:
            results.append(result(area, "upstream", "warn", f"{behind} commits behind (as of the last fetch)",
                                  "git pull --ff-only after the user confirms"))
        else:
            results.append(result(area, "upstream", "ok", f"up to date with the last fetch ({ahead} ahead)"))
    leftovers = [name for name in LEGACY_DIRECTORIES if (root / name).exists()]
    if leftovers:
        results.append(result(area, "0.8.0 leftovers", "warn", ", ".join(leftovers),
                              "untracked files left by the 0.8.0 layout (node_modules, outputs ...); review with the user before deleting"))
    return results


def conda_environments() -> tuple[str | None, set[str]]:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        return None, set()
    listing = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=120)
    if listing.returncode:
        return conda, set()
    return conda, {Path(prefix).name for prefix in json.loads(listing.stdout)["envs"]}


def check_environment(conda: str, name: str, root: Path) -> dict:
    area = "environments"
    probe = subprocess.run([conda, "run", "-n", name, "python", "-I", "-c", PROBE],
                           capture_output=True, text=True, timeout=300)
    lines = probe.stdout.strip().splitlines()
    if probe.returncode or not lines:
        return result(area, name, "fail", f"python does not start: {probe.stderr.strip()[-300:]}",
                      f"rebuild with python envs/setup.py --role <role> (dry-run first)")
    report = json.loads(lines[-1])
    origins = {package: report["packages"][package] for package in PACKAGES}
    foreign = {package: origin for package, origin in origins.items()
               if origin is not None and Path(origin).resolve() != root / package / "__init__.py"}
    reinstall = f"conda run -n {name} python -m pip install --no-deps -e ."
    if foreign:
        return result(area, name, "fail", f"packages imported from elsewhere: {foreign}",
                      f"uninstall the shadowing distribution, then {reinstall}")
    if report["stale"]:
        stale = " ".join(report["stale"])
        return result(area, name, "warn", f"python {report['python']}; 0.8.0 editable installs remain: {report['stale']}",
                      f"conda run -n {name} python -m pip uninstall -y {stale} && {reinstall}")
    if any(origin is None for origin in origins.values()):
        return result(area, name, "warn", f"python {report['python']}; repository not installed (commands work only from the repository root)",
                      reinstall)
    return result(area, name, "ok", f"python {report['python']}; packages import from this checkout")


def generated(path: str) -> bool:
    """Bytecode and packaging metadata that installs and imports create inside a checkout."""
    return "__pycache__/" in path or path.endswith((".pyc", ".pyo")) or ".egg-info" in path


def check_upstream(name: str, pin: dict, third_party: Path) -> dict:
    area = "third_party"
    path = third_party / pin["checkout"]
    if not path.exists():
        return result(area, name, "fail", f"third_party/{pin['checkout']} is missing",
                      "python envs/setup.py --role <role> clones it")
    head = git(path, "rev-parse", "HEAD")
    if head.returncode:
        return result(area, name, "fail", f"third_party/{pin['checkout']} is not a git checkout")
    head = head.stdout.strip()
    if head != pin["commit"]:
        return result(area, name, "fail", f"at {head[:12]}, the lock pins {pin['commit'][:12]}",
                      "check out the pinned commit with the user; never reset local work")
    if "patch" in pin:
        patch = third_party / pin["patch"]["path"]
        if git(path, "apply", "--reverse", "--check", str(patch)).returncode:
            return result(area, name, "fail", "the locked patch is not applied",
                          "python envs/setup.py applies it (patch step)")
    elif "files" in pin:
        changed = [file for file, digest in pin["files"].items()
                   if not (path / file).is_file() or hashlib.sha256((path / file).read_bytes()).hexdigest() != digest]
        if changed:
            return result(area, name, "fail", f"official sources differ from the lock: {changed[:5]}")
    elif changes := [line[3:] for line in git(path, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
                     if line and not generated(line[3:])]:
        return result(area, name, "warn", f"checkout has local modifications: {changes[:5]}",
                      "the pinned upstream must stay clean; ask the user before touching it")
    return result(area, name, "ok", f"third_party/{pin['checkout']} @ {head[:12]}")


def run(roles: list[str], families: list[str], *, root: Path = ROOT) -> dict:
    setup = load_setup()
    config, lock = setup.load_yaml(setup.ROLES), setup.load_yaml(setup.LOCK)
    uses_family = any(config["roles"].get(role, {}).get("family") for role in roles)
    families = families or ([setup.DEFAULT_FAMILY] if uses_family else [])
    needed_environments, needed_upstreams, needs_web = required(config, setup, roles, families)
    results = check_repository(root)

    conda, existing = conda_environments()
    if conda is None:
        results.append(result("environments", "conda", "fail", "conda not found", "install Miniconda first"))
    for name in needed_environments or sorted(set(config["environments"]) & existing):
        if conda is None:
            break
        if name in existing:
            results.append(check_environment(conda, name, root))
        else:
            results.append(result("environments", name, "fail", "missing",
                                  f"python envs/setup.py {' '.join(f'--role {r}' for r in roles)} --dry-run"))

    upstreams = [name for name, pin in lock.items() if isinstance(pin, dict)]
    third_party = root / "third_party"
    for name in needed_upstreams or [n for n in upstreams if (third_party / lock[n]["checkout"]).exists()]:
        results.append(check_upstream(name, lock[name], third_party))

    if needs_web and not (root / "web" / "node_modules").is_dir():
        results.append(result("web", "node_modules", "fail", "frontend dependencies missing",
                              "the web unit of envs/setup.py runs npm ci and the build"))
    return {"roles": roles, "families": families, "results": results,
            "failures": sum(item["status"] == "fail" for item in results),
            "warnings": sum(item["status"] == "warn" for item in results)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", action="append", default=[], help="Device role from envs/roles.yaml; repeatable")
    parser.add_argument("--family", action="append", default=[], help="Policy family for family-dependent roles")
    parser.add_argument("--json", action="store_true", help="Print one JSON object instead of text")
    args = parser.parse_args(argv)
    report = run(args.role, args.family)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        scope = ", ".join(report["roles"]) or "everything present"
        print(f"preflight · {scope}" + (f" · families: {', '.join(report['families'])}" if report["families"] else ""))
        area = None
        for item in report["results"]:
            if item["area"] != area:
                area = item["area"]
                print(f"[{area}]")
            print(f"  {item['status'].upper():5} {item['item']:18} {item['detail']}")
            if item["hint"]:
                print(f"        -> {item['hint']}")
        print(f"{report['failures']} failures, {report['warnings']} warnings")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
