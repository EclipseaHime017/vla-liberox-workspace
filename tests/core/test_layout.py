"""Repository layout rules: package layering, no stale names, no shadowed packages."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from core.paths import ROOT

# A package may import itself and strictly lower layers only; peers stay independent.
RANK = {"core": 0, "sim": 1, "rewards": 1, "train": 2, "cluster": 3, "studio": 3}
STALE = ("vla_rynn_iql", "vla-adapter-rynn-iql", "vla_adapter_robometer", "vla-adapter-robometer",
         "liberox-vla-adapter-terminal", "backend.app")


def imported_packages(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


@pytest.mark.parametrize("package", sorted(RANK))
def test_packages_only_depend_on_lower_layers(package):
    violations = [
        f"{path.relative_to(ROOT)}:{line} imports {target}"
        for path in sorted((ROOT / package).rglob("*.py"))
        for line, target in imported_packages(path)
        if target in RANK and target != package and RANK[target] >= RANK[package]
    ]
    assert violations == []


def test_product_code_and_configs_use_current_names():
    roots = [*(ROOT / name for name in RANK), ROOT / "configs", ROOT / "envs", ROOT / "web" / "src"]
    files = [path for root in roots for path in root.rglob("*")
             if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".txt", ".ts", ".tsx", ".json"}
             and "local" not in path.relative_to(ROOT).parts]
    stale = [f"{path.relative_to(ROOT)}: {name}" for path in files
             for name in STALE if name in path.read_text(encoding="utf-8", errors="ignore")]
    assert stale == []


@pytest.mark.parametrize("package", sorted(RANK))
def test_packages_resolve_to_this_repository(package):
    spec = importlib.util.find_spec(package)
    assert spec is not None and Path(spec.origin).resolve() == ROOT / package / "__init__.py"


def test_no_script_shadows_a_package():
    names = set(RANK)
    shadows = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py")
               if path.stem in names and not {"node_modules", "third_party", "logs"} & set(path.parts)]
    assert shadows == []
