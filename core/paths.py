"""Repository locations shared by every package.

Code resolves workspace paths from here instead of counting ``__file__`` parents,
so moving a module never changes where configs, upstream checkouts or outputs live.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
THIRD_PARTY = ROOT / "third_party"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
WEB = ROOT / "web"


def module_command(module: str, *args: object, python: str = "python") -> list[str]:
    """argv for a repository command module; run it with ``cwd=ROOT``."""
    return [python, "-m", module, *(str(arg) for arg in args)]
