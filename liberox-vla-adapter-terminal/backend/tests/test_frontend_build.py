from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.app.core import frontend


def make_frontend(root: Path, *, with_dependencies: bool = True) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.tsx").write_text("export const version = 1;\n", encoding="utf-8")
    (root / "package.json").write_text('{"scripts":{"build":"vite build"}}\n', encoding="utf-8")
    if with_dependencies:
        (root / "node_modules").mkdir()
    return root


def test_stale_frontend_is_built_once_and_rebuilt_after_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = make_frontend(tmp_path / "frontend")
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(frontend.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        (cwd / "dist").mkdir(exist_ok=True)
        (cwd / "dist" / "index.html").write_text("<html></html>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    assert frontend.ensure_frontend_build(root, run_command=fake_run) is True
    assert calls == [(["/usr/bin/npm", "run", "build"], root.resolve(), True)]
    assert frontend.frontend_build_is_current(root)

    assert frontend.ensure_frontend_build(root, run_command=fake_run) is False
    assert len(calls) == 1

    (root / "src" / "main.tsx").write_text("export const version = 2;\n", encoding="utf-8")
    assert frontend.ensure_frontend_build(root, run_command=fake_run) is True
    assert len(calls) == 2


def test_stale_frontend_without_dependencies_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = make_frontend(tmp_path / "frontend", with_dependencies=False)
    monkeypatch.setattr(frontend.shutil, "which", lambda command: f"/usr/bin/{command}")

    with pytest.raises(frontend.FrontendBuildError, match="npm ci"):
        frontend.ensure_frontend_build(root)

