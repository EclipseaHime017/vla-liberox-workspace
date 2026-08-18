"""Keep the ignored Vite build synchronized with the checked-out sources."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


LOGGER = logging.getLogger(__name__)
BUILD_STAMP_NAME = ".source-fingerprint"
_ROOT_SOURCES = (
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "vite.config.ts",
)


class FrontendBuildError(RuntimeError):
    """Raised when a current frontend build cannot be prepared safely."""


def frontend_source_fingerprint(frontend_root: Path) -> str:
    """Hash every input that can change the production frontend bundle."""
    root = frontend_root.resolve()
    sources = [root / name for name in _ROOT_SOURCES]
    source_root = root / "src"
    if source_root.is_dir():
        sources.extend(path for path in source_root.rglob("*") if path.is_file())

    digest = hashlib.sha256()
    existing = sorted((path for path in sources if path.is_file()), key=lambda path: path.as_posix())
    if not existing:
        raise FrontendBuildError(f"没有找到前端源码：{root}")
    for path in existing:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def frontend_build_is_current(frontend_root: Path, fingerprint: str | None = None) -> bool:
    """Return whether dist was built from the current source fingerprint."""
    root = frontend_root.resolve()
    index = root / "dist" / "index.html"
    stamp = root / "dist" / BUILD_STAMP_NAME
    if not index.is_file() or not stamp.is_file():
        return False
    expected = fingerprint or frontend_source_fingerprint(root)
    return stamp.read_text(encoding="utf-8").strip() == expected


def ensure_frontend_build(
    frontend_root: Path,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Build stale Vite assets and return True only when a rebuild occurred."""
    root = frontend_root.resolve()
    fingerprint = frontend_source_fingerprint(root)
    if frontend_build_is_current(root, fingerprint):
        LOGGER.info("前端静态资源已是最新版本（源码指纹 %s）。", fingerprint[:12])
        return False

    npm = shutil.which("npm")
    if npm is None:
        raise FrontendBuildError(
            "前端源码已更新，但系统找不到 npm。请安装 Node.js/npm 后重新启动 UI。"
        )
    if not (root / "node_modules").is_dir():
        raise FrontendBuildError(
            "前端源码已更新，但依赖尚未安装。请先执行：\n"
            f"  cd {root}\n"
            "  npm ci"
        )

    LOGGER.info("检测到前端源码变化（源码指纹 %s）。", fingerprint[:12])
    LOGGER.info("即将执行前端构建：cd %s && npm run build", root)
    try:
        run_command([npm, "run", "build"], cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrontendBuildError(
            "前端自动构建失败。请在 frontend 目录运行 `npm ci && npm run build` 后重试。"
        ) from exc

    index = root / "dist" / "index.html"
    if not index.is_file():
        raise FrontendBuildError("npm 构建结束后仍未生成 frontend/dist/index.html。")
    stamp = root / "dist" / BUILD_STAMP_NAME
    temporary = stamp.with_suffix(".tmp")
    temporary.write_text(fingerprint + "\n", encoding="utf-8")
    temporary.replace(stamp)
    LOGGER.info("前端静态资源已更新。")
    return True


__all__ = [
    "BUILD_STAMP_NAME",
    "FrontendBuildError",
    "ensure_frontend_build",
    "frontend_build_is_current",
    "frontend_source_fingerprint",
]
