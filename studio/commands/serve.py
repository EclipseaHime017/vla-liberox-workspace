#!/usr/bin/env python3
"""Start the local LIBERO-X simulation and intervention Web UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import uvicorn

from core.paths import ROOT, WEB

from studio.frontend import FrontendBuildError, ensure_frontend_build, frontend_build_info


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("项目目录：%s", ROOT)
    try:
        ensure_frontend_build(WEB)
    except FrontendBuildError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    build = frontend_build_info(WEB)
    logger.info(
        "UI 构建：%s（current=%s, assets=%s）",
        str(build["dist_fingerprint"])[:12],
        build["current"],
        ", ".join(str(value) for value in build["assets"]),
    )

    import sim.env as direct
    from studio.main import create_app
    from studio.config import DEFAULT_UI_CONFIG, load_ui_config

    ui_config = load_ui_config(DEFAULT_UI_CONFIG)
    eval_config = direct.load_config(direct.DEFAULT_CONFIG_PATH.resolve())
    app = create_app(ui_config=ui_config, eval_config=eval_config)
    uvicorn.run(
        app,
        host=ui_config.host,
        port=ui_config.port,
        log_level="info",
        # The UI intentionally polls lightweight status endpoints. Keep
        # warnings, errors, and application milestones without printing one
        # access line per poll in the terminal.
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
