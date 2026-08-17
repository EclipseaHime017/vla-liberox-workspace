#!/usr/bin/env python3
"""Start the local LIBERO-X simulation and intervention Web UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import uvicorn

import eval_pickplace_direct as direct
from backend.app.main import create_app
from backend.app.core.config import DEFAULT_UI_CONFIG, load_ui_config


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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
