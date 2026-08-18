# LIBERO-X Local Data Studio

Local-first simulation, VLA evaluation, trajectory rewind, and SpaceMouse takeover for the three validated Franka/LIBERO-X tasks.

- Backend: FastAPI application service with a background simulation worker.
- Frontend: React + TypeScript, served by FastAPI after a Vite production build.
- Storage: portable run directories plus a rebuildable SQLite catalog.
- Compatibility: the existing evaluation, intervention, and SpaceMouse CLI scripts remain available.
- Configuration: fixed runtime settings live in [`configs/`](configs/); application code lives in [`liberox-vla-adapter-terminal/`](liberox-vla-adapter-terminal/).
- Operator preview: a transient 2x2 stream shows agent, wrist, −45°, and +45° cameras; VLA input and recorded artifacts remain the original two cameras.

Start from `vla-liberox-workspace/` after activating `vla-liberox`:

```bash
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

On a new checkout, run `npm ci` once in `liberox-vla-adapter-terminal/frontend/`. The launcher fingerprints the frontend sources, prints the exact build command before running it, and automatically rebuilds the Git-ignored `frontend/dist` after later pulls. Run `npm run build` there for a manual source-only rebuild, or `npm ci && npm run build` after `package-lock.json` changes. Neither `npm run build` nor `npm test` installs or upgrades dependencies.

Open <http://127.0.0.1:8000>. See [README_CN.md](README_CN.md) for setup and operation, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries, and [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for persistence rules.
