# Architecture

The studio uses a one-way dependency flow:

```text
React client
  ↓ HTTP / WebSocket
FastAPI routers (backend/app/api)
  ↓
RunService (backend/app/services/run_service.py)
  ↓
SimulationManager / worker (backend/app/workers/simulation_worker.py)
  ├─ LiberoXSimulator       simulator lifecycle; never persists data
  ├─ VLAAdapterPolicyProvider  model load/predict/unload
  ├─ EpisodeRecorderFactory trajectory/media publishing; never calls the UI
  ├─ LiberoEvaluator        LIBERO `done` success semantics
  ├─ ConfiguredTaskCatalog  BDDL/init-state discovery
  └─ RunRepository          SQLite index; run files remain authoritative
```

## State ownership

- `SimulationSession` and `SimulationDraft` are domain state objects and import no FastAPI, MuJoCo, or SQLite code.
- One worker owns the active control environment. Its state machine remains `IDLE → LOADING → READY → RUNNING → STOPPING → POSTPROCESSING → COMPLETED/ERROR`.
- The shared preview service owns its own read-only MuJoCo environment and consumes only the latest submitted state.
- `RunService` is the only interface used by HTTP and WebSocket routers.
- React never imports simulator concepts or touches files; it consumes documented API resources.

## Extension points

- Add a policy by implementing `policies/base.py` and registering it in the composition root.
- Add a simulator through `simulators/base.py`; simulator adapters must not write the catalog.
- Add tasks through `ui_config.yaml`; task identity is frozen into every run.
- Add exporters under `recording/` without introducing UI dependencies.

The native host launch is intentional: the current GPU, EGL/MuJoCo, and HID device stack is hardware-coupled. A Docker Compose file is omitted until NVIDIA, EGL, and `/dev/hidraw` passthrough can be supported without weakening device permissions.
