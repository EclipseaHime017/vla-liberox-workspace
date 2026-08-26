# Architecture

The studio uses a one-way dependency flow:

```text
React client
  ↓ HTTP / WebSocket
FastAPI routers (backend/app/api)
  ├─ RunService → SimulationManager / worker
  │    ├─ LiberoXSimulator       simulator lifecycle; never persists data
  │    ├─ VLAAdapterPolicyProvider  model load/predict/unload
  │    ├─ EpisodeRecorderFactory trajectory/media publishing; never calls the UI
  │    ├─ LiberoEvaluator        LIBERO `done` success semantics
  │    ├─ ConfiguredTaskCatalog  BDDL/init-state discovery
  │    └─ RunRepository          SQLite index; run files remain authoritative
  ├─ TrainingDatasetService → immutable manifests + hash verification
  ├─ OfflineJobService / evaluation.batch → frozen schedule + evaluation index
  └─ OfflineJobService → detached runner
       ├─ vla-liberox: prepare / Pixel-IQL
       ├─ rynnvalue-reward: frozen reward annotation
       ├─ vla-liberox: lightweight batch policy evaluation
       └─ cross-process GPU file lock
```

## State ownership

- `SimulationSession` and `SimulationDraft` are domain state objects and import no FastAPI, MuJoCo, or SQLite code.
- One worker owns the active control environment. Its state machine remains `IDLE → LOADING → READY → RUNNING → STOPPING → POSTPROCESSING → COMPLETED/ERROR`.
- The shared preview service owns its own read-only MuJoCo environment and consumes only the latest submitted state.
- `RunService` is the only interface used by HTTP and WebSocket routers.
- React never imports simulator concepts or touches files; it consumes documented API resources.
- Offline jobs own their Conda child process groups and durable logs. FastAPI may restart without terminating them, then reconciles PID/heartbeat state from `job.json`.
- A batch evaluation freezes its complete `init_state_index`/environment-seed schedule before the detached worker starts. The worker owns the policy and simulator lifecycle; its only result artifact is the atomically updated `evaluation.json`.
- Simulation, annotation, training, and batch evaluation share one GPU task guard; TensorBoard is read-only and outside that lock.

## Extension points

- Add a policy by implementing `policies/base.py` and registering it in the composition root.
- The independent `vla-adapter-rynn-iql/` trainer publishes immutable component overlays to `policy-registry/`. The UI validates manifests and hashes but never imports training or RynnValue code; `VLAAdapterPolicyProvider` alone applies a selected overlay at the model boundary.
- Add a simulator through `simulators/base.py`; simulator adapters must not write the catalog.
- Add tasks through `configs/ui_config.yaml`; task identity is frozen into every run.
- Batch evaluation selects exactly one registered task and one policy snapshot. It may vary only that task's finite init states and environment seeds; adding a new randomization axis belongs in `EvaluationService`, not in React or the simulator adapter.
- Add exporters under `recording/` without introducing UI dependencies.

The native host launch is intentional: the current GPU, EGL/MuJoCo, and HID device stack is hardware-coupled. A Docker Compose file is omitted until NVIDIA, EGL, and `/dev/hidraw` passthrough can be supported without weakening device permissions.
