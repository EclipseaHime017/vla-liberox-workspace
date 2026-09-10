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
  │    ├─ Manual controller     SpaceMouse → OSC_POSE recorder; FACTR → joint-only worker
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
- Manual input has separate connection/calibration and armed-session state. A session fixes its controller choice; probe/calibration cannot drive the simulator. SpaceMouse keeps its existing HID path. FACTR reads complete joint packets through its official runtime, with a sequence and host-monotonic successful-packet receipt time; the 20 Hz worker never waits for the next device packet. Its latency display is sample age, not a firmware timestamp or physical end-to-end latency.
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

## FACTR boundaries: shared official runtime

GUI `FactrControllerService` and CLI `test_factr.py` both use
`devices/factr_client.py`. The shared IPC client handles process lifetime,
snapshots and gripper latching; the GUI service only manages controller/session
state. SpaceMouse retains its existing service and installed PySpaceMouse package.

`factr_official_runtime.py` owns one serial loop in an isolated system-Python
ROS/Pinocchio process. It calls the pinned official calibration, gravity, measured
velocity, friction and limit-barrier methods. Normal compensation delegates the
null-space term unchanged; preparation substitutes a nonblocking position PD term.
`factr.py` contains shared configuration/snapshots and read-only startup preflight,
not another GUI sampling thread. `factr_runtime_config.py` resolves runtime paths.
`factr_calibration.py` persists/checks profiles; it does not implement joint-offset
selection or duplicate encoder decoding.

One calibration captures the whole-arm reference and released trigger zero,
using official 0.8 rad trigger travel. The host profile binds offsets, identity
and mapping configuration. Only explicit GUI ON / CLI g enables motor support.
Disarming simulation does not disable support: normal completion, countdown and
rewind keep it active. Simulation errors and backend shutdown close the official
client before postprocessing or waiting for the simulation thread. Manual OFF is
available while armed. Errors invalidate calibration and never automatically rearm.
Shutdown failure is exposed as unknown output state. Browser disconnect alone is
not application shutdown.

A Unix socketpair carries commands, heartbeat and snapshots. The child handles
bus faults and parent loss. Port exclusivity, hardware current limits, 100 ms
motor watchdog and verified cleanup remain; no custom speed/time limits.
Trigger torque stays OFF. No automatic USB, EEPROM, mode or permission changes.
Official USB latency 1 ms / 4 Mbps requirements apply.

`simulation_worker.py` owns the shared recording/publishing lifecycle for both
controllers. `factr_control_worker.py` provides only leader alignment and the
joint-following loop; `factr_joint_control.py` installs absolute Panda joint
control without state teleport, plus inverse OSC scaling for measured FK pose
increments. After preview/alignment/countdown it follows joint targets at 20 Hz.
Proprio is refreshed after each integration to match the saved state. Seven-D
labels use world translation and `Log(R_next R_current^T)` rotation; they are
achieved-motion relabels, not executed joint targets or guaranteed inverse
dynamics. Both controllers record `human` transitions, preserve parent prefixes,
reconstruct dual cameras from states, and enter the same dataset/training path.
No new joint sample CSV is written; raw_action retains unclipped labels and
controller diagnostics report clipping. Normal completion retains physical
compensation; faults disable it before publishing partial recordings.
The physical leader uses official small-arm dynamics, not Panda inertias.

`factr_alignment.py` manages preparation reference ramp/settling. The official
runtime applies official joint-position PD gains nonblockingly in place of its
rest-pose term during preparation, retaining gravity/friction/barriers. This
adapter is distinct from the upstream blocking alignment method. Alignment's
>=200 Hz precondition (rolling 50-cycle mean) and bounded ramp apply only to
active preparation. An isolated >5 ms interval suspends that tick's alignment
PD/reference advance without exiting; sustained low rate remains an error.
The IPC shutdown message reports verified OFF separately from process exit.
A nonzero worker exit does not invalidate a successfully read-back OFF state;
missing confirmation or failed read-back remains an unknown/failed shutdown.

The immutable checkout/runtime under third_party are ignored by Git and verified
with configs/factr_official.lock.json. setup_factr.py --check never opens hardware.
No duplicate gravity model or URDF remains. CLI tests save only calibration;
physical direction, compensation and device latency still require hardware
acceptance. Training algorithms, UI training and server branch are unchanged.

Stage-based reward work is currently [research only](STAGE_REWARD_RESEARCH.md).
It does not add a stage model to this dependency graph or alter reward
materialization, evaluation sidecars, IQL, or the `server` branch.

The native host launch is intentional: the current GPU, EGL/MuJoCo, and HID device stack is hardware-coupled. A Docker Compose file is omitted until NVIDIA, EGL, and `/dev/hidraw` passthrough can be supported without weakening device permissions.
