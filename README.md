# LIBERO-X Local Data Studio

Current release: **v0.5.0**

Release highlights: FACTR teleoperation alongside SpaceMouse, and independent
human Stage-based rewards with keyframe annotation and training source selection.

Local-first simulation, VLA evaluation, trajectory rewind, SpaceMouse / FACTR takeover,
offline post-training, and reproducible batch policy testing for the three
validated Franka/LIBERO-X tasks.

- Backend: FastAPI application service with a background simulation worker.
- Frontend: React + TypeScript, served by FastAPI after a Vite production build.
- Storage: portable run directories plus a rebuildable SQLite catalog.
- Compatibility: the existing evaluation, intervention, and SpaceMouse CLI scripts remain available.
- Configuration: fixed runtime settings live in [`configs/`](configs/); application code lives in [`liberox-vla-adapter-terminal/`](liberox-vla-adapter-terminal/).
- Operator preview: a transient 2x2 stream shows agent, wrist, −45°, and +45° cameras; VLA input and recorded artifacts remain the original two cameras.
- FACTR Franka: GUI and CLI share official calibration and gravity compensation, with one reference capture and explicit ON/OFF. Joint following includes slow leader alignment; GUI records measured end-effector action labels, trajectories and dual-camera video through the standard manual-data pipeline. Physical acceptance is still required.
- Run drafts can choose a reproducible random seed and ablate either VLA camera by replacing only that fixed model-input slot with a black frame; raw preview and recording data remain intact.
- Offline post-training: [`vla-adapter-rynn-iql/`](vla-adapter-rynn-iql/) imports the read-only dataset, annotates temporal value with pinned RynnValue, trains a PyTorch IQL overlay, and publishes only the action head and proprio projector to `policy-registry/`.
- Integrated workflow: the Dataset page evaluates RynnValue once per trajectory, preserves its complete output sidecar, paginates run previews, exposes video/action/EEF, absolute/relative remaining-time, observation-potential and entropy estimates, plus Shape/Final Reward details, and independently packages hash-verified training datasets. The Training page derives rewards from those cached model outputs using its selected `gamma`, shaping coefficient, and macro/primitive reduction, then launches resumable IQL jobs without rerunning RynnValue.
- Model registry: a dedicated sidebar page inspects base/overlay metadata and matching training history, and safely renames, copies, or removes local IQL overlays.
- Human stage rewards: mark positive/negative keyframes in trajectory details without cutting the recording. Training selects Sparse, RynnValue, or Stage-based rewards; Stage validates every member and freezes annotations per run. See [the implemented formulas and workflow](docs/STAGE_REWARD_RESEARCH.md#6-已实现人工关键帧直接奖励) and [Chinese usage §4.4.2](README_CN.md#442-annotate-与-reward-materialize-的边界).
- Batch testing: the Test page, immediately after Training in the sidebar, evaluates one task and one base/overlay policy over a frozen, deterministically balanced schedule of benchmark init states and environment seeds.

## Repository layout

```text
vla-liberox-workspace/
├── liberox-vla-adapter-terminal/  # simulation, intervention, Web UI and recording
├── vla-adapter-rynn-iql/          # standalone reward annotation and offline RL
├── configs/                       # terminal/UI runtime configuration
├── dataset-root/                  # recorded source data (Git-ignored)
├── policy-registry/               # immutable policy overlays (Git-ignored)
├── docs/                          # architecture and data-layout documentation
└── README_CN.md                   # complete Chinese setup and operating guide
```

The collection terminal and offline trainer are separate systems. The trainer
never rewrites recorded actions, states, observations, or media. UI-launched
RynnValue evaluation adds only hash-checked `rynnvalue_evaluation.json/npz`
sidecars beside an episode so later dataset packages can reuse the result. The
deployment integration boundary remains a hash-checked `policy.yaml` overlay
published to `policy-registry/`.

The Web UI orchestrates them without importing RynnValue into the simulation
process. Prepare/training/testing subprocesses use `vla-liberox`, annotation
uses `rynnvalue-reward`, and simulation/annotation/training/testing share a
persistent cross-process GPU lock. TensorBoard is read-only and remains outside
that lock. Dataset manifests reference and hash source artifacts instead of
copying trajectories or videos.

Start from `vla-liberox-workspace/` after activating `vla-liberox`:

```bash
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

On a new checkout, run `npm ci` once in `liberox-vla-adapter-terminal/frontend/`. The launcher fingerprints the frontend sources, prints the exact build command before running it, and automatically rebuilds the Git-ignored `frontend/dist` after later pulls. Run `npm run build` there for a manual source-only rebuild, or `npm ci && npm run build` after `package-lock.json` changes. Neither `npm run build` nor `npm test` installs or upgrades dependencies.

Open <http://127.0.0.1:8000>. See [README_CN.md](README_CN.md) for setup and operation, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries, and [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for persistence rules.

## FACTR calibration and manual gravity compensation

GUI and CLI share the pinned official FACTR classes, driver and control parameters.
The unmodified checkout/runtime live under `third_party/`; commit/file hashes are
verified against `configs/factr_official.lock.json`. No custom gravity model remains.

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/setup_factr.py
python liberox-vla-adapter-terminal/scripts/setup_factr.py --check
python liberox-vla-adapter-terminal/scripts/test_factr.py
```

Setup/check do not open hardware. System ROS 2/Pinocchio dependencies must be
installed first; SDK packages go into an isolated system-Python runtime, not the
VLA Python environment. Configure the explicit serial path in
[configs/factr_test_config.yaml](configs/factr_test_config.yaml).
The former `standalone` section is now `runtime`.

In GUI, select FACTR, support the complete arm in the official approximate resting
configuration and release the trigger, click Calibrate once, then Enable gravity
compensation. Calibration calls the official offset method and captures trigger
zero using the official 0.8 rad travel. No individual-motor or separate endpoint
calibration, and no arbitrary-pose physical-zero assumption.

Compensation is explicitly enabled, never enabled by calibration or takeover.
It persists through movement, normal simulation completion, countdown and rewind.
Manual OFF, simulation faults or backend shutdown disable output before expensive
postprocessing. Closing the browser alone does not stop the backend. Support the
arm before OFF; an unverified shutdown is shown as unknown, not falsely confirmed.

CLI: `c` calibrates, `s` tests, `g` enables without an ENABLE prompt,
`d` disables immediately, `i` shows status; `q`/Ctrl+C disable output then exit
the entire test program. No second confirmation is requested.
Normal test completion returns to the menu with support retained; Ctrl+C/faults
stop output immediately. Tests create no trajectories/videos/run directories;
only successful calibration persists.

Official gravity/friction/null-space/limit-barrier control uses a 500 Hz target
and gain 0.85. USB latency 1 ms, 4 Mbps and correct current-mode configuration
are required. No custom speed cutoff or support time limit; encoder integrity,
hardware current limits, watchdog and shutdown handling remain. Trigger torque
stays OFF. No automatic permissions, USB or EEPROM changes.

FACTR mirrors calibrated seven-joint targets through a Panda joint-position
controller (physics still runs; no qpos teleport). Before takeover, the simulation
stays fixed while the physical leader slowly aligns to it, holds through the
countdown, then follows 1:1. Keep the physical workspace clear. Alignment uses
the official PD gains in a nonblocking adapter with gravity support; it is not
the unmodified blocking upstream alignment routine. Its documented >=200 Hz
requirement is checked over a rolling 50-cycle window only for alignment, not
as a human motion-speed cutoff or an every-cycle 5 ms deadline. A single late
tick skips alignment PD/reference advance and retains normal compensation;
sustained low rate still stops the run. This is an application timing check,
not an assertion implemented by upstream FACTR. Hardware watchdogs remain.
FACTR and SpaceMouse use the same recorded manual-session and training workflow.
FACTR still follows joint targets; each 20 Hz step's measured end-effector world
translation and relative rotation are inverse-scaled into seven-dimensional OSC
labels, with the gripper command appended. These describe achieved motion, not
an exactly equivalent executed OSC command. No separate joint demonstration log
is written. Unclipped labels remain in raw_action; bounded labels in env_action,
with clipping statistics in controller diagnostics. Parent prefixes, state-based
dual-camera reconstruction and dataset export use the shared recording pipeline.
Normal completion retains compensation; errors disable it and save partial data.
SpaceMouse remains an installed `pyspacemouse` dependency, not a vendored checkout.
Do not run GUI, CLI or upstream demos against the same serial port concurrently.
See [the Chinese guide](README_CN.md#361-factr-franka-校准手动重力补偿与无-vla-测试)
and [official source](https://github.com/JasonJZLiu/FACTR_Teleop).

## Batch policy testing

The Test page runs a selected task and policy repeatedly without publishing
rollout media or trajectories. “Random environments” are the finite
`init_state_index` values belonging to that exact BDDL task, not other scenes;
environment seeds form the second randomization axis. The backend freezes a
reproducible, balanced schedule before launch, then runs every normal episode to
`max_steps`. A success is latched only after five consecutive `done=true`
control steps, while the remainder of the episode still executes.

The default mode is wall-clock-limited to 20 Hz; accelerated mode removes only
that wall-clock wait and keeps the MuJoCo control-frequency semantics unchanged.
Each test stores one lightweight `evaluation.json` under the project
`evaluations/` tree. Detached-job diagnostics (`job.json`, `job.log`, and the
validated effective YAML) remain in `jobs/`; no video, observation, action,
trajectory, image, or plot is written by batch testing.

## RynnValue + IQL offline post-training

The standalone pipeline keeps `VLA-Adapter/LIBERO-Object-Pro` as the deployed
Franka policy, freezes its vision/language backbone, and updates only the
continuous action head and proprio projector:

```text
dataset-root
        │
        ├── validate trajectories; bind reusable RynnValue sidecars per episode
        ▼
frozen RynnValue-4B ── absolute/relative heads + exact Analysis output
        │               (immutable, reward-agnostic annotation cache)
        ▼
deterministic reward materialization ── sparse + Shape + Final Reward
        │                              (gamma/kappa/reduction keyed cache)
        ▼
Pixel-IQL critics/value + advantage-weighted VLA behavior cloning
        │
        ▼
policy-registry/<policy_id>/policy.yaml
        ├── standalone LIBERO-X evaluation
        └── selectable policy overlay in the existing Web UI
```

RynnValue follows the pinned official inference implementation and is used only
as an offline trajectory evaluator. Annotation schema v6 retains decoded absolute
and relative temporal distances, both distributional-head logits, absolute-head
entropy, and the exact generated Analysis text/token IDs, but no training
reward. No overlapping-window average is applied, and PBRS is never labeled as
a native model output. A separate deterministic cache derives sparse, Shape,
and Final Reward. Existing compatible schema-v4/v5 sidecars are migrated by
reusing their model heads, even when their old reward settings differ; no new
RynnValue forward is performed. The
PyTorch trainer implements IQL with double-Q
critics, expectile value regression and advantage-weighted behavior cloning; it
does not perform online exploration or modify the upstream VLA-Adapter source.

Reuse the existing `vla-liberox` environment for dataset preparation, IQL and
evaluation; only RynnValue needs the additional `rynnvalue-reward` environment. After
following the installation steps in the [standalone trainer guide](vla-adapter-rynn-iql/README.md),
run from the workspace root:

```bash
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/prepare_dataset.py \
  --config vla-adapter-rynn-iql/configs/liberox_iql.yaml
conda run -n rynnvalue-reward python vla-adapter-rynn-iql/scripts/annotate_rewards.py \
  --config vla-adapter-rynn-iql/configs/liberox_iql.yaml
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/materialize_rewards.py \
  --config vla-adapter-rynn-iql/configs/liberox_iql.yaml
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/train_iql.py \
  --config vla-adapter-rynn-iql/configs/liberox_iql.yaml
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/evaluate.py \
  --config vla-adapter-rynn-iql/configs/inference.yaml
```

The default profile pins the RynnValue source commit and 4B model snapshot,
uses 8×7 action chunks and 8-D proprio at 20 Hz, and is designed for staged use
on a 16 GB GPU. A dataset containing only failures is accepted for integration
testing with an explicit warning, but is not evidence that offline training will
improve task success.

IQL training writes TensorBoard events beside the raw `metrics.jsonl` log. Run
`tensorboard --logdir vla-adapter-rynn-iql/outputs/training` to compare runs;
existing JSONL-only runs can be imported with
`vla-adapter-rynn-iql/scripts/metrics_to_tensorboard.py`.

## Robometer trajectory evaluation

The independent [Robometer evaluator](vla-adapter-robometer/README.md) adds the
official `Robometer-4B-LIBERO` single-trajectory progress and success
probability outputs to existing episodes. In the data page, RynnValue and
Robometer may be selected independently or run sequentially in one GPU job.
Their files, cache validation and overwrite behavior remain independent;
Robometer does not alter IQL rewards or frozen training manifests. Episode
details show both original Robometer curves and an explicitly UI-derived,
time-aligned comparison with normalized RynnValue remaining time.

Robometer uses the separate `robometer-reward` Conda environment. Its pinned
model/source revisions, CUDA device, BF16 mode, 3 Hz evaluation rate, four-frame
prefix protocol and batch-size tuning are documented in the
[evaluator configuration guide](vla-adapter-robometer/README.md#configuration).

### References

- [RynnValue paper — temporal distance and potential-based reward shaping](https://arxiv.org/abs/2608.09853)
- [RynnValue official implementation](https://github.com/alibaba-damo-academy/RynnValue)
- [Implicit Q-Learning paper](https://arxiv.org/abs/2110.06169)
- [VLA-Adapter official implementation](https://github.com/OpenHelix-Team/VLA-Adapter)
- [LIBERO-X official implementation](https://github.com/meituan/LIBERO-X)
- [Robometer paper](https://arxiv.org/abs/2603.02115)
- [Robometer official implementation](https://github.com/robometer/robometer)

## Stage-based reward research

[Stage reward research and ablations](docs/STAGE_REWARD_RESEARCH.md) compares
SARM, STDR, Reward Machines and Relay Policy Learning, then separates two proposed
experiments: stage-potential PBRS and stage-dependent time cost. It covers
failure rollback, uncertain labels, macro/Semi-MDP discount consistency and
controlled evaluation. This is research documentation only: no stage model,
training reward, IQL update or existing evaluation sidecar is changed.
