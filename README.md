# LIBERO-X Local Data Studio

Current release: **v0.4.0**

Local-first simulation, VLA evaluation, trajectory rewind, SpaceMouse takeover,
offline post-training, and reproducible batch policy testing for the three
validated Franka/LIBERO-X tasks.

- Backend: FastAPI application service with a background simulation worker.
- Frontend: React + TypeScript, served by FastAPI after a Vite production build.
- Storage: portable run directories plus a rebuildable SQLite catalog.
- Compatibility: the existing evaluation, intervention, and SpaceMouse CLI scripts remain available.
- Configuration: fixed runtime settings live in [`configs/`](configs/); application code lives in [`liberox-vla-adapter-terminal/`](liberox-vla-adapter-terminal/).
- Operator preview: a transient 2x2 stream shows agent, wrist, −45°, and +45° cameras; VLA input and recorded artifacts remain the original two cameras.
- Run drafts can choose a reproducible random seed and ablate either VLA camera by replacing only that fixed model-input slot with a black frame; raw preview and recording data remain intact.
- Offline post-training: [`vla-adapter-rynn-iql/`](vla-adapter-rynn-iql/) imports the read-only dataset, annotates temporal value with pinned RynnValue, trains a PyTorch IQL overlay, and publishes only the action head and proprio projector to `policy-registry/`.
- Integrated workflow: the Dataset page evaluates RynnValue once per trajectory, preserves its complete output sidecar, paginates run previews, exposes video/action/EEF, absolute/relative remaining-time, observation-potential and entropy estimates, plus Shape/Final Reward details, and independently packages hash-verified training datasets. The Training page derives rewards from those cached model outputs using its selected `gamma`, shaping coefficient, and macro/primitive reduction, then launches resumable IQL jobs without rerunning RynnValue.
- Model registry: a dedicated sidebar page inspects base/overlay metadata and matching training history, and safely renames, copies, or removes local IQL overlays.
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
