# LIBERO-X Local Data Studio

Local-first simulation, VLA evaluation, trajectory rewind, and SpaceMouse takeover for the three validated Franka/LIBERO-X tasks.

- Backend: FastAPI application service with a background simulation worker.
- Frontend: React + TypeScript, served by FastAPI after a Vite production build.
- Storage: portable run directories plus a rebuildable SQLite catalog.
- Compatibility: the existing evaluation, intervention, and SpaceMouse CLI scripts remain available.
- Configuration: fixed runtime settings live in [`configs/`](configs/); application code lives in [`liberox-vla-adapter-terminal/`](liberox-vla-adapter-terminal/).
- Operator preview: a transient 2x2 stream shows agent, wrist, −45°, and +45° cameras; VLA input and recorded artifacts remain the original two cameras.
- Offline post-training: [`vla-adapter-rynn-iql/`](vla-adapter-rynn-iql/) imports the read-only dataset, annotates temporal value with pinned RynnValue, trains a PyTorch IQL overlay, and publishes only the action head and proprio projector to `policy-registry/`.

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
never rewrites `dataset-root`; their only runtime integration boundary is a
hash-checked `policy.yaml` overlay published to `policy-registry/`.

Start from `vla-liberox-workspace/` after activating `vla-liberox`:

```bash
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

On a new checkout, run `npm ci` once in `liberox-vla-adapter-terminal/frontend/`. The launcher fingerprints the frontend sources, prints the exact build command before running it, and automatically rebuilds the Git-ignored `frontend/dist` after later pulls. Run `npm run build` there for a manual source-only rebuild, or `npm ci && npm run build` after `package-lock.json` changes. Neither `npm run build` nor `npm test` installs or upgrades dependencies.

Open <http://127.0.0.1:8000>. See [README_CN.md](README_CN.md) for setup and operation, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries, and [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for persistence rules.

## RynnValue + IQL offline post-training

The standalone pipeline keeps `VLA-Adapter/LIBERO-Object-Pro` as the deployed
Franka policy, freezes its vision/language backbone, and updates only the
continuous action head and proprio projector:

```text
dataset-root (read-only)
        │
        ├── validate trajectories and de-duplicate branch prefixes
        ▼
frozen RynnValue-4B ── temporal distance ── PBRS chunk rewards
        │
        ▼
Pixel-IQL critics/value + advantage-weighted VLA behavior cloning
        │
        ▼
policy-registry/<policy_id>/policy.yaml
        ├── standalone LIBERO-X evaluation
        └── selectable policy overlay in the existing Web UI
```

RynnValue follows the pinned official inference implementation and is used only
as an offline reward annotator. The PyTorch trainer implements IQL with double-Q
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

### References

- [RynnValue paper — temporal distance and potential-based reward shaping](https://arxiv.org/abs/2608.09853)
- [RynnValue official implementation](https://github.com/alibaba-damo-academy/RynnValue)
- [Implicit Q-Learning paper](https://arxiv.org/abs/2110.06169)
- [VLA-Adapter official implementation](https://github.com/OpenHelix-Team/VLA-Adapter)
- [LIBERO-X official implementation](https://github.com/meituan/LIBERO-X)
