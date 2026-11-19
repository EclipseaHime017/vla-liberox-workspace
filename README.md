# LIBERO-X Simulation, Takeover and Post-Training Platform

Version **0.8.5** · [Changelog (Chinese)](CHANGELOG.md) · [中文说明](README_CN.md)

Run VLA policies in LIBERO-X simulation, take over by hand and record trajectories, score them with reward models and human keyframes, then post-train the policy with BC or IQL and test it in closed loop. Everything runs from a local web platform; training can also run on multi-GPU servers or PBS clusters.

| | Supported |
| --- | --- |
| Policies | VLA-Adapter (LIBERO-Object-Pro, LIBERO-X), π₀.₅ (LIBERO, LIBERO-X) |
| Methods | BC, IQL; IFQL (π₀.₅: best-of-N action chunks scored by an exported critic) |
| Rewards | sparse, RynnValue (PBRS shaping), Stage (human keyframes), Final Reward fusion; Robometer progress/success scoring |
| Scenes | 21 LIBERO-X scenes (5 tasks × LEVEL1–4); custom scenes via YAML |
| Takeover | SpaceMouse, FACTR force-feedback leader arm |

## Layout

The six top-level directories are Python packages. `studio`, `train`, `rewards` and `cluster` provide commands, run as `python -m <package> <command>`; `core` and `sim` are libraries only:

| Directory | Contents |
| --- | --- |
| `core/` | Shared config, storage layout, model registry, method catalog, upstream lock |
| `sim/` | LIBERO-X simulation and evaluation rollouts |
| `rewards/` | PBRS, Stage, fusion, reward versions; RynnValue and Robometer integrations |
| `train/` | Model families, methods (BC, IQL, IFQL, critics, selectors), replay, training loop, export |
| `cluster/` | Multi-GPU training, queue, PBS, closed-loop and remote evaluation |
| `studio/` | Platform backend (FastAPI) |
| `web/` | Platform frontend (React + Vite) |

Dependencies point one way: `core ← sim, rewards ← train ← cluster, studio`. Configuration lives in `configs/`, per-device environments in `envs/`, and upstream pins and patches in `third_party/`.

## Install

Each machine installs only what its role needs:

```bash
git clone https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace
python envs/setup.py --list                                   # roles
python envs/setup.py --role train --family pi05 --dry-run     # print the plan
python envs/setup.py --role train --family pi05               # install
```

Roles: `dev`, `studio`, `teleop`, `reward-rynnvalue`, `reward-robometer`, `train`, `hpc`, `eval`, `infer`. System packages and manual steps are listed in [docs/install.md](docs/install.md) (Chinese).

## Quick start

```bash
conda activate vla-liberox
python -m studio                      # platform at http://127.0.0.1:8000
python -m train pipeline --dry-run    # single-GPU terminal pipeline (configs/pipeline.yaml)
python -m cluster train               # multi-GPU server training (configs/cluster/pipeline.yaml)
python -m cluster evaluate            # headless closed-loop evaluation
```

## Documentation

Detailed guides are in Chinese under [`docs/`](docs/): install, architecture, studio, training, methods, models, tasks, rewards, evaluation, cluster, data.

AI agents working in this repository start with [docs/agent.md](docs/agent.md); deployment and environment checks follow the [deploy skill](skills/deploy/SKILL.md) (`python skills/deploy/scripts/preflight.py --role <role>` is read-only).

## Tests

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests
conda run -n vla-liberox npm --prefix web test
```
