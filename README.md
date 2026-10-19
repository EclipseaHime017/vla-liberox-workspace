# LIBERO-X server training

server tracks the main release version; both are currently **v0.7.2**. The branches are maintained independently: version alignment does not merge main's runtime workflow into server.

Workflow: install environments → upload a dataset → configure training → run the launcher → download the trained model. Run commands from the server repository root unless marked as local.

[中文说明](README_CN.md)

[Command reference](docs/RUN_COMMANDS.md) · [Dependencies](docs/DEPENDENCIES.md)

## 1. Environment setup

### Clone the server branch

```bash
mkdir -p ~/eclipseaws
cd ~/eclipseaws
git clone --branch server https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace
```

To update an existing server checkout, first ensure its working tree is clean:

```bash
git fetch origin
git switch server
git pull --ff-only origin server
```

You need NVIDIA GPUs, Conda, Git, and a driver supporting CUDA 12.8. Install the environment for each model you intend to use.

### VLA-Adapter

```bash
conda create -n vla-liberox python=3.10 pip -y
conda activate vla-liberox
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git third_party/VLA-Adapter
git -C third_party/VLA-Adapter checkout 23fa0c9c159e2aa04341cdd3e924f44061311060
python -m pip install -e ./third_party/VLA-Adapter
python -m pip install -r vla-adapter-rynn-iql/requirements-train.txt
python -m pip install -e ./vla-adapter-rynn-iql
python -m pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 setuptools==69.5.1
git -C third_party/VLA-Adapter apply ../../vla-adapter-rynn-iql/vla_adapter_hf_local_autoclass.patch
```

The default base model downloads to `models/VLA-Adapter-LIBERO-Object-Pro/` on first load. Alternatively, copy the complete directory containing the same base weights to that location.

### π₀.₅

```bash
conda create -n pi05 python=3.11 pip -y
conda run -n pi05 python -m pip install uv
conda run --no-capture-output -n pi05 python vla-adapter-rynn-iql/scripts/setup_pi05.py
conda run --no-capture-output -n pi05 python vla-adapter-rynn-iql/scripts/prepare_pi05.py --base-model pi05-libero-base
conda activate pi05
```

The installer places pinned OpenPI sources in `third_party/OpenPI/`. Weights are converted to BF16 by default in `models/pi05_libero_torch/`. Copy an existing complete converted directory instead of converting it again. Other registered variants use their corresponding `--base-model` ID.

Keep the environments separate. Run the launcher from an installed training environment; workers use the environment declared by the selected model. Reusing exported evaluations and rewards does not require RynnValue or Robometer installations.

## 2. Upload and import a dataset

Prepare a complete exported dataset ZIP or directory, including its manifests, trajectories, and evaluations. This guide uses `dataset-root/imports/` on the server.

Run locally after replacing the host and repository path:

```bash
TRAIN_HOST=user@server
TRAIN_REPO=/home/user/eclipseaws/vla-liberox-workspace
ssh "$TRAIN_HOST" "mkdir -p '$TRAIN_REPO/dataset-root/imports'"
scp ./task-export.zip "$TRAIN_HOST:$TRAIN_REPO/dataset-root/imports/"
```

For a directory, replace the last command with:

```bash
scp -r ./task-export "$TRAIN_HOST:$TRAIN_REPO/dataset-root/imports/"
```

Dataset exports retain their membership and train/validation splits. Complete runs directories are also supported, using their existing trajectory selection marks. ZIP inputs are temporarily extracted for training and cleaned up afterwards; manual extraction is unnecessary.

## 3. Configure training

Edit [server_pipeline.yaml](vla-adapter-rynn-iql/configs/server_pipeline.yaml). This example uses VLA-Adapter, IQL, and eight GPUs:

```yaml
schema_version: 4
training_config: ./training/iql.yaml
runs_root: ../../dataset-root/imports/task-export.zip
task_id: null
output_root: ../outputs/server-training
cache_root: ../outputs/server-replay-cache
reward_source: final

distributed:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  backend: nccl
  zero_stage: 1
  timeout_seconds: 1800
  data_workers_per_rank: 2
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

model_profiles:
  vla_adapter:
    training:
      micro_batch_size: 64
      gradient_accumulation_steps: 1
    model:
      backbone: lora
      lora: {rank: 64}
  pi05:
    training:
      micro_batch_size: 32
      gradient_accumulation_steps: 1
    model:
      backbone: full

overrides:
  model:
    family: vla_adapter
    base_id: base
  training:
    method: iql
    train_steps: 10000
    checkpoint_interval: 512
  reward: {}
```

Relative paths are resolved against their configuration file. Set `runs_root` to a directory when importing a directory. Leave `task_id: null` for a single task; otherwise set the canonical task ID or pass `--task`.

### GPUs and batch size

- `gpu_ids` selects 1–8 GPUs on one machine. If `CUDA_VISIBLE_DEVICES` is already set, these indices refer to that visible-device list.
- `micro_batch_size` is global and must be divisible by the GPU count. Per-GPU batch size is global batch ÷ GPU count.
- Effective actor batch size = micro batch × gradient accumulation. Defaults use no accumulation: VLA-Adapter uses global batch 64 (eight per GPU), while π₀.₅ uses 32 (four per GPU) on eight GPUs.
- `checkpoint_interval` must be divisible by accumulation steps. Changing the GPU count does not automatically change training steps or learning rates.

`model_profiles` supplies server defaults for the selected model family, shared by BC and IQL. Explicit `overrides` take precedence. Training remains at 10,000 steps; learning rates, optimizers, and rewards are unchanged. Without accumulation, the actor updates every step, increasing total actor updates and sampled windows over the old defaults. Large-batch/full-model GPU memory requirements must be measured; the runner never silently reduces the batch or enables accumulation.

### Model and training method

For π₀.₅, replace `overrides.model` with:

```yaml
model:
  family: pi05
  base_id: pi05-libero-base
```

By default, VLA-Adapter uses rank-64 LoRA along with trainable action queries, action head, and proprio projector; π₀.₅ uses full fine-tuning. Registered variants, including those selected with `--model`, use their family's profile. VLA-Adapter still supports `frozen/lora/full`; π₀.₅ supports `frozen/full`, where frozen trains the action expert and projections. See [Model.md](docs/Model.md) for further model settings.

For BC, set `training_config: ./training/bc.yaml` and `overrides.training.method: bc`. BC learns from dataset actions without rewards or Q/V training. Select or filter samples when preparing the dataset, not in the training method.

IQL accepts `reward_source: final/rynnvalue/stage/sparse`. Existing evaluations are reused; missing required rewards stop the job rather than triggering model evaluation. Exported results supply p, α, κ, and other reward-recipe settings. Training only permits these overrides under `overrides.reward`:

```yaml
reward:
  gamma: 0.99
  accumulate_primitive_steps: false
```

Keep `reward: {}` to inherit exported settings. Cumulative reward must be Off for multiplicative Final Reward.

## 4. Run training

Activate an installed training environment, then validate the input and configuration:

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml --dry-run
```

Start training:

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml
```

You can also override the model and method for a run:

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --model pi05-libero-base --method bc
```

The launcher validates inputs, loads rewards, prepares the mmap cache, and starts DDP + ZeRO-1. Use tmux for long remote jobs so an SSH disconnect does not stop training.

### Training queue

Submit experiments and start one detached FIFO worker. Jobs run sequentially and survive terminal disconnects:

```bash
python vla-adapter-rynn-iql/scripts/train_queue.py add --config vla-adapter-rynn-iql/configs/server_pipeline.yaml --name baseline-iql
python vla-adapter-rynn-iql/scripts/train_queue.py add --config vla-adapter-rynn-iql/configs/server_pipeline.yaml --method bc --name baseline-bc
python vla-adapter-rynn-iql/scripts/train_queue.py start
python vla-adapter-rynn-iql/scripts/train_queue.py list
```

Submission freezes resolved configuration, model revision, observation contract, data/reward fingerprints, and any resume checkpoint. Later YAML edits do not change pending jobs. Changed inputs or implementation code require a new submission rather than silently changing the experiment. Enqueue validates and hashes assets without loading a GPU model.

State defaults to `vla-adapter-rynn-iql/outputs/server-queue/`, with daemon output in `worker.log`; experiments retain separate training runs. Pass `--queue /path/to/queue` before the subcommand to use another state directory.

`train_queue.py stop` finishes the active job and stops before the next, preserving pending jobs. `cancel <job-ID>` cancels a pending job or safely stops an active one; `retry <job-ID>` appends an explicit retry with the same configuration. Failures do not block later jobs; interrupted jobs are never automatically repeated. An empty daemon stays idle and accepts further submissions.

Direct training and all queues share one local-account training lease. Surviving workers continue to block competing jobs after supervisor failure. Stop and restart the queue worker after code updates. Use `train_queue.py run` for foreground execution; Ctrl+C safely stops the active training and the worker.

### Progress, stopping, and resuming

Default run directories are `vla-adapter-rynn-iql/outputs/server-training/server_<timestamp>_<ID>/`. The terminal prints the run path and progress. From another terminal:

```bash
RUN="vla-adapter-rynn-iql/outputs/server-training/server_<timestamp>_<ID>"
tail -f "$RUN/training.log"
tail -n 1 "$RUN/metrics.jsonl"
```

TensorBoard is enabled by default. For W&B, run `wandb login` in the model's training environment and set `overrides.logging.wandb.enabled: true`.

Ctrl+C requests a checkpoint at a safe gradient-accumulation boundary. To resume, set:

```yaml
overrides:
  training:
    resume_checkpoint: /absolute/path/to/checkpoints/step_00000512
```

Keep the rest of the original configuration and set `train_steps` above the completed step. Model, data, rewards, global batch, accumulation, and seed must remain consistent. GPU count may change if the global batch remains divisible.

## 5. Export and download the trained model

After completion, `policy_overlay` in `$RUN/summary.json` points to `policy-registry/<policy-ID>/policy.yaml`. Download its entire model directory for inference, not just the YAML file or a training checkpoint.

On the server, create a deployment copy. Replace `TARGET_BASE` with the corresponding base-weight location on your local device:

```bash
POLICY_MANIFEST=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_overlay"])' "$RUN/summary.json")
POLICY_ID=$(basename "$(dirname "$POLICY_MANIFEST")")
TARGET_BASE=/home/localuser/eclipseaws/vla-liberox-workspace/models/VLA-Adapter-LIBERO-Object-Pro

python vla-adapter-rynn-iql/scripts/export_policy.py \
  --policy "$POLICY_MANIFEST" \
  --output "$PWD/vla-adapter-rynn-iql/outputs/deploy/$POLICY_ID" \
  --base-checkpoint "$TARGET_BASE"
```

For π₀.₅, use the matching base-weight directory, such as `models/pi05_libero_torch/`. Base-weight contents must match between machines. Deployment export leaves the original training result unchanged and requires a new output directory.

From your local repository root, enter the policy ID obtained above and download:

```bash
TRAIN_HOST=user@server
TRAIN_REPO=/home/user/eclipseaws/vla-liberox-workspace
POLICY_ID=your-policy-ID
mkdir -p policy-registry
scp -r "$TRAIN_HOST:$TRAIN_REPO/vla-adapter-rynn-iql/outputs/deploy/$POLICY_ID" ./policy-registry/
```

Keep the name and all files of `policy-registry/<policy-ID>/`. Retain server run directories and checkpoints for further training; local inference does not require training checkpoints.

## 6. Server architecture

```text
server_pipeline.yaml + train_server.py / train_queue.py
  → select model variant and BC/IQL method
  → validate imported data and freeze members, splits, and rewards
  → build or reuse the mmap replay cache
  → start torchrun in the selected model environment
  → DDP training with ZeRO-1 optimizers
  → save checkpoints, logs, and deployment models
```

`server_inputs.py` handles inputs and rewards, `server_pipeline.py` manages configuration and processes, `server_queue.py` persists the queue, and `server_resources.py` prevents overlapping training. `server_replay.py` owns caching and sampling; `distributed_training.py` runs training and saves results.

Model YAMLs and adapters own weights, preprocessing, native losses, and adaptation scopes. Training methods determine how losses are used; the distributed layer synchronizes gradients and shards optimizer state. Rank 0 writes logs and publishes results, while every rank participates in synchronization and checkpoint consolidation. ZeRO groups parameters by dtype to support mixed BF16/FP32 models.

server has no frontend, database, or controller dependency. Imported source data remains read-only; each run saves its own input and reward snapshots and reuses evaluations without running evaluator models. The mmap cache reduces repeated observation decompression during training. Local maintenance notes belong in the untracked `logs/SERVER_MAINTENANCE.md`.
