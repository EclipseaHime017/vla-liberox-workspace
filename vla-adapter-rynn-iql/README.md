# VLA-Adapter RynnValue IQL

Standalone offline post-training for the Franka VLA-Adapter policy. RynnValue
is a frozen offline reward annotator; the deployed policy remains
`VLA-Adapter/LIBERO-Object-Pro` with an IQL-trained action-head overlay.

## Environments

Reuse the existing Python 3.10 `vla-liberox` environment for dataset preparation,
IQL training and evaluation. RynnValue requires a newer Transformers stack, so
only its reward annotator gets a new environment.

```bash
conda create -n rynnvalue-reward python=3.10 -y
conda run -n rynnvalue-reward pip install -r vla-adapter-rynn-iql/requirements-reward.txt

# Add only the lightweight trainer package to the existing VLA/LIBERO environment.
conda run -n vla-liberox pip install -r vla-adapter-rynn-iql/requirements-train.txt
conda run -n vla-liberox pip install -e ./vla-adapter-rynn-iql
```

Clone the official RynnValue source at the commit pinned in
`configs/dependency-lock.yaml` (currently `10e0d333…`), then verify it:

```bash
git clone https://github.com/alibaba-damo-academy/RynnValue.git ./RynnValue
git -C ./RynnValue checkout 10e0d333f5f3811d0d130587e50f1faf48da49e5
conda run -n rynnvalue-reward python vla-adapter-rynn-iql/scripts/verify_reward_environment.py \
  --checkout ./RynnValue
```

Do not run `pip install -e ./RynnValue`: the pinned upstream project declares
`tool.uv.package = false` and is intentionally not an editable setuptools
distribution. `paths.rynnvalue_root` in `liberox_iql.yaml` points the annotator
at this audited source checkout directly; no persistent `PYTHONPATH` is needed.

`liberox_iql.yaml` pins the 4B Hugging Face snapshot to
`3f73b5d2b5e53b21f248c8791004dde6a8cf2b92`. The annotator imports the audited
local model classes, loads the immutable snapshot with
`trust_remote_code=False`, and records the code commit, resolved snapshot and
model-file SHA-256 hashes in the reward cache.

## Pipeline

Run from `vla-liberox-workspace/`:

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

Each script loads its adjacent default YAML. `--config` may select an explicit
file for reproducible experiments. Source trajectories are read-only. Every
branch retains its complete physical trajectory from step zero through the new
suffix so RynnValue can score the natural rollout before takeover. The training
replay de-duplicates physically copied parent prefixes across the parent and
sibling branches.

For terminal-only training on a remote machine, use the stateful orchestrator;
it does not import or start the web backend or frontend:

```bash
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml
```

The terminal YAML selects exactly one task and supports quota, random-size, and
all-eligible membership. It can override data, reward, VLA, IQL, logging, and
non-managed path settings from `liberox_iql.yaml`. Before execution it prints a
reproducible selection and stage plan and asks for confirmation; pass `--yes`
for SSH batch jobs or `--dry-run` to validate without creating pipeline output.
Its A100-oriented default uses `micro_batch_size: 8` and
`gradient_accumulation_steps: 4`, retaining an actor effective batch of 32.
The base 16 GB-compatible config remains `1 x 32`. Batched VLA processing is
restricted to a prepared training split with exactly one task and prompt;
legacy multi-task manifests remain supported with micro batch one.

Prepare is skipped only when its selection/source/structural fingerprint and
manifest hash match. RynnValue annotation and deterministic reward reduction
use separate caches. The first cache depends only on model inference inputs and
settings; the second depends on `gamma`, shaping weight, and reward reduction
mode. Changing only reward semantics therefore performs a fast CPU rebuild and
never reloads RynnValue. Completed annotations are atomically bound beside each
source trajectory as
`rynnvalue_evaluation.{json,npz}`. Training always creates a new run unless an
explicit `resume_checkpoint` override is provided. `--force-prepare` and
`--force-annotate` are available for deliberate rebuilds. Pipeline state,
effective configuration, timings, cache decisions, and the resulting overlay
are recorded below `outputs/terminal-pipelines/`.

The older `run_pipeline.py` remains a simple stateless stage launcher. Prefer
`train_terminal.py` for unattended or resumable remote workflows.

### Isolated single-node multi-GPU server trainer

The `server` Git branch adds an isolated DDP + PyTorch ZeRO-1 entry point. It
does not replace `train_iql.py`, `training.py`, or the UI job path; those remain
the original single-process trainer. Select 1–8 physical GPUs explicitly in
`configs/server_pipeline.yaml`, then inspect and launch the plan:

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml \
  --dry-run

python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml
```

The server pipeline performs the same selection, Prepare, and RynnValue binding
stages serially before launching `torchrun`. Only IQL optimization is
distributed. Q, V, and the actor overlay each use DDP gradient synchronization
and an independent `ZeroRedundancyOptimizer`; every rank retains its own frozen
VLA backbone. Target Q remains local and is updated identically from the
synchronized online Q replicas.

In server mode, `iql.micro_batch_size` is the **global** micro batch and must be
divisible by the number of configured GPUs. For example, eight GPUs with global
micro batch 8 use one transition per rank. With
`gradient_accumulation_steps: 4`, the actor effective global batch is 32;
`train_steps`, the learning-rate schedule, and accumulation semantics do not
change with world size.

Before `torchrun`, `build_server_cache.py` materializes only the de-duplicated
training chunks into hash-keyed read-only `.npy` mmap arrays. Actor images and
critic current/next images are shared through the OS page cache, while compact
actions, proprioception, masks, and rewards are assembled once in each rank.
Changing the dataset, critic image size, or source image hashes creates a
different mmap cache. Reward-only changes reuse the image cache. Source
trajectories and RynnValue sidecars remain read-only.

Server and single-GPU training share the same reward-reduction switch. Set
`reward.rynnvalue: false` for a sparse-reward-only ablation; this keeps the
stored RynnValue evaluation outputs for diagnostics but sets the dense reward
to zero and makes the final training reward equal the sparse reward. The
default is `true`. Set
`reward.accumulate_primitive_steps: false` (the default) to treat each action
chunk as one macro transition with one sparse reward and one Bellman discount.
Set it to `true` to accumulate discounted primitive-step rewards and bootstrap
with `gamma ** chunk_length`. Changing either switch, `reward.gamma`, or
`reward.shaping_weight` rebuilds only the deterministic second-level reward
cache from existing RynnValue outputs; it does not rerun the model.

Only rank zero writes JSONL, TensorBoard, W&B, checkpoints, and the standard
policy overlay. Checkpoint save first consolidates all three ZeRO optimizer
states on rank zero; unwrapped model keys allow a server checkpoint to resume
with a different 1–8 GPU world size. `Ctrl+C` is handled at a shared safe step
boundary and records one diagnostic checkpoint.

Deploy this branch on the training host with:

```bash
git fetch origin
git switch server
git pull --ff-only origin server
```

The implementation follows PyTorch's documented one-process-per-GPU DDP model
and its supported integration with `ZeroRedundancyOptimizer`:
[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[distributed optimizers](https://docs.pytorch.org/docs/stable/distributed.optim.html).

The LIBERO Studio UI can generate `data.selection_manifest` automatically from
an immutable, single-task dataset version. In that mode prepare does not scan
the rest of `dataset-root`: it verifies and imports exactly the listed members,
hashes, segment boundaries, and frozen train/validation split. The UI runs
prepare/training with `vla-liberox` and annotation with `rynnvalue-reward`; it
does not merge either dependency stack or pass browser-provided shell commands.

Creating or deriving a dataset in the UI still runs preparation. The backend
launches a detached three-stage job: `prepare_dataset.py` first materializes that
frozen selection's transition/split manifest under
`datasets/<dataset_id>/annotations/<job_id>/work/`, then
`annotate_rewards.py` resolves or computes its RynnValue entries, and
`materialize_rewards.py` creates the default deterministic reward cache. Durable
per-trajectory evaluations seed the content cache before this job, so existing
model evaluations are reused; preparation itself is still required because a
dataset version has its own members, interrupted chunks, terminal threshold,
and root-grouped split.

## Training monitoring

New training runs write both the auditable `metrics.jsonl` stream and
TensorBoard events under `outputs/training/<run>/tensorboard/`. Logging is
controlled by the strict YAML section:

```yaml
logging:
  tensorboard: true
  wandb:
    enabled: false
    mode: online
    project: vla-adapter-rynn-iql
    entity: null
    run_name: null
    group: null
    tags: [liberox, rynnvalue, iql]
    log_interval_steps: 10
  flush_seconds: 5
  console_interval_steps: 10
```

The terminal prints the first step, every `console_interval_steps`, and the
final step. Each line contains the progress bar, current/total steps, training
phase, elapsed time, rolling ETA and estimated finish time, throughput, the
Q/value/actor losses, Q/V/advantage means, advantage weight, actor learning
rate, and peak allocated CUDA memory. The rolling ETA uses the most recent 100
steps, so checkpoint pauses and early startup do not permanently distort it.
Both `steps_per_second` and `samples_per_second` are recorded; use the latter
when comparing runs with different micro-batch sizes.

Start the local viewer from the workspace root:

```bash
conda run -n vla-liberox tensorboard \
  --logdir vla-adapter-rynn-iql/outputs/training \
  --host 127.0.0.1 --port 6006
```

Then open `http://127.0.0.1:6006`. For a completed run created before
TensorBoard logging was added, convert its existing JSONL metrics without
retraining:

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/metrics_to_tensorboard.py \
  --run-dir vla-adapter-rynn-iql/outputs/training/<run>
```

The dashboards group Q/value/actor losses, IQL advantage weights, all seven
action-axis L1 errors, gripper predictions and targets, actor learning rate and
gradient/parameter norms, progress/ETA, throughput, and CUDA peak memory. Legacy
conversion can only show fields that existed in the old JSONL.

For remote monitoring, install `requirements-train.txt`, run `wandb login`, and
set `logging.wandb.enabled: true`. Online mode fails clearly when credentials
are unavailable rather than silently changing modes. Air-gapped servers can use
`mode: offline` and later run `wandb sync <run>/wandb/offline-run-*`. W&B uses
the same grouped metric names as TensorBoard and writes connection metadata to
`<run>/wandb.json`. `logging.wandb.log_interval_steps` controls network logging
frequency without changing the per-step JSONL/TensorBoard records. See the
root Chinese README sections 4.4.8 and 4.4.9 for complete configuration and
8xA100 deployment guidance.

On `main`, the UI and `train_terminal.py` remain single-process and single-GPU;
one GPU can still process multiple same-task replay transitions per forward.
The `server` branch's `train_server.py` is the only entry point that distributes
one training run. Merely exposing multiple devices to the original entry points
does not enable DDP.

For a fast CPU test without model downloads:

```bash
conda run -n vla-liberox pytest -q vla-adapter-rynn-iql/tests
```

The UI scans `policy-registry/` for exported `policy.yaml` overlays. An overlay
contains only the action head and proprio projector; it never copies the base
VLA checkpoint.

## Data and reward semantics

Completed original and branch trajectories are evaluated from step zero. If
takeover occurs inside a nominal 8-step chunk, the final policy transition ends
at `resume_step` with its actual `chunk_length=E`, followed by separate `human`
or `policy_requery` transitions. No transition crosses an `action_source`
boundary. When the training replay is assembled, physically identical parent
prefix transitions are represented once across the parent and sibling branches. Thus the
error-policy continuation and the intervention alternative remain available
from the same state without treating padding as executed time. If a fixed-duration recording continues after success, `done` may stay
latched or fluctuate as the object moves out of and back into the goal region.
`data.success_consecutive_steps` debounces this signal (default 5 steps, or
250 ms at 20 Hz). A false sample resets the streak; the action that reaches the
threshold becomes the effective terminal. Unconfirmed pulses are treated as a
failed trajectory. Later actions are excluded from replay but remain in the
full-trajectory RynnValue evaluation without changing the source NPZ. The
manifest retains both raw and debounced
success diagnostics, recorded/effective lengths, transition source/type, and terminal metadata. The importer groups
train/validation splits by root trajectory, validates the N+1 state/image
invariant, and constructs masked 8×7 tensors for variable-duration chunks of
at most eight actions. Each recorded chunk is one IQL macro-action decision:
the actual `chunk_length` selects `s[t+L]` and controls the action mask, while
the sparse reward and Bellman discount are each applied once per chunk by
default. `reward.accumulate_primitive_steps: true` selects discounted
primitive-step accumulation with an actual-duration Bellman discount.

RynnValue receives only upright `agentview` frames and the BDDL task prompt. At
each action-chunk boundary, the adapter follows the pinned official inference
program: it uniformly resamples the visual prefix ending at that boundary and
reads the last value slot. `annotation_batch_size: 1` is the 16 GB default;
every boundary is evaluated from its complete prefix, so long trajectories are
not merged by averaging overlapping windows. Annotation schema v6 records only the
official decoded absolute distance, absolute logits/entropy, decoded relative
distance, relative logits, and exact generated Analysis text/token IDs. Parsed
Description / Match / Success values are display-only. Environment `done` is
the sole success/terminal authority.

Reward materialization is a separate deterministic CPU stage. It reads those
immutable official outputs and writes a second-level cache containing
`pbrs_shaping_reward` (the raw Shape Reward `γΦ(s')-Φ(s)`) and
`pbrs_chunk_reward` (the Final Reward `r_sparse+κ·r_shape`). Here
`r_sparse` is `-1` for an incomplete macro action and `0` when that chunk
completes the task, and `Φ=-absolute temporal distance`. Its cache key includes
`gamma`, `shaping_weight`, and `accumulate_primitive_steps`; changing any of
them recomputes only these inexpensive arrays. Hash-valid schema-v4/v5
sidecars reuse their complete official model heads during migration, regardless
of the reward reduction stored beside them, so RynnValue is not run again. Chunks recorded
after the confirmed terminal are inspection-only and never enter ReplayDataset.

The training default uses four uniformly sampled prefix frames, matching the
offline reward-relabeling protocol in paper Appendix B.3. Upstream's standalone
trend-video demo defaults to 64 frames; that demo default is not the paper's
IQL relabeling setting.

IQL update semantics follow the pinned `pi-rl` implementation: V is first fit
by expectile regression against the minimum frozen target Q, online Q is then
fit with the newly updated next-state V, target Q receives a Polyak update, and
policy weights use the updated online `min(Q1,Q2)-V`. Auxiliary Q/V networks use
Adam without weight decay. The VLA component optimizer uses AdamW with
`betas=(0.9,0.95)`, `eps=1e-8`, weight decay `1e-10`, and gradient clipping at
1.0.

This is an algorithm-compatible port, not a layer-for-layer reproduction of
the paper's π0.5 experiment. Like the paper, one predicted action chunk is one
IQL decision step and receives one reward/discount. The paper uses a 16-step flow-matching policy,
224px ResNet-18 IQL encoder, absolute joint actions, and no proprioception. The
Object-Pro integration necessarily uses masked L1 on its 8×7 continuous action
head, 8D proprio, normalized OSC_POSE execution, and a lighter two-view critic
for the validated 16 GB profile. Interrupted or terminal chunks retain their
actual `L` for padding masks and `s[t+L]`, but they are still a single macro
decision and therefore do not introduce `gamma ** L`.
The same profile does not claim paper-exact batch 64, 224px inputs, random-crop
augmentation, 2,000-step policy-LR warmup, or policy EMA 0.99; those require a
separate reproduction profile instead of being silently attributed to this
VLA-Adapter port.

The pinned 4B checkpoint is a BF16 RynnValue model, not a separately quantized
Qwen model. Its Qwen text hidden width is 2560; eight consecutive value-token
states are concatenated into the dedicated value head's 20480-wide input. The
upstream custom value-head constructors default to FP32, so this adapter casts
the **complete** loaded model (backbone and value heads) to the configured BF16
dtype and verifies every floating parameter before annotation. Value-bin
decoding and entropy softmax still run in FP32 for numerical stability.

Set `iql.resume_checkpoint` to a saved `step_XXXXXXXX` directory to resume Q/V,
targets, actor components, optimizers, replay sampler and RNG state. Checkpoint
intervals must be divisible by gradient accumulation so no partial actor
gradient is lost. Every checkpoint also stores the resolved effective YAML,
dataset/reward hashes and workspace Git commit.

## Outputs and safety boundaries

- `outputs/work/dataset_manifest.json`: schema-v4 validated read-only replay
  index, variable-duration transition metadata and source hashes; source runs
  are never rewritten. Older prepared manifests must be prepared and annotated
  again before training.
- `paths.annotation_cache/<content_hash>.{npz,json}`: atomic, per-trajectory
  official absolute/relative decoded values and logits, absolute entropy, exact
  Analysis generation, and evaluator provenance. These files contain no sparse,
  Shape, or Final Reward arrays.
  Dataset identity is excluded from the key, so derived dataset versions reuse
  unchanged members.
- `outputs/work/annotations/annotation_manifest.json`: the current prepared
  dataset's complete reference index into the immutable RynnValue outputs.
- `outputs/work/rewards/<reward_hash>.npz` and `reward_manifest.json`: the
  deterministic second-level reward cache for the active `gamma`, `kappa`, and
  macro/primitive-step reduction. Exact repeats are reused; a mismatch is
  rebuilt from annotations without loading RynnValue.
- `outputs/training/<run>/`: metrics, full checkpoints, provenance and effective
  config.
- `policy-registry/<policy_id>/`: immutable action-head and proprio-projector
  components plus a hash-checked `policy.yaml` consumed by the UI.
- `outputs/evaluation/<run>/`: per-policy trajectory NPZ, synchronized
  `agentview.mp4` and `vla_views.mp4`, plus success-rate summary.

No stage silently falls back to CPU after a CUDA OOM; the failing reward,
training, or evaluation stage is named in the exception. A no-success dataset
is accepted only when `data.allow_no_success: true` and always emits a warning.

Platform-managed datasets, jobs, annotations and training outputs live under
`dataset-root/projects/<project_id>/`; see the workspace
[`docs/DATA_LAYOUT.md`](../docs/DATA_LAYOUT.md). Their JSON/YAML files remain the
recoverable source of truth while SQLite is only a rebuildable query index.
