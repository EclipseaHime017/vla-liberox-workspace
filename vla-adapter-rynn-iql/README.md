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
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/train_iql.py \
  --config vla-adapter-rynn-iql/configs/liberox_iql.yaml
conda run -n vla-liberox python vla-adapter-rynn-iql/scripts/evaluate.py \
  --config vla-adapter-rynn-iql/configs/inference.yaml
```

Each script loads its adjacent default YAML. `--config` may select an explicit
file for reproducible experiments. Source trajectories are read-only. Branch
prefixes are de-duplicated and only the new suffix contributes extra replay
chunks.

## TensorBoard monitoring

New training runs write both the auditable `metrics.jsonl` stream and
TensorBoard events under `outputs/training/<run>/tensorboard/`. Logging is
controlled by the strict YAML section:

```yaml
logging:
  tensorboard: true
  flush_seconds: 5
```

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
gradient/parameter norms, throughput, and CUDA peak memory. Legacy conversion can only
show fields that existed in the old JSONL. W&B is intentionally not a required
dependency; TensorBoard keeps local runs usable without an account or network.

For a fast CPU test without model downloads:

```bash
conda run -n vla-liberox pytest -q vla-adapter-rynn-iql/tests
```

The UI scans `policy-registry/` for exported `policy.yaml` overlays. An overlay
contains only the action head and proprio projector; it never copies the base
VLA checkpoint.

## Data and reward semantics

Completed original trajectories enter replay once. A branch contributes only
its `resume_step..end` suffix, so the copied parent prefix is never counted
twice. If a fixed-duration recording continues after success, `done` may stay
latched or fluctuate as the object moves out of and back into the goal region.
`data.success_consecutive_steps` debounces this signal (default 5 steps, or
250 ms at 20 Hz). A false sample resets the streak; the action that reaches the
threshold becomes the effective terminal. Unconfirmed pulses are treated as a
failed trajectory. Later actions are excluded from replay and reward annotation
without changing the source NPZ. The manifest retains both raw and debounced
success diagnostics, recorded/effective lengths, and terminal metadata. The importer groups
train/validation splits by root trajectory, validates the N+1 state/image
invariant, and constructs masked 8×7 action chunks.

RynnValue receives only upright `agentview` frames and the BDDL task prompt. At
each action-chunk boundary, the adapter follows the pinned official inference
program: it uniformly resamples the visual prefix ending at that boundary and
reads the last value slot. `annotation_batch_size: 1` is the 16 GB default;
sequences longer than 64 boundaries use overlapping windows. Environment `done` is
the sole success/terminal authority; language `Success` output is diagnostic
only. Chunk reward is the discounted `-1`-until-success sparse return plus
potential shaping `κ(γ^L Φ(s')-Φ(s))`, where `Φ=-remaining_seconds`.

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

- `outputs/work/dataset_manifest.json`: validated read-only replay index and
  source hashes; source runs are never rewritten.
- `outputs/work/rewards/`: resumable RynnValue values, entropy, PBRS rewards and
  diagnostic Analysis text keyed by the data/reward/model hash.
- `outputs/training/<run>/`: metrics, full checkpoints, provenance and effective
  config.
- `policy-registry/<policy_id>/`: immutable action-head and proprio-projector
  components plus a hash-checked `policy.yaml` consumed by the UI.
- `outputs/evaluation/<run>/`: per-policy trajectory NPZ, synchronized
  `agentview.mp4` and `vla_views.mp4`, plus success-rate summary.

No stage silently falls back to CPU after a CUDA OOM; the failing reward,
training, or evaluation stage is named in the exception. A no-success dataset
is accepted only when `data.allow_no_success: true` and always emits a warning.
