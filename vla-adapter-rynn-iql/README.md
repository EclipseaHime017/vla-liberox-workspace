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
twice. If a fixed-duration recording latches `done=True` after success, the
importer keeps the first terminal action and excludes only the post-terminal
tail from replay and reward annotation without changing the source NPZ. The
manifest records both `recorded_action_count` and the effective `action_count`;
a non-monotonic `done` sequence is rejected. The importer groups
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
