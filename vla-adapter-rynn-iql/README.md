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

The LIBERO Studio UI can generate `data.selection_manifest` automatically from
an immutable, single-task dataset version. In that mode prepare does not scan
the rest of `dataset-root`: it verifies and imports exactly the listed members,
hashes, segment boundaries, and frozen train/validation split. The UI runs
prepare/training with `vla-liberox` and annotation with `rynnvalue-reward`; it
does not merge either dependency stack or pass browser-provided shell commands.

Creating or deriving a dataset freezes membership and splits, without model
evaluation. Expand its configuration to generate Final Reward from saved inputs,
evaluate RynnValue/Robometer individually, or run All. Each job prepares the
frozen selection's transition manifest under
`datasets/<dataset_id>/annotations/<job_id>/work/`. All serially reruns both
models before generating Final Reward and publishes only after every stage
succeeds. Individual Final Reward jobs never load a model. Dataset results take
precedence over global results of the same type; absent dataset results inherit
global ones. Active training pins a snapshot, independent of later reevaluation.

## Final Reward and human keyframes

Use fusion parameters instead of a training reward-source selector:

```yaml
reward:
  fusion_mode: additive  # additive | multiplicative
  final_normalization: initial_chunk_v1
  alpha: 0.0
  shaping_weight: 0.1    # kappa
  stage_exponent: 2.0
  gamma: 0.99
  accumulate_primitive_steps: false
```

Let B be sparse reward, S the Stage score and F the unweighted RynnValue PBRS.
For macro actions:

```text
Original Final Reward = B + kappa*F
additive:       R_raw = (1-alpha)*B + alpha*S + kappa*F
multiplicative: R_raw = (-S)*(B + kappa*F)
Final[i] = R_raw[i] / (-R_raw[0]), requiring R_raw[0] < 0
```

Before rescaling, additive alpha=0/kappa=0 recovers Sparse; alpha=0/kappa>0
recovers RynnValue; alpha=1/kappa=0 recovers Stage. Multiplication ignores alpha and has no positive
multiplier floor. It is **macro-only**, using the chunk endpoint Stage score and
boundary potentials; no primitive potential interpolation is performed.
Cumulative is hidden for multiplication and normalized to false in effective
YAML/API settings, so both shaping and Bellman use gamma. Additive retains the
existing cumulative option with gamma^L shaping and bootstrap. If inherited
global results include any multiplication, UI training uses macro for the whole
run; conflicting cumulative CLI requests fail explicitly.

New results use `reward.final_normalization: initial_chunk_v1`: divide by the
negative raw reward of the full recording's first chunk, so the first Final is
exactly -1. Branches use their full prefix, not their deduplicated replay start.
A nonnegative first value is an error; negative near-zero values have no cutoff.
No shifting or clipping is applied. This preserves zero but does not bound later
values to [-1,0] or guarantee monotonicity, and changes relative reward scales
between trajectories. Plots and training use the same normalized array; Original
Final, Stage and model outputs are untouched. Artifacts retain `raw_final_reward`,
`final_reward_reference` and `final_reward_scale` for inspection.

Regenerate Final Reward to apply the rule to existing results, without relabeling
keyframes or rerunning models. Historical snapshots without the field retain
`none` (unscaled) semantics. Training gamma/cumulative changes recompute fusion
and its reference from saved signals into private training artifacts, without
modifying saved evaluations or scaling their already normalized arrays again.
This fusion is a project experiment, not an official
RynnValue output or a policy-invariance guarantee.

Additive alpha=0 needs no Stage; kappa=0 needs no RynnValue outputs. Missing
required signals abort rather than downgrade or load a model during Final
materialization. With kappa=0, use `prepare_dataset.py → materialize_rewards.py
→ train_iql.py` in `vla-liberox`; no reward-model environment is required.
Terminal entry points choose stages from these dependencies. Old explicit
`reward.source` and legacy `reward.rynnvalue` YAML retain their original semantics;
remove those selectors when adopting fusion. Robometer remains diagnostic only.

Mark Positive/Negative observation steps in the Dataset trajectory detail's
keyframe editor, then save. This writes only `stage_annotation.json`, never cuts
source actions, observations or video. Initial score is −1. With P positive and
N negative manual marks, the increment is `1/(P-N+1)` for confirmed success,
`1/(P+1)` for failure. Success adds one automatic positive anchor and fixes its
score to 0; the failed tail holds its last score. Values are **not clipped**;
scores below -1 remain valid, but Final generation rejects positive S and asks
for corrected labels. Saving keyframes updates the Stage preview, not Final.
Between anchors, interpolate using `z=z_start+(z_end-z_start)*x^p`, default p=2.
Generate Final separately in dataset configuration using the selected exponent.

Stage is a direct score, not a PBRS difference: macro S=z(t+L), cumulative
S=Σ gamma^h z(t+h+1). IQL updates, action masks, replay deduplication and success
confirmation are unchanged. When Stage is required, all members need valid
saved labels, including failures and
validation members. Missing/stale labels abort before VLA loading; an explicitly
saved empty failed annotation is valid and gives −1 throughout.

UI/terminal jobs freeze annotations; training outputs retain `stage_annotations.json`
and `reward_manifest.json` for audit. Re-editing labels cannot alter a started
run. Dataset configuration owns p, alpha, kappa and fusion mode. Gamma and
cumulative (additive only) can be set there and at training launch; overrides
derive private arrays from frozen signals without overwriting dataset results.
Annotation hashes bind original `trajectory.npz` bytes and keyframes, not p;
success is resolved with the dataset's confirmation threshold. A ZIP retaining
original NPZ files and sidecars is
supported; the UI's existing lightweight CSV/video export archives labels but
cannot reuse that binding after reconstructing a different NPZ. Retain the
original trajectory files for portable Stage training; hashes are never silently
rewritten. Usage: [Chinese guide §4.4.2](../README_CN.md#442-annotate-与-reward-materialize-的边界).

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

The trainer is currently single-process and single-GPU, but one GPU can process
multiple same-task replay transitions in each forward pass. Restrict a job to one
physical GPU with `CUDA_VISIBLE_DEVICES=N` and leave both configured devices as
`cuda:0`; the visible device is remapped to process-local index zero. Eight A100s
are best used for eight independent seeded/hyperparameter runs after preparing
and annotating once. A single run does not use DDP/FSDP yet and cannot be made
eight-GPU merely by exposing all devices.

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
threshold confirms success but does not terminate replay. Unconfirmed pulses
are treated as a failed trajectory. All later recorded actions remain eligible
for Q/V and actor training; bootstrap stops only at the final recorded
observation. Source NPZ files remain unchanged. The manifest retains raw and
debounced success diagnostics and the confirmation action as `terminal_step`,
while new manifests set `action_count == recorded_action_count` and declare
`replay_policy: full_recording_v1`. The importer groups
train/validation splits by root trajectory, validates the N+1 state/image
invariant, and constructs masked 8×7 tensors for variable-duration chunks of
at most eight actions. Each recorded chunk is one IQL macro-action decision:
the actual `chunk_length` selects `s[t+L]` and controls the action mask, while
the sparse reward and Bellman discount are each applied once per chunk by
default (`reward.accumulate_primitive_steps: false`). Setting the boolean to
`true` switches both the reward reduction and Bellman bootstrap to the matching
variable-duration Semi-MDP form: primitive rewards are discounted and summed
inside the chunk, PBRS uses `gamma ** L`, and bootstrap uses `gamma ** L`.

RynnValue receives only upright `agentview` frames and the BDDL task prompt. At
each action-chunk boundary, the adapter follows the pinned official inference
program: it uniformly resamples the visual prefix ending at that boundary and
reads the last value slot. `annotation_batch_size: 1` is the 16 GB default;
every boundary is evaluated from its complete prefix, so long trajectories are
not merged by averaging overlapping windows. Annotation schema v6 records only the
official decoded absolute distance, absolute logits/entropy, decoded relative
distance, relative logits, and exact generated Analysis text/token IDs. Parsed
Description / Match / Success values are display-only. Environment `done` is
the sole success authority, independently of the recorded replay endpoint.

Reward materialization is a separate deterministic CPU stage. It reads those
immutable official outputs and writes a second-level cache containing
`pbrs_shaping_reward` (the raw Shape Reward `γΦ(s')-Φ(s)`) and
`original_final_reward` (`r_sparse+κ·r_shape`) and fused `final_reward`. The
legacy replay field `pbrs_chunk_reward` aliases the fused Final Reward. Here
`r_sparse` is `-1` for an incomplete macro action and `0` when that chunk
completes the task, and `Φ=-absolute temporal distance`. Set
`reward.alpha: 0` and `reward.shaping_weight: 0` in additive mode for a sparse-only ablation: it derives Final
Reward directly from environment terminal flags without reading model outputs.
Existing RynnValue diagnostics remain untouched. The derived-reward cache key
includes fusion parameters, gamma, cumulative mode, annotation snapshots and
an implementation fingerprint; the raw model cache does not include reward
parameters. Changing the reward recipe recomputes only inexpensive arrays.
Hash-valid schema-v4/v5
sidecars reuse their complete official model heads during migration, regardless
of the reward reduction stored beside them, so RynnValue is not run again.
Post-success chunks retain the same reward formulas and participate in replay.
Existing prepared schema-v4 files with complete `evaluation_chunks` and reward
arrays are reused in memory without modifying their snapshots or rerunning
the reward model; missing full-tail data is rejected. For a training split with
post-success actions, start a new run rather than resume a checkpoint from the
old truncated replay policy.

The training default uses four uniformly sampled prefix frames, matching the
offline reward-relabeling protocol in paper Appendix B.3. Upstream's standalone
trend-video demo defaults to 64 frames; that demo default is not the paper's
IQL relabeling setting.

IQL update semantics follow the pinned `pi-rl` implementation: V is first fit
by expectile regression against the minimum frozen target Q, online Q is then
fit with the newly updated next-state V, target Q receives a Polyak update, and
policy weights use the updated online `min(Q1,Q2)-V`. The current default restores
the earlier Q/V AdamW setup with PyTorch's `betas=(0.9,0.999)`, `eps=1e-8`, and
weight decay `0.01`; YAML can still select the official Adam ablation. The VLA component optimizer uses AdamW with
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
  deterministic second-level reward cache for the active RynnValue inclusion
  switch, `gamma`, `kappa`, and macro/primitive-step reduction. Exact repeats
  are reused; a mismatch is rebuilt from annotations without loading RynnValue.
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
[Chinese guide](../README_CN.md). Their JSON/YAML files remain the
recoverable source of truth while SQLite is only a rebuildable query index.
