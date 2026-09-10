# Data layout

`dataset_root` and `project_id` in `configs/ui_config.yaml` produce:

```text
dataset-root/
├── catalog.sqlite3
└── projects/
    └── libero_x_vla/
        ├── datasets/
        │   └── <dataset_id>/
        │       ├── dataset.json
        │       └── annotations/<annotation_id>/work/
        ├── annotation-cache/
        ├── training/<training_id>/
        ├── evaluations/
        │   └── <task_name>/
        │       └── YYYY-MM-DD/
        │           └── YYYY-MM-DD_HHMMSS__<evaluation_id>/
        │               └── evaluation.json
        ├── jobs/<job_id>/
        └── runs/
            └── <task_name>/
                └── YYYY-MM-DD/
                    └── YYYY-MM-DD_HHMMSS__<run_id>/
                        ├── run.json
                        ├── config.yaml
                        ├── summary.json
                        └── episodes/
                            └── episode_000/
                                ├── trajectory.npz
                                ├── trajectory.csv
                                ├── *_observations.npz
                                ├── agentview.mp4
                                ├── vla_views.mp4
                                ├── rynnvalue_evaluation.json
                                ├── rynnvalue_evaluation.npz
                                ├── *.png
                                ├── spacemouse_samples.csv  # SpaceMouse runs only
                                └── factr_samples.csv       # FACTR runs only
```

- `run.json` is the small mutable lifecycle manifest and deletion safety marker.
- `config.yaml` freezes user-selectable and runtime-critical inputs.
- `summary.json` contains the compact outcome and timing values normally inspected by a person.
- `episodes/episode_000` contains high-volume replay artifacts. The current UI creates one episode per run; the directory boundary permits later multi-episode runs.
- `catalog.sqlite3` is a rebuildable index for filtering and aggregate success rates. Run directories are the durable source of truth.
- `legacy_scan_roots` are indexed read-only. Existing `runs/` directories are never migrated, renamed, or deleted by the new catalog.

All JSON/YAML/CSV publications use temporary files followed by atomic replacement. New run directories are unique and never overwrite earlier experiments.

## Manual controller recordings

SpaceMouse and FACTR share the same run layout, `action_source=human` category,
dataset export and training input. `manual_source` retains the device name for
diagnostics only. FACTR follows seven joint targets in physics but stores a 7-D
end-effector label per 20 Hz step: world translation `p_next-p_current`, world
relative rotation `Log(R_next R_current^T)`, inverse-scaled using the environment's
original OSC input/output ranges, plus gripper `-1` (open) or `+1` (close).
This is achieved-motion relabeling, not a claim of exact OSC inverse dynamics.

`raw_action` retains the unbounded normalized label; `env_action` clips six axes
to `[-1,1]`. `controller_diagnostics` in trajectory metadata and controller summary
report the mapping, clipping counts/fraction, per-axis counts and maxima. There
is no separate FACTR training category or joint demonstration CSV. The existing
`sim_state` remains for physical-state restoration, video/observation rendering,
and parent-prefix preservation. Each N actions still has N+1 states/images;
success does not truncate the suffix. Faults retain completed transitions.

Existing `factr_samples.csv` files remain readable/exportable legacy diagnostics.
Independent device tests still do not record trajectories or videos.

## Immutable training datasets and background jobs

`datasets/<dataset_id>/dataset.json` is an immutable, single-task membership
manifest. It stores the explicit run IDs, provenance (`inference`, `manual`, or
`policy_requery`), root/parent IDs, resume boundary, effective action range,
train/validation grouping, and the path, size, and SHA-256 of each source
`run.json`, `trajectory.npz`, and `trajectory_observations.npz`. It does not copy
those high-volume artifacts. Branch members represent one selected record but
prepare only imports `[resume_step, end_step)` as new replay data.

The mutable fields in the same JSON are limited to integrity and annotation
lifecycle. `annotation_id` selects the evaluation version used by training,
while `annotation_history` preserves earlier versions and their effective
configuration (including `max_frames` and the boolean
`accumulate_primitive_steps`). Creating a new evaluation version
regenerates that version's prepared manifest and reward manifest, then switches
`annotation_id` only after the whole job succeeds. A failed replacement leaves
the previous ready version active. Full verification recomputes every hash. Deleting a referenced run
requires explicit force confirmation and marks every referencing dataset
`BROKEN`; existing training summaries and overlays remain auditable but the
dataset can no longer be annotated or trained.

`annotation-cache/<content_hash>.npz` contains the shared, reward-agnostic
RynnValue computation cache. Annotation schema v6 records the official
RynnValue head results without averaging overlapping windows or replacing
their semantics: `absolute_temporal_distance_seconds [N,H]`,
`absolute_value_entropy_nats [N,H]`, `absolute_value_logits [N,H,B]`,
`relative_temporal_distance_seconds [N]`, and `relative_value_logits [N,B]`,
aligned by `boundary_steps [N]`. The adjacent JSON preserves the exact generated
Analysis text and token IDs; `parsed_for_display` only extracts Description,
Match, and Success for UI display and is never treated as the original output.
For every boundary, the absolute result is the last `<value>` slot of the
official uniformly resampled prefix; the relative result is the
`<relative_value>` slot between that same prefix's final two presented images,
not a finite difference computed afterward from two absolute predictions. The
canonical annotation NPZ contains no sparse, Shape, or Final Reward arrays.
Its key depends on trajectory/observation content, prompt, required boundaries,
RynnValue model/revision, dtype, and `max_frames`; it deliberately excludes
`gamma`, `kappa`, and reward reduction mode.

Each prepared dataset has
`annotations/<annotation_id>/work/annotations/annotation_manifest.json`, which
binds its members to those immutable model outputs. A separate deterministic
stage writes `work/rewards/<reward_hash>.npz` and `reward_manifest.json`.
`pbrs_shaping_reward` stores the raw, unweighted RynnValue Shape Reward
`gamma * Phi(s_next) - Phi(s)`, `dense_reward` stores
`kappa * pbrs_shaping_reward`, and `pbrs_chunk_reward` stores the Final Reward
`r_sparse + dense_reward`. Every action chunk is one macro-action decision by
default:
`r_sparse` is `-1` for an incomplete chunk and `0` for a completing chunk, and
the IQL Bellman target uses one `gamma`. Variable chunks keep their actual `L`
only for the action mask and selection of `s[t+L]`. This second cache is keyed
by the prepared dataset and annotation hashes plus the `rynnvalue` inclusion
switch, `gamma`, `kappa`, and the macro/primitive-step switch. Exact repeated
training reuses it; a mismatch is recomputed with NumPy and never invokes
RynnValue. With `rynnvalue=false`, the diagnostic Shape Reward remains present,
but `dense_reward` is zero and Final Reward equals the sparse reward.

After a successful UI evaluation job, a combined evaluation snapshot may be
atomically copied beside the source episode as `rynnvalue_evaluation.npz`; its
adjacent JSON binds the trajectory hash, observation hash, RynnValue revision,
active reward configuration, evaluation time, and value-file hash. These two
small sidecars are the durable trajectory-level evaluation used by the Dataset
detail page. Existing hash-valid schema-v4/v5 sidecars remain valuable: schema
v6 migration extracts and preserves their official heads, ignores stale derived
reward arrays, and does not run another model forward.

With `accumulate_primitive_steps=false`, every action chunk is one macro-action
decision: `r_sparse` is `-1` for an incomplete chunk and `0` for a completing
chunk, PBRS and the Bellman target use one `gamma`, and actual `L` only controls
the action mask and `s[t+L]`. With the boolean set to `true`, primitive rewards
inside the chunk are discounted and summed, while PBRS and Bellman bootstrap
both use `gamma^L`.

The Dataset detail API retains the absolute/relative remaining-time and entropy
series. The UI also displays the observation potential directly derived as
`Phi(s) = -absolute_remaining_time(s)` without changing the persisted official
RynnValue outputs. The API exposes the reward terms as `pbrs_reward.shape_reward` and
`pbrs_reward.final_reward` together
with `chunk_start_steps`, `chunk_end_steps`, and `chunk_lengths`. The UI renders
one sample at each chunk completion and connects adjacent chunk samples; it does
not duplicate a chunk reward over every control frame.
Hash-valid v4/v5 sidecars with the same official inference contract are migrated
by reusing their RynnValue heads and recomputing only deterministic reward
arrays. Older v2
sidecars are intentionally treated as unevaluated because they may have been
produced with the standalone demo's 32/64-frame protocol.

The Dataset page has separate **trajectory evaluation** and **dataset package**
flows. Batch trajectory evaluation skips valid sidecars unless overwrite is
requested. Explicit one/multi-run evaluation never replaces a valid result from
the same evaluator; it only fills missing evaluator results. Dataset
annotation is a separate versioned operation: it prepares the frozen membership,
creates or reuses content-addressed RynnValue computations, and records the new
dataset-local reward manifest without changing trajectory sidecars. A prepared branch keeps boundaries for its complete physical
trajectory, including the natural rollout before takeover and any fixed-duration
post-success tail, so RynnValue plots span the source video from time zero.
`ReplayDataset` still stops at the confirmed terminal and removes duplicate
`(root, start, end, action_source)` copied-prefix transitions only when training
samples are assembled.

Trajectory use labels are stored in `catalog.sqlite3`, not in immutable run
artifacts. A run marked as **test** remains browsable and can still be explicitly
evaluated, but it is excluded from new training-dataset previews/packages,
offline-RL task exports, and task-wide batch trajectory evaluation. Explicitly
selecting test runs for an inspection evaluation remains allowed. Changing the
label does not retroactively mutate an already frozen dataset.

`jobs/<job_id>/job.json` is the durable state/heartbeat/PID manifest;
`job.log` is append-only and `effective_config.yaml` is the validated config
actually passed to the Conda subprocesses. Training artifacts are written under
`training/<training_id>/`, including metrics JSONL, TensorBoard events,
checkpoints, summary, and cancellation checkpoints. `catalog.sqlite3` indexes
datasets, members, annotations, training runs, evaluation runs, and jobs, but
these files remain the recoverable source of truth.

## Lightweight batch evaluations

`evaluations/<task_name>/<date>/<timestamp>__<evaluation_id>/evaluation.json`
is the only result artifact created by the Test page. It contains the selected
task and policy snapshot, base checkpoint/overlay compatibility metadata, the
validated effective configuration, the complete frozen schedule and its hash,
per-episode numeric results, aggregate success statistics, timings, lifecycle
state, and any error. The result directory must not contain MP4, image, NPZ,
CSV, observation, action, trajectory, or plot files.

The independent `jobs/<evaluation_id>/` directory is scheduler diagnostics,
not evaluation output. It retains `job.json`, append-only `job.log`, and the
generated `effective_config.yaml` so a detached process can be stopped or
reconciled after a backend restart. Deleting a terminal evaluation removes both
its result directory and corresponding job diagnostics after exact-ID
confirmation; it never removes a policy, checkpoint, dataset, or simulation
run.

Each evaluation selects exactly one BDDL task and one policy. The
`init_state_index` pool contains only the finite benchmark states of that task;
it never ranges over other BDDL scenes in the same level. The seed pool is
`base_seed .. base_seed + seed_count - 1`. Before execution, the service uses
`schedule_seed` to create a deterministic balanced rotation over init states
and seeds, with the allocation count for every state, seed, and available
combination differing by at most one. The full schedule is persisted before
the first episode and is immutable during the job.

A normal episode always records `max_steps` executed control steps. Success is
latched only after five consecutive LIBERO-X `done=true` steps; a later false
value does not clear the confirmed result, and confirmation does not terminate
the episode early. Errors count in the attempted-episode success-rate
denominator. A canceled or stopped evaluation reports aggregate values only for
attempted episodes and separately records completion coverage, so partial and
complete tests cannot be confused.

The aggregate section includes total success rate with Wilson 95% confidence
interval, success/failure/error counts, coverage, and group statistics by init
state, seed, and init-state/seed combination. Per-episode values include first
confirmed-success step, maximum done streak, final done, inference count and
latency, measured control frequency, deadline misses, and elapsed time. The
file also records policy-load time, total wall time, and total simulated time.
All updates use temporary files followed by atomic replacement.

## Offline RL export

The Dataset page exports one selected task as a ZIP without changing the source
run directories. The ZIP preserves the episode hierarchy:

```text
liberox_<task_id>.zip
├── runs.csv
├── export.json
├── DATA_FORMAT.md
└── runs/
    └── <run_id>/
        ├── run.json
        ├── config.yaml
        ├── summary.json
        └── episodes/
            └── episode_000/
                ├── trajectory.csv
                ├── trajectory_inference.csv  # policy runs only, when available
                ├── factr_samples.csv         # FACTR runs only, when available
                ├── agentview.mp4
                └── vla_views.mp4
```

Large replay/checkpoint artifacts are intentionally excluded: `trajectory.npz`,
`*_observations.npz`, plots, `source_trajectory.npz`, and raw SpaceMouse
diagnostic samples. FACTR's small `factr_samples.csv` is included when present
for input-to-OSC auditing, but is not used as a replacement training action.
MP4 files are copied without ZIP recompression.

### Transition schema

`trajectory.csv` has `N + 1` rows for `N` executed actions. Row `i < N`
represents the transition:

```text
state[i] -- action[i] --> state[i + 1]
```

The last row is the terminal state and therefore has empty action, reward, and
`action_source` cells. Column groups are:

| Columns | Meaning | Unit |
| --- | --- | --- |
| `step`, `time_seconds` | state index and simulated time | step, s |
| `eef_x/y/z` | end-effector position | m |
| `axis_angle_x/y/z` | end-effector axis-angle orientation | rad |
| `quat_x/y/z/w` | end-effector quaternion | unitless |
| `gripper_left/right` | simulated finger joint position | MuJoCo qpos |
| `vla_action_*` | raw command before environment conversion | normalized command |
| `action_*` | OSC_POSE command sent to LIBERO-X | normalized command |
| `reward`, `done` | environment transition result | scalar, boolean |
| `action_source` | transition provenance | enum below |

The `vla_action_*` name is retained for schema compatibility. On
`action_source=human` rows it contains the normalized Cartesian command from
SpaceMouse or the FACTR FK/OSC adapter, not a VLA prediction or raw leader joint
angles. Use `action_source` to distinguish policy/human transitions and the run's
`manual_source` to identify the controller, never the column prefix.

`trajectory_inference.csv` records complete VLA action chunks, including
predicted actions that were not executed because the policy was queried again.
Executed behavior must be read from `trajectory.csv`.

### Intervention segmentation

The exported `runs.csv` includes `kind`, `control_mode`, `success`,
`parent_session_id`, `root_session_id`, and `resume_step`. Branch
`trajectory.csv` files are already merged into a complete episode:

- `policy`: transitions copied from or executed by the original VLA rollout;
- `policy_requery`: VLA transitions generated after restoring `resume_step`;
- `human`: SpaceMouse or FACTR transitions generated after restoring `resume_step`.

For direct filtering, `runs.csv` also supplies `episode_category` with one of
`unassisted_success`, `unassisted_failure`, `manual_intervention`,
`policy_requery_branch`, or `error_or_incomplete`, plus the declared
`prefix_action_source` and `suffix_action_source`. Transition-level
`action_source` remains authoritative.

For a manual branch at step `K`, rows `[0, K)` are the original `policy` prefix
and rows `[K, N)` are the new `human` suffix. For a policy branch, the suffix is
`policy_requery`. A branch may itself succeed or fail; its outcome does not
change the segment boundary.

An unassisted failed sample is identified explicitly as:

```text
kind == "original"
control_mode == "policy"
success == false
```

Such a trajectory is a policy failure/negative episode, not an expert
demonstration. For corrective offline RL or behavior cloning, select the
`human` suffix as the intervention target and optionally retain the preceding
`policy` prefix as context. Do not infer takeover from filenames, action jumps,
reward, or success alone.

### Video alignment

Both exported videos have one encoded frame per executed action. Frame `i`
aligns with action row `i`; the final state row has no video frame.

- `agentview.mp4`: high-resolution external camera for inspection or auxiliary
  visual training.
- `vla_views.mp4`: exact synchronized VLA inputs. Each frame is a horizontal
  mosaic: the left half is `agentview`, the right half is
  `robot0_eye_in_hand`. Split at `frame_width // 2` to recover the two policy
  views.

The configured video FPS equals `control_hz`, so frame index is the preferred
alignment key; timestamps should only be used as a consistency check.
