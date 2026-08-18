# Data layout

`dataset_root` and `project_id` in `configs/ui_config.yaml` produce:

```text
dataset-root/
├── catalog.sqlite3
└── projects/
    └── libero_x_vla/
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
                                ├── *.png
                                └── spacemouse_samples.csv
```

- `run.json` is the small mutable lifecycle manifest and deletion safety marker.
- `config.yaml` freezes user-selectable and runtime-critical inputs.
- `summary.json` contains the compact outcome and timing values normally inspected by a person.
- `episodes/episode_000` contains high-volume replay artifacts. The current UI creates one episode per run; the directory boundary permits later multi-episode runs.
- `catalog.sqlite3` is a rebuildable index for filtering and aggregate success rates. Run directories are the durable source of truth.
- `legacy_scan_roots` are indexed read-only. Existing `runs/` directories are never migrated, renamed, or deleted by the new catalog.

All JSON/YAML/CSV publications use temporary files followed by atomic replacement. New run directories are unique and never overwrite earlier experiments.

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
                ├── agentview.mp4
                └── vla_views.mp4
```

Large replay/checkpoint artifacts are intentionally excluded: `trajectory.npz`,
`*_observations.npz`, plots, `source_trajectory.npz`, and raw SpaceMouse
diagnostic samples. MP4 files are copied without ZIP recompression.

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
`action_source=human` rows it contains the raw SpaceMouse command, not a VLA
prediction. Use `action_source`, never the column prefix, to identify the
controller.

`trajectory_inference.csv` records complete VLA action chunks, including
predicted actions that were not executed because the policy was queried again.
Executed behavior must be read from `trajectory.csv`.

### Intervention segmentation

The exported `runs.csv` includes `kind`, `control_mode`, `success`,
`parent_session_id`, `root_session_id`, and `resume_step`. Branch
`trajectory.csv` files are already merged into a complete episode:

- `policy`: transitions copied from or executed by the original VLA rollout;
- `policy_requery`: VLA transitions generated after restoring `resume_step`;
- `human`: SpaceMouse transitions generated after restoring `resume_step`.

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
