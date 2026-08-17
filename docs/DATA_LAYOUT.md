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
