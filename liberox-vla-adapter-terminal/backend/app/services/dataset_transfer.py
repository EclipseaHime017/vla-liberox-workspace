"""Read-only PC export adapter; no UI server, simulation or GPU model is started."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from ..storage.paths import storage_path, storage_lease
from types import SimpleNamespace

from .dataset_reward_versions import DatasetRewardVersions
from .training_dataset_service import TrainingDatasetService


class _ReadOnlyDatasets(TrainingDatasetService):
    def __init__(self, project_root: Path, dataset_id: str):
        self.root = project_root / "datasets"
        self.run_service = self
        self.primary_dataset = dataset_id

    def get(self, dataset_id, *, quick_verify=False):
        _, payload = self._load(dataset_id)
        error = self._immutable_error(payload)
        if error:
            raise ValueError(error)
        return self._public(payload)

    def get_run(self, run_id):
        primary = self.root / self.primary_dataset / "dataset.json"
        for path in [primary, *sorted(self.root.glob("*/dataset.json"))]:
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                if path == primary:
                    raise
                continue
            member = next((item for item in payload["members"] if item["run_id"] == run_id), None)
            if member:
                artifacts = member["artifacts"]
                manifest = storage_path(artifacts["manifest"]["path"])
                return {**json.loads(manifest.read_text()), "id": run_id,
                        "trajectory": artifacts["trajectory"]["path"], "output_dir": str(manifest.parent)}
        raise KeyError(run_id)


class _ExportRewards(DatasetRewardVersions):
    read_only = True

    def __init__(self, datasets, base_config, offline_root, scratch):
        self.datasets, self.base_config_path = datasets, base_config
        self.ui_config = SimpleNamespace(offline_rl_root=offline_root)
        self.jobs_root, self.cache_root = scratch / "jobs", scratch / "cache"

    def _load_base_config(self, *args):
        from vla_rynn_iql.config import load_train_config
        return copy.deepcopy(load_train_config(self.base_config_path).raw)

    def _effective_config(self, dataset):
        raw = self._load_base_config()
        raw["data"].update(project_id=dataset["project_id"], task_ids=[dataset["task_id"]],
            selection_manifest=str(self.datasets.root / dataset["id"] / "dataset.json"),
            **{key: dataset[key] for key in ("validation_fraction", "split_seed", "success_consecutive_steps")})
        raw["data"]["include_post_success"] = dataset.get("include_post_success", True)
        raw["paths"].update(work_dir=str(self.jobs_root / "prepare"), annotation_cache=str(self.cache_root))
        return raw


def export_dataset(project_root: Path, dataset_id: str, destination: Path, base_config: Path,
                   offline_root: Path, *, required_rewards: tuple[str, ...] = ()) -> Path:
    with storage_lease(project_root.resolve().parents[1]):
        return _export_dataset(project_root, dataset_id, destination, base_config, offline_root,
                               required_rewards=required_rewards)


def _export_dataset(project_root: Path, dataset_id: str, destination: Path, base_config: Path,
                    offline_root: Path, *, required_rewards: tuple[str, ...] = ()) -> Path:
    from vla_rynn_iql.config import LoadedConfig
    from vla_rynn_iql.data import prepare_dataset
    from vla_rynn_iql.portable_dataset import export_bundle

    datasets = _ReadOnlyDatasets(project_root, dataset_id)
    selection_path, frozen = datasets._load(dataset_id)
    dataset = datasets.get(dataset_id)
    with tempfile.TemporaryDirectory(prefix="vla-transfer-") as directory:
        context = _ExportRewards(datasets, base_config, offline_root, Path(directory))
        versions, unavailable = {}, {}
        for source in ("sparse", "stage", "rynnvalue", "final"):
            try:
                versions[source], _ = context.pinned_reward(dataset, {"reward_source": source})
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                # A corrupt explicit dataset result is never replaced by another source.
                if source in required_rewards or source in dataset.get("evaluation_version_ids", {}):
                    raise ValueError(f"Cannot export {source}: {exc}") from exc
                unavailable[source] = str(exc)
        raw = context._effective_config(dataset)
        # Evaluated datasets already have a complete, verified Prepare snapshot.
        # Reusing it also preserves geometry if installation defaults changed.
        if versions:
            first = next(iter(versions.values()))
            prepared = Path(first["prepared_manifest_path"])
        else:
            prepared = prepare_dataset(LoadedConfig(base_config, raw)).manifest
        return export_bundle(destination, frozen, selection_path, prepared, versions,
                             notes={"unavailable_rewards": unavailable,
                                    "migration": "Source files are retained unchanged; old dated layouts remain readable"})
