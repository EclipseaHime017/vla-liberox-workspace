"""Dataset-scoped evaluation orchestration and immutable training reward pins."""

from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json, atomic_write_yaml


REWARD_PARAMETERS = {
    "reward_source": "source", "reward_stage_exponent": "stage_exponent",
    "reward_gamma": "gamma", "reward_shaping_weight": "shaping_weight",
    "reward_accumulate_primitive_steps": "accumulate_primitive_steps",
}
EDITABLE_REWARD_PARAMETERS = frozenset({"reward_gamma", "reward_accumulate_primitive_steps"})


class DatasetRewardVersions:
    """Mixin kept separate from simulation and model-training orchestration."""

    def _seed_dataset_outputs(self, dataset: dict, source: str, raw: dict, work: Path) -> None:
        """Reuse compatible output snapshots across packages, never their rewards."""
        _, frozen = self.datasets._load(dataset["id"])
        wanted = {member["run_id"]: member["artifacts"] for member in frozen["members"]}
        output = work / "annotations" / "annotation_manifest.json"
        seeded = json.loads(output.read_text()) if output.is_file() else None
        inference_keys = ("model", "revision", "dtype", "max_frames", "robot_description", "camera_description")
        if source == "rynnvalue" and seeded and any(
            seeded.get("annotation_config", {}).get(key) != raw["reward"].get(key) for key in inference_keys
        ):
            seeded = None
        entries = {entry["run_id"]: entry for entry in (seeded or {}).get("episodes", [])}
        for version_path in sorted(self.datasets.root.glob("*/annotations/*/version.json"), reverse=True):
            try:
                version = json.loads(version_path.read_text())
                if not version.get("complete") or version.get("evaluator") != source:
                    continue
                owner = json.loads((version_path.parents[2] / "dataset.json").read_text())
                compatible = {member["run_id"] for member in owner["members"]
                    if member["run_id"] in wanted and all(
                        member["artifacts"][name]["sha256"] == wanted[member["run_id"]][name]["sha256"]
                        for name in ("trajectory", "observations", "manifest"))}
                if not compatible:
                    continue
                if source == "rynnvalue":
                    prior = json.loads(Path(version["annotation_manifest_path"]).read_text())
                    if any(prior["annotation_config"].get(key) != raw["reward"].get(key) for key in inference_keys):
                        continue
                    if seeded is None:
                        seeded = copy.deepcopy(prior)
                    for entry in prior["episodes"]:
                        if entry["run_id"] in compatible:
                            entries.setdefault(entry["run_id"], entry)
                else:
                    values_dir = work / "robometer" / "values"
                    values_dir.mkdir(parents=True, exist_ok=True)
                    prior = json.loads(Path(version["robometer_manifest_path"]).read_text())
                    inference = {
                        "model": {key: raw["model"][key] for key in ("checkpoint", "revision", "robometer_commit", "dtype")},
                        "evaluation": {key: raw["evaluation"][key] for key in ("control_hz", "fps", "prefix_frames")},
                    }
                    if prior.get("inference_config") != inference:
                        continue
                    for entry in prior["episodes"]:
                        run_id = entry["run_id"]
                        target = values_dir / f"{run_id}.npz"
                        if run_id not in compatible or target.exists():
                            continue
                        original = Path(entry["annotation_path"])
                        if not original.is_file() or original.is_symlink():
                            continue
                        shutil.copyfile(original, target)
                        atomic_write_json(target.with_suffix(".json"), {**entry, "annotation_path": str(target)})
            except (KeyError, ValueError, OSError, TypeError):
                # Corrupt/missing output snapshots are cache misses, not errors
                # in the immutable source dataset. The evaluator validates reuse.
                continue
        if source == "rynnvalue" and seeded is not None:
            seeded["episodes"] = list(entries.values())
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output, seeded)

    def reward_configuration(self, dataset_id: str) -> dict[str, Any]:
        self.datasets.get(dataset_id, quick_verify=False)
        reward = self._load_base_config()["reward"]
        common = {"gamma": reward["gamma"],
                  "accumulate_primitive_steps": reward["accumulate_primitive_steps"]}
        robo_path = self.ui_config.robometer_root / "configs" / "robometer_evaluation.yaml"
        robo = yaml.safe_load(robo_path.read_text()) if robo_path.is_file() else {}
        return {
            "sparse": dict(common),
            "stage": {**common, "stage_exponent": reward.get("stage_exponent", 2.0)},
            "rynnvalue": {**common, "shaping_weight": reward["shaping_weight"],
                          "max_frames": reward["max_frames"],
                          "batch_size": reward["annotation_batch_size"],
                          "checkpoint": reward["model"], "revision": reward["revision"]},
            "robometer": {"sampling_hz": robo.get("evaluation", {}).get("fps", 3.0),
                          "batch_size": robo.get("evaluation", {}).get("batch_size", 8),
                          "prefix_frames": 4, "checkpoint": robo.get("model", {}).get("checkpoint"),
                          "revision": robo.get("model", {}).get("revision")},
        }

    def start_reward_version(self, dataset_id: str, *, source: str = "rynnvalue", **options: Any) -> dict:
        if source not in {"sparse", "stage", "rynnvalue", "robometer"}:
            raise ValueError("Unsupported evaluation source")
        options = {key: value for key, value in options.items() if value is not None}
        defaults = self.reward_configuration(dataset_id)[source]
        allowed = set(defaults) - {"checkpoint", "revision", "prefix_frames"}
        unknown = set(options) - allowed - {"force_model", "overwrite_global"}
        if unknown:
            raise ValueError(f"Unsupported {source} parameters: {sorted(unknown)}")
        parameters = {**defaults, **options, "source": source}
        parameters.setdefault("force_model", False)
        parameters.setdefault("overwrite_global", False)
        for key in ("accumulate_primitive_steps", "force_model", "overwrite_global"):
            if key in parameters and type(parameters[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("gamma", "stage_exponent", "shaping_weight", "sampling_hz"):
            if key not in parameters:
                continue
            value = parameters[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if ((key == "gamma" and not 0 <= value <= 1)
                    or (key == "stage_exponent" and value < 1)
                    or (key == "shaping_weight" and value < 0)
                    or (key == "sampling_hz" and not 0 < value <= 20)):
                raise ValueError(f"Invalid {key}")
        for key in ("batch_size", "max_frames"):
            if key in parameters and (type(parameters[key]) is not int or parameters[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        if "max_frames" in parameters and not 2 <= parameters["max_frames"] <= 64:
            raise ValueError("max_frames must be in [2, 64]")
        if source in {"sparse", "stage"} and parameters["force_model"]:
            raise ValueError("This reward source does not run a model")
        with self.lock:
            dataset = self.datasets.require_ready_for_annotation(dataset_id)
            if any(item["status"] == "RUNNING" for item in dataset.get("evaluation_versions", [])):
                raise ConflictError("Dataset evaluation is running", code="ANNOTATION_RUNNING")
            snapshot = None
            if source == "stage":
                if self.stage_annotations is None:
                    raise ValueError("Stage annotation service is unavailable")
                _, frozen = self.datasets._load(dataset_id)
                snapshot = self.stage_annotations.validate_members(
                    frozen["members"], dataset["success_consecutive_steps"],
                )
            gpu = source in {"rynnvalue", "robometer"}
            if gpu:
                self._prepare_launch()
            try:
                return self._launch_reward_version(dataset, parameters, snapshot)
            finally:
                self.launch_reserved = False

    def _launch_reward_version(self, dataset: dict, parameters: dict, snapshot: dict | None) -> dict:
        source, dataset_id = parameters["source"], dataset["id"]
        version_id = f"ann_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / version_id
        job_dir.mkdir(parents=True, exist_ok=False)
        root = self.datasets.root / dataset_id / "annotations" / version_id
        work = root / "work"
        work.mkdir(parents=True, exist_ok=False)
        raw = self._effective_config(dataset)
        raw["data"].pop("stage_annotations_manifest", None)
        raw["reward"].update(source=source if source != "robometer" else "sparse",
                             rynnvalue=source == "rynnvalue")
        for key in ("gamma", "stage_exponent", "shaping_weight", "accumulate_primitive_steps", "max_frames"):
            if key in parameters:
                raw["reward"][key] = parameters[key]
        if source == "rynnvalue":
            raw["reward"]["annotation_batch_size"] = parameters["batch_size"]
        raw["paths"]["work_dir"] = str(work.resolve())
        raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None)
        config_path = job_dir / "effective_config.yaml"
        if snapshot is not None:
            snapshot_path = root / "stage_annotations.json"
            atomic_write_json(snapshot_path, {"schema_version": 1, "annotations": snapshot})
            raw["data"]["stage_annotations_manifest"] = str(snapshot_path.resolve())
        scripts = self.ui_config.offline_rl_root / "scripts"
        stages = []

        def stage(name: str, label: str, script: Path, config: Path, environment: str, *extra: str) -> None:
            stages.append({"id": name, "label": label, "environment": environment,
                           "argv": ["python", str(script), "--config", str(config), *extra],
                           "cwd": str(script.parent.parent)})

        previous = next((item for item in reversed(dataset.get("evaluation_versions", []))
                         if item["evaluator"] == source and item["status"] == "READY"), None)
        previous_id = previous["id"] if previous else dataset.get("annotation_id") if source == "rynnvalue" else None
        previous_work = self.datasets.root / dataset_id / "annotations" / str(previous_id) / "work"
        if source == "robometer":
            base_path = self.ui_config.robometer_root / "configs" / "robometer_evaluation.yaml"
            robo = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            checkout = Path(robo["paths"]["robometer_root"])
            if not checkout.is_absolute():
                checkout = (base_path.parent / checkout).resolve()
            if not (checkout / "robometer" / "__init__.py").is_file():
                raise FileNotFoundError(f"Official Robometer checkout is unavailable: {checkout}")
            robo["paths"].update(robometer_root=str(checkout),
                selection_manifest=raw["data"]["selection_manifest"], output_dir=str((work / "robometer").resolve()))
            robo["evaluation"].update(fps=parameters["sampling_hz"], batch_size=parameters["batch_size"])
            if not parameters["force_model"]:
                self._seed_dataset_outputs(dataset, source, robo, work)
            if not parameters["force_model"] and getattr(self, "robometer_evaluations", None) is not None:
                self.robometer_evaluations.seed_version_cache(
                    [m["run_id"] for m in dataset["members"]], work / "robometer" / "values",
                    {"model": {key: robo["model"][key] for key in (
                        "checkpoint", "revision", "robometer_commit", "dtype")},
                     "evaluation": {key: robo["evaluation"][key] for key in (
                         "control_hz", "fps", "prefix_frames")}},
                )
            atomic_write_yaml(config_path, robo)
            stage("robometer", "Robometer 轨迹评价", self.ui_config.robometer_root / "scripts" / "evaluate_trajectories.py",
                  config_path, self.ui_config.robometer_environment, *( ["--overwrite"] if parameters["force_model"] else []))
        else:
            if source == "rynnvalue" and not parameters["force_model"]:
                if self.trajectory_evaluations is not None:
                    self.trajectory_evaluations.seed_cache([m["run_id"] for m in dataset["members"]], self.cache_root)
                prior = previous_work / "annotations" / "annotation_manifest.json"
                if prior.is_file():
                    target = work / "annotations" / "annotation_manifest.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(prior, target)
                else:
                    legacy = previous_work / "rewards" / "reward_manifest.json"
                    if legacy.is_file():
                        target = work / "rewards" / "reward_manifest.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(legacy, target)
                self._seed_dataset_outputs(dataset, source, raw, work)
            atomic_write_yaml(config_path, raw)
            stage("prepare", "准备冻结数据集", scripts / "prepare_dataset.py", config_path, self.ui_config.train_environment)
            if source == "rynnvalue":
                stage("annotate", "RynnValue 轨迹评价", scripts / "annotate_rewards.py", config_path,
                      self.ui_config.reward_environment, *(["--overwrite"] if parameters["force_model"] else []))
            stage("rewards", "计算数据集奖励", scripts / "materialize_rewards.py", config_path, self.ui_config.train_environment)
        now = datetime.now(timezone.utc).isoformat()
        version = {"schema_version": 1, "id": version_id, "dataset_id": dataset_id,
                   "dataset_sha256": dataset["dataset_sha256"], "evaluator": source,
                   "parameters": parameters, "status": "RUNNING", "complete": False,
                   "created_at": now, "completed_at": None, "error": None,
                   "work_dir": str(work.resolve()), "config_path": str(config_path.resolve()),
                   "run_ids": [m["run_id"] for m in dataset["members"]]}
        version_path = root / "version.json"
        atomic_write_json(version_path, version)
        finalizer = Path(__file__).resolve().parents[1] / "workers" / "finalize_reward_version.py"
        stages.append({"id": "seal", "label": "冻结评价与奖励版本", "environment": self.ui_config.train_environment,
                       "argv": ["python", str(finalizer), "--version", str(version_path)]})
        self.datasets.update_version(dataset_id, version)
        try:
            return self._new_job(kind="annotation", dataset_id=dataset_id, stages=stages,
                config_path=config_path, output_path=root,
                parameters={**parameters, "reward_version_id": version_id, "task_id": dataset["task_id"],
                            "member_count": dataset["member_count"], "requires_gpu": source in {"rynnvalue", "robometer"}})
        except Exception as exc:
            self.datasets.update_version(dataset_id, {**version, "status": "ERROR", "error": str(exc)})
            raise

    def pinned_reward(self, dataset: dict, parameters: dict) -> tuple[dict, dict]:
        selected_source = parameters.get("reward_source")
        if selected_source is None and "reward_rynnvalue" in parameters:
            selected_source = "rynnvalue" if parameters["reward_rynnvalue"] else "sparse"
        version_id = parameters.get("reward_version_id") or (
            dataset.get("evaluation_version_ids", {}).get(selected_source)
            if selected_source else dataset.get("reward_version_id"))
        if version_id:
            version = self.datasets.validate_version(dataset["id"], version_id)
        else:
            from .global_reward_binding import bind_global_rewards
            selected_source = selected_source or self._reward_source(self._load_base_config(), {})
            version = bind_global_rewards(self, dataset, selected_source)
            version_id = version["id"]
        if selected_source is not None and selected_source != version["evaluator"]:
            raise ConflictError("reward_source conflicts with selected evaluation", code="REWARD_CONFIG_LOCKED")
        if version.get("status") != "READY" or not version.get("complete") or version["evaluator"] == "robometer":
            raise ConflictError("Training reward version is not ready", code="REWARD_VERSION_NOT_READY")
        settings = yaml.safe_load(Path(version["config_path"]).read_text(encoding="utf-8"))["reward"]
        expected = {key: settings[name] for key, name in REWARD_PARAMETERS.items() if name in settings}
        expected["reward_rynnvalue"] = version["evaluator"] == "rynnvalue"
        for key, value in expected.items():
            if key not in EDITABLE_REWARD_PARAMETERS and key in parameters and parameters[key] != value:
                raise ConflictError(f"{key} is locked to dataset reward version {version_id}", code="REWARD_CONFIG_LOCKED")
        # Omitted knobs retain the dataset defaults. Explicit values only affect
        # this training run's derived reward and Bellman discount.
        return version, {**expected, **parameters, "reward_version_id": version_id}

    def _schedule_result_binding(self, job: dict) -> None:
        """Publish globals once on completion without doing disk work in polling."""
        if job.get("status") != "COMPLETED" or job.get("trajectory_binding") or job.get("trajectory_binding_error"):
            return
        with self.lock:
            workers = getattr(self, "_binding_workers", None)
            if workers is None:
                workers = self._binding_workers = {}
            if job["id"] in workers and not workers[job["id"]].done():
                job["trajectory_binding_status"] = "RUNNING"
                return
            path, current = self._load_job(job["id"])
            if current.get("trajectory_binding") or current.get("trajectory_binding_error"):
                job.update(current)
                return
            current["trajectory_binding_status"] = "RUNNING"
            job["trajectory_binding_status"] = "RUNNING"
            atomic_write_json(path, current)

            def bind() -> None:
                try:
                    result = self._bind_completed_result(current)
                    changes = {"trajectory_binding": result, "trajectory_binding_status": "READY"}
                except Exception as exc:
                    changes = {"trajectory_binding_error": f"{type(exc).__name__}: {exc}",
                               "trajectory_binding_status": "ERROR"}
                with self.lock:
                    final_path, final = self._load_job(current["id"])
                    final.update(changes)
                    atomic_write_json(final_path, final)
                    self.repository.upsert(final, final_path)

            executor = getattr(self, "_binding_executor", None)
            if executor is None:
                executor = self._binding_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="trajectory-result-bind")
            # A service restart may discover many completed historical jobs.
            # Serialize disk work instead of launching one hashing thread each.
            for key in [key for key, future in workers.items() if future.done()]:
                del workers[key]
            workers[job["id"]] = executor.submit(bind)

    def schedule_first_reward_snapshot(self, run: dict) -> None:
        """Queue legacy global initialization; detail requests never wait for IO."""
        if not run.get("trajectory"):
            return
        from .trajectory_reward_snapshot import ensure_first_reward_snapshot

        with self.lock:
            workers = getattr(self, "_binding_workers", None)
            if workers is None:
                workers = self._binding_workers = {}
            key = f"trajectory:{run['id']}"
            if key in workers and not workers[key].done():
                return
            executor = getattr(self, "_binding_executor", None)
            if executor is None:
                executor = self._binding_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="trajectory-result-bind")
            workers[key] = executor.submit(ensure_first_reward_snapshot, dict(run), self.datasets)

    def _first_dataset_result(self, run_id: str, source: str) -> dict | None:
        candidates = []
        for dataset in self.datasets.list():
            if any(member["run_id"] == run_id for member in dataset.get("members", [])):
                for version in dataset.get("evaluation_versions", []):
                    if version.get("status") == "READY" and version.get("evaluator") == source:
                        candidates.append((version.get("completed_at") or version.get("created_at") or "",
                                           dataset["id"], version["id"]))
        for _, dataset_id, version_id in sorted(candidates):
            try:
                version = self.datasets.get_version(dataset_id, version_id)
            except (OSError, KeyError, ValueError):
                continue
            if source == "robometer":
                path = version.get("robometer_manifest_path")
            else:
                path = version.get("reward_manifest_path")
            if path and Path(path).is_file():
                return version
        return None

    def _bind_completed_result(self, job: dict) -> dict:
        from .trajectory_reward_snapshot import bind_reward_snapshot, ensure_first_reward_snapshot

        parameters = job.get("parameters", {})
        result = {}
        if job["kind"] == "annotation":
            version = self.datasets.get_version(job["dataset_id"], job["id"])
            source = version["evaluator"]
            overwrite = bool(parameters.get("overwrite_global", False))
            if overwrite and source != "robometer":
                # Validate all replacements before publishing the first one.
                result["reward"] = bind_reward_snapshot(
                    Path(version["prepared_manifest_path"]), Path(version["reward_manifest_path"]),
                    overwrite=True, origin="manual", evaluation_id=job["id"], run_ids=version["run_ids"],
                    evaluated_at=version.get("completed_at"))
            for run_id in version["run_ids"]:
                run = self.datasets.run_service.get_run(run_id)
                if source != "robometer" and not overwrite:
                    ensure_first_reward_snapshot(run, self.datasets)
                service = (self.trajectory_evaluations if source == "rynnvalue" else
                           self.robometer_evaluations if source == "robometer" else None)
                if service is None:
                    continue
                if not overwrite and service.exists(run):
                    continue
                selected = version if overwrite else self._first_dataset_result(run_id, source) or version
                if source == "rynnvalue":
                    value = service.bind(Path(selected["prepared_manifest_path"]), Path(selected["reward_manifest_path"]),
                                         overwrite=overwrite, run_ids=[run_id])
                else:
                    value = service.bind(Path(selected["robometer_manifest_path"]), overwrite=overwrite, run_ids=[run_id])
                result.setdefault(source, []).append(value)
        else:
            work = Path(job["output_path"]) / "work"
            overwrite = bool(parameters.get("overwrite", False))
            selected = parameters.get("selected_by_evaluator") or {}
            if selected.get("rynnvalue"):
                prepared = work / "rynnvalue" / "dataset_manifest.json"
                rewards = work / "rynnvalue" / "rewards" / "reward_manifest.json"
                # Bind the generic snapshot first so this operation does not
                # mistake its own just-written Rynn sidecar for older data.
                result["reward"] = bind_reward_snapshot(prepared, rewards, overwrite=overwrite,
                    origin="manual", evaluation_id=job["id"], evaluated_at=job.get("completed_at"))
                result["rynnvalue"] = self.trajectory_evaluations.bind(prepared, rewards, overwrite=overwrite)
            if selected.get("robometer"):
                result["robometer"] = self.robometer_evaluations.bind(
                    work / "robometer" / "robometer_manifest.json", overwrite=overwrite)
        return result or {"skipped": True}
