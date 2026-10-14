"""Read copied run directories and their global evaluations, without a UI database.

Discovery reads only small JSON files. Full validation and private reward snapshots
belong to preparation, never to the terminal's repaint loop. No evaluator is run.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import LoadedConfig, needs_rynnvalue, needs_stage
from .data import load_manifest, prepare_dataset
from .io import atomic_json, sha256_file, stable_hash
from .methods import training_method
from .rewards import (
    _episode_reward_metadata, _episode_timeline_arrays, load_pinned_reward_index, materialize_reward_manifest,
    reward_derivation_config, OFFICIAL_OUTPUT_KEYS, validate_official_outputs,
)

REWARD_SOURCES = ("final", "rynnvalue", "stage", "sparse")


@contextmanager
def open_run_source(path: Path):
    """Open a directory or a temporary training-only view of a portable ZIP."""
    path = path.expanduser().resolve()
    if path.is_dir():
        yield path
        return
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Expected a runs directory or exported ZIP: {path}")
    with tempfile.TemporaryDirectory(prefix="server-export-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                _export_file(root, member.filename.rstrip("/"))
                if Path(member.filename).suffix in {".json", ".npz"}:
                    archive.extract(member, root)
        exports = list(root.rglob("export.json"))
        if len(exports) > 1:
            raise ValueError("Select one exported dataset ZIP at a time")
        yield exports[0].parent if exports else root


def _json(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"Symlink metadata is not supported: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


@dataclass(frozen=True)
class GlobalRun:
    run_id: str
    path: Path
    rewards: dict[str, Path]
    marked: bool
    member: dict | None = None

    @property
    def episode_dir(self) -> Path:
        return self.path.parent / "episodes" / "episode_000"


@dataclass(frozen=True)
class GlobalTask:
    task_id: str
    runs: tuple[GlobalRun, ...]
    export_path: Path | None = None
    dataset_defaults: dict | None = None
    export_digest: str | None = None
    dataset_digest: str | None = None

    def selected(self, source: str | None) -> list[GlobalRun]:
        # BC and Sparse use every marked trajectory. Robometer is a marker, not
        # an IQL reward. Other sources require that particular saved evaluation.
        if self.export_path is not None:
            missing = [run.run_id for run in self.runs
                       if source not in (None, "sparse") and source not in run.rewards]
            if missing:
                raise ValueError(f"Exported dataset is missing {source} rewards for: {missing}")
            return list(self.runs)
        return [run for run in self.runs if run.marked and
                (source in (None, "sparse") or source in run.rewards)]

    def summary(self) -> dict:
        return {"task_id": self.task_id, "candidates": len(self.runs),
                "marked": len(self.selected(None)),
                "rewards": {source: sum(run.marked and (source == "sparse" or source in run.rewards)
                                        for run in self.runs) for source in REWARD_SOURCES}}


def _export_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"Unsafe dataset export path: {value!r}")
    path = root / value
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset export traverses a symlink or escapes its root: {value}")
    return path


def _export_context(root: Path, project_id: str):
    path = next((candidate for candidate in (root / "export.json", root.parent / "export.json")
                 if candidate.is_file()), None)
    if path is None:
        return None, None, None
    payload = _json(path)
    if (payload.get("schema_version") != 1 or payload.get("complete") is not True
            or payload.get("kind") != "dataset_runs_export"
            or not isinstance(payload.get("members"), list) or not payload["members"]
            or not isinstance(payload.get("files"), dict) or not payload["files"]):
        raise ValueError(f"Incomplete or unsupported dataset export: {path}")
    runs = _export_file(path.parent, payload.get("runs_root"))
    if root.resolve() not in (path.parent.resolve(), runs.resolve()):
        raise ValueError("runs_root does not refer to the exported runs tree")
    frozen = _json(path.parent / "dataset.json")
    if not isinstance(frozen.get("members"), list) or not frozen["members"]:
        raise ValueError("Exported frozen dataset requires members")
    if any(not isinstance(member, dict) or not isinstance(member.get("run_id"), str)
           or not member["run_id"] for member in frozen["members"]):
        raise ValueError("Invalid frozen dataset members")
    immutable = {key: frozen.get(key) for key in ("task_id", "selection", "validation_fraction",
                 "split_seed", "success_consecutive_steps", "members")}
    if (frozen.get("schema_version") != 1 or frozen.get("project_id") != project_id
            or frozen.get("task_id") != payload.get("task_id")
            or frozen.get("id") != payload.get("dataset_id")
            or stable_hash(immutable) != frozen.get("dataset_sha256")
            or frozen["dataset_sha256"] != payload.get("dataset_sha256")):
        raise ValueError("Exported dataset identity or frozen selection hash mismatch")
    frozen_members = {member["run_id"]: member for member in frozen["members"]}
    if len(frozen_members) != len(frozen["members"]):
        raise ValueError("Duplicate frozen dataset run IDs")
    members = {}
    for member in payload["members"]:
        if not isinstance(member, dict) or not isinstance(member.get("run_id"), str):
            raise ValueError("Invalid exported dataset member")
        run_id = member.get("run_id")
        if run_id in members or run_id not in frozen_members:
            raise ValueError(f"Invalid or duplicate exported dataset member: {run_id}")
        if any(member.get(key) != frozen_members[run_id].get(key)
               for key in ("split", "root_run_id", "parent_run_id")):
            raise ValueError(f"Exported dataset member split/identity mismatch: {run_id}")
        directory = _export_file(path.parent, member.get("path"))
        if not directory.is_relative_to(runs):
            raise ValueError("Exported member is outside runs_root")
        members[run_id] = {**frozen_members[run_id], "run_path": directory / "run.json"}
    if set(members) != set(frozen_members):
        raise ValueError("Exported dataset members do not match the frozen selection")
    return path, frozen, members


def discover_tasks(root: Path, project_id: str) -> list[GlobalTask]:
    """Group by task identity, not directory names; accept dated and flat layouts."""
    if not root.is_dir():
        raise FileNotFoundError(f"Copied runs directory not found: {root}")
    root = root.resolve()
    export_path, frozen, members = _export_context(root, project_id)
    groups, seen = {}, set()
    paths = [member["run_path"] for member in members.values()] if members else sorted(root.rglob("run.json"))
    for path in paths:
        raw = _json(path)
        if (raw.get("project_id") not in (None, project_id)
                or raw.get("status") != "COMPLETED" or raw.get("error")):
            if members:
                raise ValueError(f"Exported member is not a completed run: {path}")
            continue
        run_id, task_id = raw.get("id"), raw.get("task_id")
        if not isinstance(run_id, str) or not run_id or not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Completed run has no run/task identity: {path}")
        if run_id in seen:
            raise ValueError(f"Duplicate copied run ID: {run_id}")
        seen.add(run_id)
        if members and (run_id not in members or raw.get("task_id") != frozen["task_id"]
                        or path != members[run_id]["run_path"]):
            raise ValueError(f"Exported run identity mismatch: {path}")
        episode = path.parent / "episodes" / "episode_000"
        rewards = {source: episode / f"trajectory_reward.{source}.json"
                   for source in REWARD_SOURCES if (episode / f"trajectory_reward.{source}.json").is_file()}
        legacy = episode / "trajectory_reward.json"
        if legacy.is_file():
            try:
                source = _json(legacy).get("source")
                if source in REWARD_SOURCES:
                    rewards.setdefault(source, legacy)
            except (ValueError, OSError):
                # Defer the error to reward use, so unrelated tasks and BC are
                # still selectable. Never fall back from a broken saved reward.
                for source in REWARD_SOURCES:
                    rewards.setdefault(source, legacy)
        native = episode / "rynnvalue_evaluation.json"
        if native.is_file():
            rewards.setdefault("rynnvalue", native)
        labels = episode / "stage_annotation.json"
        if labels.is_file():
            rewards.setdefault("stage", labels)
        marked = bool(rewards) or any((episode / name).is_file() for name in
                                     ("stage_annotation.json", "robometer_evaluation.json"))
        groups.setdefault(task_id, []).append(GlobalRun(run_id, path.resolve(), rewards, marked,
                                                       members[run_id] if members else None))
    defaults = ({key: frozen.get(key, True) for key in ("validation_fraction", "split_seed",
                 "success_consecutive_steps", "include_post_success")} if frozen else None)
    return [GlobalTask(task, tuple(runs), export_path, defaults,
                       sha256_file(export_path) if export_path else None,
                       sha256_file(export_path.parent / "dataset.json") if export_path else None)
            for task, runs in sorted(groups.items())]


def _export_snapshot(task: GlobalTask) -> dict:
    if (sha256_file(task.export_path) != task.export_digest
            or sha256_file(task.export_path.parent / "dataset.json") != task.dataset_digest):
        raise ValueError("Dataset export changed after discovery; select the input again")
    return _json(task.export_path)


def _check_export_asset(task: GlobalTask, export: dict, path: Path):
    root = task.export_path.parent
    if not path.is_relative_to(root):
        raise ValueError("Training input is outside dataset export")
    name = str(path.relative_to(root))
    _export_file(root, name)
    signature = export["files"].get(name)
    if (not isinstance(signature, dict) or type(signature.get("size")) is not int
            or signature["size"] < 0 or not isinstance(signature.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", signature["sha256"]) is None):
        raise ValueError(f"Required input is not bound to export manifest: {name}")
    if (not path.is_file() or path.stat().st_size != signature["size"]
            or sha256_file(path) != signature["sha256"]):
        raise ValueError(f"Exported file missing or hash mismatch: {name}")


def _freeze_export_selection(task: GlobalTask, directory: Path, raw: dict, data_overrides: dict):
    root = task.export_path.parent
    if directory.is_relative_to(root):
        raise ValueError("Training inputs must be outside the entire dataset export")
    export = _export_snapshot(task)
    _check_export_asset(task, export, root / "dataset.json")
    members = []
    for run in task.runs:
        member = copy.deepcopy(run.member)
        member.pop("run_path")
        artifacts = {name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("manifest", run.path), ("trajectory", run.episode_dir / "trajectory.npz"),
                               ("observations", run.episode_dir / "trajectory_observations.npz"))}
        for name, artifact in artifacts.items():
            _check_export_asset(task, export, Path(artifact["path"]))
            if member["artifacts"][name]["sha256"] != artifact["sha256"]:
                raise ValueError(f"Frozen dataset {name} changed: {run.run_id}")
        member["artifacts"] = artifacts
        members.append(member)
    frozen = _json(root / "dataset.json")
    immutable = {key: frozen[key] for key in ("task_id", "selection", "validation_fraction",
                 "split_seed", "success_consecutive_steps")}
    immutable["members"] = members
    selection = {"schema_version": 1, "id": frozen["id"], "project_id": frozen["project_id"],
                 **immutable, "dataset_sha256": stable_hash(immutable)}
    path = directory / "selection.json"
    atomic_json(path, selection)
    raw["data"].update(task.dataset_defaults)
    raw["data"].update(data_overrides)
    raw["data"]["selection_manifest"] = str(path)
    atomic_json(directory / "export_provenance.json", {
        "export_sha256": sha256_file(task.export_path), "dataset_id": frozen["id"],
        "source_dataset_sha256": export["dataset_sha256"], "selection_sha256": selection["dataset_sha256"]})


def choose_task(tasks: list[GlobalTask], requested: str | None) -> GlobalTask:
    matches = [task for task in tasks if task.task_id == requested]
    if requested is None and len(tasks) == 1:
        return tasks[0]
    if len(matches) != 1:
        raise ValueError(f"Select task_id from: {[task.task_id for task in tasks]}")
    return matches[0]


def saved_reward(run: GlobalRun, source: str) -> tuple[dict, Path]:
    path = run.rewards[source]
    metadata = _json(path)
    native = path.name == "rynnvalue_evaluation.json"
    if metadata.get("schema_version") not in ((5, 6) if native else (1,)):
        raise ValueError(f"Unsupported saved {source} evaluation: {path}")
    recipe = metadata.get("reward_config")
    if (metadata.get("run_id") != run.run_id or not isinstance(recipe, dict)
            or recipe.get("source", "rynnvalue") != source
            or (not native and metadata.get("source") != source)):
        raise ValueError(f"Global evaluation identity/reward mismatch: {path}")
    name = metadata.get("values_file", "rynnvalue_evaluation.npz" if native else "")
    if not isinstance(name, str) or not name or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError(f"Unsafe global evaluation values_file: {path}")
    return metadata, path.with_name(name)


def _stage_from_labels(run: GlobalRun, episode: dict, directory: Path, raw: dict) -> tuple[dict, Path]:
    """Same implicit global Stage semantics as the PC: labels + configured p."""
    from .stage_rewards import (
        stage_annotation_context, stage_chunk_reward, stage_scores, validate_stage_annotation,
    )
    arrays = _episode_timeline_arrays(episode)
    threshold = raw["data"]["success_consecutive_steps"]
    annotation = validate_stage_annotation(_json(run.rewards["stage"]), run_id=run.run_id,
        trajectory_sha256=episode["trajectory_sha256"], done=arrays["environment_done"],
        success_consecutive_steps=threshold)
    recipe = reward_derivation_config(raw["reward"])
    context = stage_annotation_context(annotation, done=arrays["environment_done"],
        success_consecutive_steps=threshold, exponent=recipe["stage_exponent"])
    scores = stage_scores(context)
    final = np.asarray([stage_chunk_reward(scores, chunk["start"], chunk["length"],
        recipe["gamma"], recipe["accumulate_primitive_steps"]) for chunk in episode["evaluation_chunks"]], dtype=np.float32)
    arrays.update(stage_score=scores, stage_chunk_reward=final, final_reward=final,
                  pbrs_chunk_reward=final, boundary_steps=np.asarray(episode["reward_boundaries"], dtype=np.int64))
    path = directory / "label_rewards" / f"{stable_hash(run.run_id)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"run_id": run.run_id, "source": "stage", "reward_config": recipe,
            "trajectory_sha256": episode["trajectory_sha256"],
            "observations_sha256": episode["observations_sha256"],
            "values_sha256": sha256_file(path), "stage_annotation": annotation}, path


def prepare_global_inputs(task: GlobalTask, directory: Path, raw: dict, source: str, *,
                          data_overrides: dict | None = None) -> dict:
    """Freeze all eligible raw runs and their saved values for one training run.

    Only runtime manifests/arrays are created. Device-specific inode/mtime hints
    and old absolute paths are deliberately ignored; content hashes are checked.
    """
    raw = copy.deepcopy(raw)
    needs_rewards = training_method(raw).requires_rewards
    selected = task.selected(source if needs_rewards else None)
    if not selected:
        raise ValueError(f"No marked trajectories with {source if needs_rewards else 'any'} evaluation: {task.task_id}")
    directory = directory.resolve()
    for run in selected:
        if directory.is_relative_to(run.path.parent):
            raise ValueError("Training inputs must be outside copied run directories")
        for name in ("trajectory.npz", "trajectory_observations.npz"):
            path = run.episode_dir / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Incomplete copied trajectory (no reconstruction on server): {path}")
    raw["paths"].update(dataset_sources=[str(run.path.parent) for run in selected],
                        work_dir=str(directory / "prepared"))
    raw["data"].update(task_ids=[task.task_id], selection_manifest=None, stage_annotations_manifest=None)
    if task.export_path is not None:
        _freeze_export_selection(task, directory, raw, data_overrides or {})
    raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None)
    config = LoadedConfig(directory / "effective.yaml", raw)
    prepare_dataset(config)
    prepared = load_manifest(config)
    episodes = {episode["run_id"]: episode for episode in prepared["episodes"]}
    if set(episodes) != {run.run_id for run in selected}:
        raise ValueError("Prepared runs differ from the selected global records")
    if not needs_rewards:
        return raw
    # Sparse has no learned evaluator; its complete definition comes from done.
    if source == "sparse":
        path = materialize_reward_manifest(config)
        raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path),
                             version_id="server-sparse")
        return raw
    entries = []
    for run in selected:
        episode = episodes[run.run_id]
        if run.rewards[source].name == "stage_annotation.json":
            if task.export_path:
                _check_export_asset(task, _export_snapshot(task), run.rewards[source])
            metadata, values = _stage_from_labels(run, episode, directory, raw)
        else:
            if task.export_path:
                _check_export_asset(task, _export_snapshot(task), run.rewards[source])
            metadata, values = saved_reward(run, source)
            if task.export_path:
                _check_export_asset(task, _export_snapshot(task), values)
        for key in ("trajectory_sha256", "observations_sha256"):
            if metadata.get(key) != episode[key]:
                raise ValueError(f"Global evaluation {key} mismatch: {run.run_id}")
        saved = metadata.get("episode", {})
        for key in ("source_manifest_sha256", "prompt", "task_id", "terminal_step", "resume_step",
                    "success_consecutive_steps"):
            if key in saved and saved[key] != episode[key]:
                raise ValueError(f"Global evaluation {key} mismatch: {run.run_id}")
        for key in ("action_horizon", "action_dim", "proprio_dim", "control_hz", "success_consecutive_steps"):
            header = metadata.get("prepared", {})
            if key in header and header[key] != raw["data"][key]:
                raise ValueError(f"Global evaluation {key} differs from training: {run.run_id}")
        digest = metadata.get("values_sha256")
        if values.is_symlink() or not values.is_file() or sha256_file(values) != digest:
            raise ValueError(f"Global evaluation arrays missing or hash mismatch: {run.run_id}")
        recipe = {**metadata["reward_config"], "source": source}
        with np.load(values, allow_pickle=False) as arrays:
            if needs_rynnvalue(recipe):
                validate_official_outputs({key: arrays[key] for key in OFFICIAL_OUTPUT_KEYS},
                                          len(episode["reward_boundaries"]))
            if needs_stage(recipe):
                scores = arrays["stage_score"]
                if scores.shape != (episode["recorded_action_count"] + 1,) or not np.isfinite(scores).all():
                    raise ValueError(f"Global Stage timeline is incomplete: {run.run_id}")
        destination = directory / "rewards" / f"{digest}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(values, destination)
        if sha256_file(destination) != digest:
            raise ValueError(f"Global evaluation changed while copying: {run.run_id}")
        # Validate the original indices too, not just our freshly rebuilt metadata.
        for key, expected in _episode_reward_metadata(episode).items():
            previous = metadata.get("entry", {})
            if key in previous and previous[key] != expected:
                raise ValueError(f"Global evaluation {key} mismatch: {run.run_id}")
        entry = {"run_id": run.run_id, **_episode_reward_metadata(episode),
                 "reward_path": str(destination), "annotation_path": str(destination),
                 "reward_sha256": digest, "annotation_sha256": digest,
                 "saved_reward_config": recipe,
                 "saved_annotation_config": metadata.get("annotation_config", {}),
                 "annotator": metadata.get("annotator", {}),
                 "official_outputs": metadata.get("official_outputs", {}),
                 "source_evaluated_at": metadata.get("evaluated_at"),
                 "source_evaluation_id": metadata.get("evaluation_id"),
                 "source_origin": metadata.get("origin", "global"),
                 "source_metadata_sha256": sha256_file(run.rewards[source])}
        entries.append(entry)
        atomic_json(directory / "metadata" / f"{stable_hash(run.run_id)}.json", metadata)
    identifier = "global_" + stable_hash(entries)[:16]
    index = {"schema_version": 1, "kind": "derived_iql_reward", "complete": True,
             "binding_kind": "global_trajectory_snapshots", "version_id": identifier,
             "dataset_sha256": prepared["dataset_sha256"],
             "reward_config": reward_derivation_config(raw["reward"]), "episodes": entries}
    path = directory / "reward_manifest.json"
    atomic_json(path, index)
    raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id=identifier)
    # This verifies chunk order/length and reuses the normal private gamma adapter.
    load_pinned_reward_index(config)
    return raw
