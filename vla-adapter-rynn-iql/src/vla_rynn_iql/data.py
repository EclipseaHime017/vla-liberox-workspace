from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import LoadedConfig
from .io import atomic_json, sha256_file, stable_hash


LOG = logging.getLogger(__name__)
MANIFEST_NAME = "dataset_manifest.json"
MANIFEST_SCHEMA_VERSION = 4
REPLAY_POLICY = "full_recording_v1"


@dataclass(frozen=True)
class PreparedPaths:
    manifest: Path
    reward_dir: Path


def confirmed_terminal_step(done: np.ndarray, consecutive_steps: int) -> int | None:
    """Return the action index that confirms a consecutive success streak."""
    if consecutive_steps < 1:
        raise ValueError("consecutive_steps must be positive")
    streak = 0
    for index, value in enumerate(np.asarray(done, dtype=bool)):
        streak = streak + 1 if bool(value) else 0
        if streak >= consecutive_steps:
            return index
    return None


def action_source_segments(sources: list[str], end: int) -> list[dict[str, Any]]:
    """Return a compact partition of ``[0, end)`` by action producer."""
    if not 0 < end <= len(sources):
        raise ValueError(f"Invalid action-source endpoint {end} for {len(sources)} actions")
    segments: list[dict[str, Any]] = []
    start = 0
    current = sources[0]
    for index in range(1, end):
        if sources[index] == current:
            continue
        segments.append({"start": start, "end": index, "action_source": current})
        start, current = index, sources[index]
    segments.append({"start": start, "end": end, "action_source": current})
    return segments


def build_semi_mdp_chunks(
    *,
    first: int,
    end: int,
    horizon: int,
    source_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build variable-duration transitions without crossing controller boundaries.

    The fixed horizon is only a maximum. Action-source changes and branch resume
    points are hard boundaries, so a takeover can end the final policy chunk at
    its actual executed length instead of pretending that the full horizon ran.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 <= first < end:
        raise ValueError(f"Invalid replay interval [{first}, {end})")
    boundaries = {first, end}
    segment_lookup: list[dict[str, Any]] = []
    for segment in source_segments:
        segment_start = max(first, int(segment["start"]))
        segment_end = min(end, int(segment["end"]))
        if segment_start >= segment_end:
            continue
        source = str(segment["action_source"])
        segment_lookup.append({
            "start": segment_start, "end": segment_end, "action_source": source,
        })
        # Each controller/source segment owns its own horizon grid. A source
        # change therefore starts a new macro-action sequence.
        boundaries.update(range(segment_start, segment_end, horizon))
        boundaries.add(segment_end)
    covered = sum(segment["end"] - segment["start"] for segment in segment_lookup)
    if covered != end - first:
        raise ValueError(
            f"Action-source segments cover {covered} actions, expected {end - first}"
        )

    ordered = sorted(boundaries)
    chunks: list[dict[str, Any]] = []
    segment_index = 0
    for start, stop in zip(ordered, ordered[1:]):
        while segment_lookup[segment_index]["end"] <= start:
            segment_index += 1
        segment = segment_lookup[segment_index]
        if not (segment["start"] <= start < stop <= segment["end"]):
            raise ValueError(f"Chunk [{start}, {stop}) crosses an action-source boundary")
        chunks.append({
            "start": start,
            "length": stop - start,
            "end": stop,
            "action_source": segment["action_source"],
            "transition_type": str(segment["action_source"]),
            "interrupted": False,
            "copied_prefix": False,
        })
    return chunks


def replay_chunks(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Return full-recording replay without changing prepared/cache identities.

    Older schema-4 manifests kept the post-success tail in evaluation_chunks
    and annotated its rewards at those same indices. Reuse that complete list
    in memory, while retaining the manifest and its existing reward hash.
    """
    recorded_end = int(episode["recorded_action_count"])
    prepared = episode["chunks"]
    complete = bool(prepared) and int(prepared[-1]["end"]) == recorded_end
    chunks = prepared if complete else episode.get("evaluation_chunks", [])
    run_id = episode["run_id"]
    error = (
        f"Run {run_id} lacks aligned full-recording replay chunks; "
        "rerun preparation and reward materialization for the complete recording"
    )
    expected_start = 0
    for chunk in chunks:
        start, end, length = int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
        if start != expected_start or length <= 0 or end - start != length:
            raise ValueError(error)
        expected_start = end
    if expected_start != recorded_end or not chunks:
        raise ValueError(error)
    if not complete:
        if len(prepared) > len(chunks) or any(
            any(original[key] != full[key] for key in ("start", "end", "length", "action_source"))
            for original, full in zip(prepared, chunks)
        ):
            raise ValueError(error)
        return [
            {**chunk, "transition_type": chunk["action_source"]}
            if chunk.get("transition_type") == "post_terminal_evaluation" else chunk
            for chunk in chunks
        ]
    return chunks


def iter_unique_replay_chunks(
    episodes: list[dict[str, Any]], *, split: str | None = None,
) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield replay chunks while counting physically copied prefixes once."""
    full_chunks = {str(episode["run_id"]): replay_chunks(episode) for episode in episodes}
    copied_prefix_keys = {
        (
            str(episode["root_run_id"]), int(chunk["start"]),
            int(chunk["end"]), str(chunk["action_source"]),
        )
        for episode in episodes
        for chunk in full_chunks[str(episode["run_id"])]
        if bool(chunk.get("copied_prefix", False))
    }
    seen_copied_prefixes: set[tuple[str, int, int, str]] = set()
    ordered = sorted(
        episodes,
        key=lambda item: (item.get("kind") == "branch", str(item["run_id"])),
    )
    for episode in ordered:
        if split is not None and episode["split"] != split:
            continue
        for chunk_index, chunk in enumerate(full_chunks[str(episode["run_id"])]):
            key = (
                str(episode["root_run_id"]), int(chunk["start"]),
                int(chunk["end"]), str(chunk["action_source"]),
            )
            if key in copied_prefix_keys:
                if key in seen_copied_prefixes:
                    continue
                seen_copied_prefixes.add(key)
            yield episode, chunk_index


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def _source_roots(config: LoadedConfig) -> list[Path]:
    work = Path(config.section("paths")["work_dir"])
    roots: list[Path] = []
    for source_value in config.section("paths")["dataset_sources"]:
        source = Path(source_value)
        if source.is_dir():
            roots.append(source)
        elif source.is_file() and source.suffix.lower() == ".zip":
            fingerprint = sha256_file(source)[:16]
            target = work / "imports" / fingerprint
            marker = target / ".complete"
            if not marker.exists():
                if target.exists():
                    shutil.rmtree(target)
                _safe_extract(source, target)
                marker.touch()
            roots.append(target)
        else:
            raise FileNotFoundError(f"Dataset source not found or unsupported: {source}")
    return roots


def _selected_runs(config: LoadedConfig) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Return exact UI-selected run manifests, or an empty map for CLI discovery."""
    value = config.section("data").get("selection_manifest")
    if value is None:
        return {}, None
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Training dataset selection manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("members"), list):
        raise ValueError(f"Unsupported training dataset manifest: {path}")
    immutable = {
        "task_id": payload.get("task_id"),
        "selection": payload.get("selection"),
        "validation_fraction": payload.get("validation_fraction"),
        "split_seed": payload.get("split_seed"),
        "success_consecutive_steps": payload.get("success_consecutive_steps"),
        "members": payload.get("members"),
    }
    if stable_hash(immutable) != payload.get("dataset_sha256"):
        raise ValueError("Immutable training dataset manifest hash changed")
    if payload.get("project_id") != config.section("data")["project_id"]:
        raise ValueError("Training dataset project_id does not match the config")
    configured_tasks = set(config.section("data")["task_ids"])
    if configured_tasks and payload.get("task_id") not in configured_tasks:
        raise ValueError("Training dataset task does not match data.task_ids")
    selected: dict[str, dict[str, Any]] = {}
    for member in payload["members"]:
        run_id = member.get("run_id")
        artifacts = member.get("artifacts") or {}
        source = artifacts.get("manifest") or {}
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        if not isinstance(run_id, str) or not run_id or run_id in selected:
            raise ValueError("Training dataset members must have unique non-empty run_id values")
        if member.get("split") not in {"train", "validation"}:
            raise ValueError(f"Selected run has invalid split: {run_id}")
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"Selected run manifest is missing: {source_path}")
        if sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"Selected run manifest SHA256 changed: {run_id}")
        selected[run_id] = {**member, "manifest_path": source_path}
    if not selected:
        raise ValueError("Training dataset selection is empty")
    return selected, payload


def _column(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            return float(value)
    return default


def _materialize_export(run_dir: Path, cache_dir: Path) -> tuple[Path, Path]:
    episode = run_dir / "episodes" / "episode_000"
    trajectory_csv = episode / "trajectory.csv"
    video = episode / "vla_views.mp4"
    if not trajectory_csv.is_file() or not video.is_file():
        raise FileNotFoundError(f"Exported run lacks trajectory.csv or vla_views.mp4: {run_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = cache_dir / "trajectory.npz"
    observations_path = cache_dir / "trajectory_observations.npz"
    if trajectory_path.is_file() and observations_path.is_file():
        return trajectory_path, observations_path

    with trajectory_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"Trajectory CSV must contain N+1 rows: {trajectory_csv}")
    state_rows, action_rows = rows, rows[:-1]
    time_seconds = np.asarray([float(r["time_seconds"]) for r in state_rows], dtype=np.float64)
    eef_position = np.asarray([[float(r[f"eef_{axis}"]) for axis in "xyz"] for r in state_rows], dtype=np.float32)
    eef_axis_angle = np.asarray([[float(r[f"axis_angle_{axis}"]) for axis in "xyz"] for r in state_rows], dtype=np.float32)
    gripper_qpos = np.asarray([[float(r["gripper_left"]), float(r["gripper_right"])] for r in state_rows], dtype=np.float32)
    raw_names = tuple(f"vla_action_{name}" for name in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    env_names = tuple(f"action_{name}" for name in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))
    raw_action = np.asarray([[float(r[name]) for name in raw_names] for r in action_rows], dtype=np.float32)
    env_action = np.asarray([[float(r[name]) for name in env_names] for r in action_rows], dtype=np.float32)
    reward = np.asarray([_column(r, ("reward",)) for r in action_rows], dtype=np.float32)
    done = np.asarray([str(r.get("done", "false")).lower() == "true" for r in action_rows], dtype=bool)
    action_source = np.asarray([r.get("action_source", "unknown") for r in action_rows])
    np.savez_compressed(
        trajectory_path, time_seconds=time_seconds,
        eef_position=eef_position, eef_axis_angle=eef_axis_angle,
        gripper_qpos=gripper_qpos, raw_action=raw_action, env_action=env_action,
        reward=reward, done=done, action_source=action_source,
    )

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required to import dataset export ZIP files") from exc
    frames = [np.asarray(frame, dtype=np.uint8) for frame in iio.imiter(video)]
    if len(frames) != len(action_rows):
        raise ValueError(f"Video/action count mismatch for {run_dir}: {len(frames)} != {len(action_rows)}")
    width = frames[0].shape[1]
    if width % 2:
        raise ValueError(f"VLA mosaic width must be even, got {width}")
    split = width // 2
    agent = [frame[:, :split] for frame in frames]
    wrist = [frame[:, split:] for frame in frames]
    # Exports contain one frame per action. Terminal next observations are only
    # used with mask=0, so repeating the last frame preserves N+1 indexing.
    agent.append(agent[-1].copy())
    wrist.append(wrist[-1].copy())
    np.savez_compressed(
        observations_path,
        agentview_image=np.stack(agent), wrist_image=np.stack(wrist),
    )
    return trajectory_path, observations_path


def _load_run(run_json: Path, config: LoadedConfig) -> dict[str, Any] | None:
    run = json.loads(run_json.read_text(encoding="utf-8"))
    if run.get("status") != "COMPLETED" or run.get("error"):
        return None
    data_cfg = config.section("data")
    task_ids = set(data_cfg["task_ids"])
    if task_ids and run.get("task_id") not in task_ids:
        return None
    episode_dir = run_json.parent / "episodes" / "episode_000"
    trajectory = episode_dir / "trajectory.npz"
    observations = episode_dir / "trajectory_observations.npz"
    observation_orientation = "libero_raw"
    if not trajectory.is_file() or not observations.is_file():
        cache = Path(config.section("paths")["work_dir"]) / "materialized" / str(run["id"])
        trajectory, observations = _materialize_export(run_json.parent, cache)
        observation_orientation = "vla_policy"
    prompt = run.get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Run has no task prompt: {run_json}")
    if run.get("project_id") not in (None, data_cfg["project_id"]):
        raise ValueError(
            f"Run project_id={run.get('project_id')!r} does not match "
            f"data.project_id={data_cfg['project_id']!r}: {run_json}"
        )
    with np.load(trajectory, allow_pickle=False) as arrays:
        required = {
            "time_seconds", "eef_position", "eef_axis_angle", "gripper_qpos",
            "raw_action", "env_action", "done", "action_source",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"{trajectory} is missing arrays: {missing}")
        recorded_action_count = len(arrays["env_action"])
        if recorded_action_count < 1:
            raise ValueError(f"Trajectory must contain at least one action: {trajectory}")
        if len(arrays["eef_position"]) != recorded_action_count + 1:
            raise ValueError(f"N+1 state invariant failed: {trajectory}")
        for key, shape in {
            "eef_axis_angle": (recorded_action_count + 1, 3),
            "gripper_qpos": (recorded_action_count + 1, 2),
            "raw_action": (recorded_action_count, 7),
            "env_action": (recorded_action_count, 7),
            "done": (recorded_action_count,),
            "action_source": (recorded_action_count,),
        }.items():
            if arrays[key].shape != shape:
                raise ValueError(
                    f"Invalid {key} shape {arrays[key].shape}; expected {shape}: {trajectory}"
                )
        numeric_keys = (
            "time_seconds", "eef_position", "eef_axis_angle", "gripper_qpos",
            "raw_action", "env_action",
        )
        for key in numeric_keys:
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} contains NaN or Inf: {trajectory}")
        times = np.asarray(arrays["time_seconds"], dtype=np.float64)
        expected_period = 1.0 / float(data_cfg["control_hz"])
        if times.shape != (recorded_action_count + 1,) or not np.allclose(
            np.diff(times), expected_period, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"Trajectory is not recorded on the configured {data_cfg['control_hz']} Hz grid: "
                f"{trajectory}"
            )
        env_action = np.asarray(arrays["env_action"], dtype=np.float32)
        if np.any(np.abs(env_action) > 1.0001):
            raise ValueError(f"env_action exceeds normalized OSC_POSE range [-1, 1]: {trajectory}")
        sources = [str(value) for value in arrays["action_source"]]
        policy_mask = np.asarray([value in {"policy", "policy_requery"} for value in sources])
        if np.any(policy_mask):
            raw_action = np.asarray(arrays["raw_action"], dtype=np.float32)[policy_mask]
            executed = env_action[policy_mask]
            if not np.allclose(raw_action[:, :6], executed[:, :6], rtol=0.0, atol=1e-5):
                raise ValueError(f"Policy action round-trip mismatch in OSC axes: {trajectory}")
            expected_gripper = -np.sign(2.0 * raw_action[:, 6] - 1.0)
            if not np.array_equal(expected_gripper, executed[:, 6]):
                raise ValueError(f"Policy gripper round-trip mismatch: {trajectory}")
        done = np.asarray(arrays["done"], dtype=bool)
        raw_done_true_count = int(np.count_nonzero(done))
        raw_environment_success = raw_done_true_count > 0
        required_success_steps = int(data_cfg["success_consecutive_steps"])
        terminal_step = confirmed_terminal_step(done, required_success_steps)
        environment_success = terminal_step is not None
        # Success confirmation remains the reward/outcome boundary. Replay keeps
        # the complete recording, including actions after the confirming streak.
        action_count = recorded_action_count
        success_end = terminal_step + 1 if terminal_step is not None else recorded_action_count
        trailing_action_count = recorded_action_count - success_end
        post_terminal_false_count = (
            int(np.count_nonzero(~done[success_end:])) if terminal_step is not None else 0
        )
        recorded_success = bool(run.get("success", False))
        if recorded_success != raw_environment_success:
            raise ValueError(f"run.success and environment done disagree: {run_json}")
    with np.load(observations, allow_pickle=False) as images:
        for key in ("agentview_image", "wrist_image"):
            if key not in images or images[key].shape[0] != recorded_action_count + 1:
                raise ValueError(f"Invalid {key} alignment: {observations}")
            if images[key].ndim != 4 or images[key].shape[-1] != 3:
                raise ValueError(f"Invalid {key} image dimensions: {observations}")
            if images[key].dtype != np.uint8:
                raise ValueError(f"{key} must contain uint8 RGB images: {observations}")
    kind = str(run.get("kind", "original"))
    resume = int(run.get("resume_step") or 0) if kind == "branch" else 0
    if not 0 <= resume < action_count:
        raise ValueError(f"Invalid resume_step={resume} for {run_json}")
    if kind == "branch" and any(
        value not in {"human", "policy_requery"} for value in sources[resume:action_count]
    ):
        raise ValueError(f"Branch suffix contains unexpected action_source values: {run_json}")
    if trailing_action_count:
        LOG.info(
            "Run %s confirmed success at action %d; retaining %d post-success actions for replay "
            "(%d later done=False)",
            run["id"], terminal_step, trailing_action_count, post_terminal_false_count,
        )
    elif raw_environment_success and not environment_success:
        LOG.warning(
            "Run %s contains %d done=True actions but no streak of %d; treating it as failure",
            run["id"], raw_done_true_count, required_success_steps,
        )
    return {
        "run_id": str(run["id"]),
        "root_run_id": str(run.get("root_session_id") or run["id"]),
        "parent_run_id": run.get("parent_session_id"),
        "kind": kind,
        "control_mode": str(run.get("control_mode", "policy")),
        "resume_step": resume if kind == "branch" else None,
        "task_id": str(run.get("task_id") or ""),
        "task_name": str(run.get("task_name") or ""),
        "prompt": prompt.strip(),
        "success": environment_success,
        "recorded_success": recorded_success,
        "raw_done_true_count": raw_done_true_count,
        "success_consecutive_steps": required_success_steps,
        "success_streak_start": (
            terminal_step - required_success_steps + 1 if terminal_step is not None else None
        ),
        "action_count": action_count,
        "recorded_action_count": recorded_action_count,
        "terminal_step": terminal_step,
        "trailing_action_count": trailing_action_count,
        "post_terminal_false_count": post_terminal_false_count,
        "action_source_segments": action_source_segments(sources, action_count),
        "recorded_action_source_segments": action_source_segments(
            sources, recorded_action_count,
        ),
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": sha256_file(trajectory),
        "observations_path": str(observations.resolve()),
        "observations_sha256": sha256_file(observations),
        "observation_orientation": observation_orientation,
        "source_manifest": str(run_json.resolve()),
        "source_manifest_sha256": sha256_file(run_json),
    }


def _split(root_run_id: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{root_run_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def prepare_dataset(config: LoadedConfig) -> PreparedPaths:
    work = Path(config.section("paths")["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    discovered: dict[str, dict[str, Any]] = {}
    selected, selection_payload = _selected_runs(config)
    if selected:
        manifest_paths = [member["manifest_path"] for member in selected.values()]
    else:
        manifest_paths = [
            run_json
            for root in _source_roots(config)
            for run_json in sorted(root.rglob("run.json"))
        ]
    for run_json in manifest_paths:
        episode = _load_run(run_json, config)
        if episode is None:
            if selected:
                raise ValueError(
                    f"Selected run is not a valid completed episode: {run_json}"
                )
            continue
        run_id = episode["run_id"]
        if selected and run_id not in selected:
            raise ValueError(f"Selected manifest resolved to unexpected run id: {run_id}")
        if run_id in discovered and episode["source_manifest_sha256"] != discovered[run_id]["source_manifest_sha256"]:
            raise ValueError(f"Conflicting duplicate run id: {run_id}")
        if selected:
            member = selected[run_id]
            expected_resume = int(episode["resume_step"] or 0) if episode["kind"] == "branch" else 0
            if member.get("resume_step") != expected_resume:
                raise ValueError(f"Selected resume_step changed: {run_id}")
            if member.get("end_step") != episode["recorded_action_count"]:
                raise ValueError(f"Selected recorded action length changed: {run_id}")
            for name, episode_key in (
                ("trajectory", "trajectory_sha256"),
                ("observations", "observations_sha256"),
            ):
                expected = member["artifacts"][name]["sha256"]
                if episode[episode_key] != expected:
                    raise ValueError(f"Selected {name} SHA256 changed: {run_id}")
        discovered[run_id] = episode
    if selected and set(discovered) != set(selected):
        raise ValueError("Prepared runs do not exactly match the immutable selection")
    if not discovered:
        raise RuntimeError("No valid completed episodes were found")

    horizon = int(config.section("data")["action_horizon"])
    seed = int(config.section("data")["split_seed"])
    fraction = float(config.section("data")["validation_fraction"])
    episodes = sorted(discovered.values(), key=lambda item: item["run_id"])
    # Keep every episode's transition description independent of the other
    # episodes selected for this particular dataset.  This is important for
    # trajectory-level reward annotations: selecting another sibling branch
    # must not change the cache key or boundary layout of an existing run.
    # Every branch keeps its complete physical trajectory for trajectory-level
    # RynnValue evaluation. Copied parent prefixes are de-duplicated only when
    # ReplayDataset assembles the training replay.
    for episode in episodes:
        episode["split"] = (
            selected[episode["run_id"]]["split"]
            if selected else _split(episode["root_run_id"], seed, fraction)
        )
        resume_step = (
            int(episode["resume_step"] or 0) if episode["kind"] == "branch" else 0
        )
        # Preserve the existing success-aligned reward boundaries and annotation
        # cache keys, then promote the complete recorded tail into training.
        success_end = (
            int(episode["terminal_step"]) + 1
            if episode["terminal_step"] is not None else int(episode["recorded_action_count"])
        )
        chunks = build_semi_mdp_chunks(
            first=0,
            end=success_end,
            horizon=horizon,
            source_segments=episode["action_source_segments"],
        )
        if success_end < int(episode["recorded_action_count"]):
            chunks.extend(build_semi_mdp_chunks(
                first=success_end,
                end=int(episode["recorded_action_count"]),
                horizon=horizon,
                source_segments=episode["recorded_action_source_segments"],
            ))
        if episode["kind"] == "branch":
            for chunk in chunks:
                if int(chunk["end"]) > resume_step:
                    continue
                if str(chunk["action_source"]) != "policy":
                    raise ValueError(
                        f"Branch {episode['run_id']} has non-policy action "
                        f"before resume_step={resume_step}: {chunk['action_source']!r}"
                    )
                chunk["copied_prefix"] = True
                if int(chunk["end"]) == resume_step and int(chunk["length"]) < horizon:
                    chunk["transition_type"] = "policy_interrupted"
                    chunk["interrupted"] = True
                else:
                    chunk["transition_type"] = "policy_prefix"
        episode["chunks"] = chunks
        # The historical evaluation-only label remains in annotation metadata
        # for cache compatibility. It does not control replay inclusion.
        evaluation_chunks = [
            {**chunk, "transition_type": "post_terminal_evaluation"}
            if int(chunk["start"]) >= success_end else dict(chunk)
            for chunk in chunks
        ]
        episode["evaluation_chunks"] = evaluation_chunks
        episode["reward_boundaries"] = sorted({
            value for chunk in evaluation_chunks
            for value in (chunk["start"], chunk["end"])
        })
    if not selected and all(ep["split"] == "validation" for ep in episodes):
        # Tiny smoke-test datasets still need at least one train root.
        root = episodes[0]["root_run_id"]
        for episode in episodes:
            if episode["root_run_id"] == root:
                episode["split"] = "train"
    successes = sum(int(ep["success"]) for ep in episodes)
    if successes == 0:
        message = (
            "Dataset contains no success confirmed by the consecutive-step threshold; "
            "pipeline may run but policy improvement is not expected"
        )
        if config.section("data")["allow_no_success"]:
            LOG.warning(message)
        else:
            raise RuntimeError(message)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "replay_policy": REPLAY_POLICY,
        "chunking": "variable_duration_action_source_v2_full_branch_prefix",
        "source_dataset_id": None if selection_payload is None else selection_payload["id"],
        "source_dataset_sha256": None if selection_payload is None else selection_payload["dataset_sha256"],
        "config_sha256": config.digest,
        "dataset_sha256": stable_hash([{k: ep[k] for k in (
            "run_id", "source_manifest_sha256", "trajectory_sha256",
            "observations_sha256", "task_id", "prompt", "resume_step",
            "action_count", "recorded_action_count", "terminal_step",
            "trailing_action_count", "post_terminal_false_count", "recorded_success",
            "raw_done_true_count", "success_consecutive_steps", "success_streak_start",
            "action_source_segments", "recorded_action_source_segments", "chunks",
            "evaluation_chunks", "split")}
            for ep in episodes]),
        "action_horizon": horizon,
        "action_dim": config.section("data")["action_dim"],
        "proprio_dim": config.section("data")["proprio_dim"],
        "control_hz": config.section("data")["control_hz"],
        "success_consecutive_steps": config.section("data")["success_consecutive_steps"],
        "episode_count": len(episodes),
        "success_count": successes,
        "trajectory_chunk_count": sum(len(ep["chunks"]) for ep in episodes),
        "chunk_count": sum(1 for _ in iter_unique_replay_chunks(episodes)),
        "episodes": episodes,
    }
    manifest = work / MANIFEST_NAME
    atomic_json(manifest, payload)
    return PreparedPaths(manifest=manifest, reward_dir=work / "rewards")


def load_manifest(config: LoadedConfig) -> dict[str, Any]:
    path = Path(config.section("paths")["work_dir"]) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported dataset manifest schema; rerun prepare_dataset.py before annotation"
        )
    return payload


def iter_episode_arrays(manifest: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]]:
    for episode in manifest["episodes"]:
        with np.load(episode["trajectory_path"], allow_pickle=False) as trajectory, np.load(
            episode["observations_path"], allow_pickle=False
        ) as observations:
            yield episode, {key: trajectory[key] for key in trajectory.files}, {
                key: observations[key] for key in observations.files
            }
