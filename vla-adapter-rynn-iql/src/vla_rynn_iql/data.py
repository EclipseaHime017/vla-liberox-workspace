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


@dataclass(frozen=True)
class PreparedPaths:
    manifest: Path
    reward_dir: Path


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
        action_count = len(arrays["env_action"])
        if action_count < 1:
            raise ValueError(f"Trajectory must contain at least one action: {trajectory}")
        if len(arrays["eef_position"]) != action_count + 1:
            raise ValueError(f"N+1 state invariant failed: {trajectory}")
        for key, shape in {
            "eef_axis_angle": (action_count + 1, 3),
            "gripper_qpos": (action_count + 1, 2),
            "raw_action": (action_count, 7),
            "env_action": (action_count, 7),
            "done": (action_count,),
            "action_source": (action_count,),
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
        if times.shape != (action_count + 1,) or not np.allclose(
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
        environment_success = bool(np.any(arrays["done"]))
        done_indices = np.flatnonzero(arrays["done"])
        if done_indices.size and (
            done_indices.size != 1 or int(done_indices[0]) != action_count - 1
        ):
            raise ValueError(f"Trajectory contains actions after environment completion: {trajectory}")
        if bool(run.get("success", False)) != environment_success:
            raise ValueError(f"run.success and environment done disagree: {run_json}")
    with np.load(observations, allow_pickle=False) as images:
        for key in ("agentview_image", "wrist_image"):
            if key not in images or images[key].shape[0] != action_count + 1:
                raise ValueError(f"Invalid {key} alignment: {observations}")
            if images[key].ndim != 4 or images[key].shape[-1] != 3:
                raise ValueError(f"Invalid {key} image dimensions: {observations}")
            if images[key].dtype != np.uint8:
                raise ValueError(f"{key} must contain uint8 RGB images: {observations}")
    kind = str(run.get("kind", "original"))
    resume = int(run.get("resume_step") or 0) if kind == "branch" else 0
    if not 0 <= resume < action_count:
        raise ValueError(f"Invalid resume_step={resume} for {run_json}")
    if kind == "branch" and any(value not in {"human", "policy_requery"} for value in sources[resume:]):
        raise ValueError(f"Branch suffix contains unexpected action_source values: {run_json}")
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
        "success": bool(run.get("success", False)),
        "action_count": action_count,
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
    for root in _source_roots(config):
        for run_json in sorted(root.rglob("run.json")):
            episode = _load_run(run_json, config)
            if episode is None:
                continue
            run_id = episode["run_id"]
            if run_id in discovered and episode["source_manifest_sha256"] != discovered[run_id]["source_manifest_sha256"]:
                raise ValueError(f"Conflicting duplicate run id: {run_id}")
            discovered[run_id] = episode
    if not discovered:
        raise RuntimeError("No valid completed episodes were found")

    horizon = int(config.section("data")["action_horizon"])
    seed = int(config.section("data")["split_seed"])
    fraction = float(config.section("data")["validation_fraction"])
    episodes = sorted(discovered.values(), key=lambda item: item["run_id"])
    for episode in episodes:
        episode["split"] = _split(episode["root_run_id"], seed, fraction)
        first = int(episode["resume_step"] or 0) if episode["kind"] == "branch" else 0
        chunks = []
        for start in range(first, episode["action_count"], horizon):
            length = min(horizon, episode["action_count"] - start)
            chunks.append({"start": start, "length": length, "end": start + length})
        episode["chunks"] = chunks
        episode["reward_boundaries"] = sorted({value for chunk in chunks for value in (chunk["start"], chunk["end"])})
    if all(ep["split"] == "validation" for ep in episodes):
        # Tiny smoke-test datasets still need at least one train root.
        root = episodes[0]["root_run_id"]
        for episode in episodes:
            if episode["root_run_id"] == root:
                episode["split"] = "train"
    successes = sum(int(ep["success"]) for ep in episodes)
    if successes == 0:
        message = "Dataset contains no environment-confirmed success; pipeline may run but policy improvement is not expected"
        if config.section("data")["allow_no_success"]:
            LOG.warning(message)
        else:
            raise RuntimeError(message)
    payload = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "dataset_sha256": stable_hash([{k: ep[k] for k in (
            "run_id", "source_manifest_sha256", "trajectory_sha256",
            "observations_sha256", "task_id", "prompt", "resume_step",
            "action_count", "split")}
            for ep in episodes]),
        "action_horizon": horizon,
        "action_dim": config.section("data")["action_dim"],
        "proprio_dim": config.section("data")["proprio_dim"],
        "control_hz": config.section("data")["control_hz"],
        "episode_count": len(episodes),
        "success_count": successes,
        "chunk_count": sum(len(ep["chunks"]) for ep in episodes),
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
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    return payload


def iter_episode_arrays(manifest: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]]:
    for episode in manifest["episodes"]:
        with np.load(episode["trajectory_path"], allow_pickle=False) as trajectory, np.load(
            episode["observations_path"], allow_pickle=False
        ) as observations:
            yield episode, {key: trajectory[key] for key in trajectory.files}, {
                key: observations[key] for key in observations.files
            }
