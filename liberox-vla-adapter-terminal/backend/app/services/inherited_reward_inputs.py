"""CPU-only adapters for global labels whose old Prepare directory is gone."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import sys
import threading
from pathlib import Path

import numpy as np

_IMPORT_LOCK = threading.Lock()


def offline_module(root: Path, part: str):
    """Load a lightweight local definition without sys.path or GPU changes."""
    package = (root / "src/vla_rynn_iql").resolve()
    name = "_inherited_rewards_" + hashlib.sha256(str(package).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                          submodule_search_locations=[str(package)])
            if spec is None or spec.loader is None:
                raise RuntimeError("Offline reward definitions are unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return importlib.import_module(f"{name}.{part}")


def reward_core(root: Path):
    return tuple(offline_module(root, part) for part in ("config", "data", "rewards", "stage_rewards"))


def reconstruct_episode(jobs, dataset: dict, member: dict) -> tuple[dict, dict]:
    """Rebuild indices from frozen control data, never from evaluator predictions."""
    config, data, _, _ = reward_core(jobs.ui_config.offline_rl_root)
    raw = jobs._effective_config(dataset)
    artifacts = member["artifacts"]
    for artifact in artifacts.values():
        path = Path(artifact["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["size"]:
            raise ValueError(f"冻结源文件缺失或已改变：{path.name}")
    episode = data._load_run(Path(artifacts["manifest"]["path"]),
        config.LoadedConfig(jobs.base_config_path, raw),
        frozen_observations_sha256=artifacts["observations"]["sha256"])
    if episode is None or episode["run_id"] != member["run_id"]:
        raise ValueError("冻结轨迹身份或任务不匹配")
    for name, field in (("manifest", "source_manifest"), ("trajectory", "trajectory"),
                        ("observations", "observations")):
        path_field = "source_manifest" if name == "manifest" else f"{field}_path"
        if (Path(episode[path_field]).resolve() != Path(artifacts[name]["path"]).resolve()
                or episode[f"{field}_sha256"] != artifacts[name]["sha256"]):
            raise ValueError(f"冻结 {name} 与全局评价源不匹配")
    if (episode["recorded_action_count"] != member["end_step"]
            or int(episode.get("resume_step") or 0) != int(member.get("resume_step") or 0)):
        raise ValueError("冻结轨迹长度或接管边界已改变")
    episode["split"] = member["split"]
    data.add_episode_chunks(episode, raw["data"]["action_horizon"])
    header = {"schema_version": data.MANIFEST_SCHEMA_VERSION, "replay_policy": data.REPLAY_POLICY,
              **{key: raw["data"][key] for key in ("action_horizon", "action_dim", "proprio_dim",
                                                  "control_hz", "success_consecutive_steps")}}
    return episode, header


def direct_global_reward(jobs, dataset: dict, member: dict, source: str) -> tuple[dict, bytes]:
    """Global Stage labels and environment outcomes need no evaluation job."""
    _, _, rewards, stage = reward_core(jobs.ui_config.offline_rl_root)
    episode, header = reconstruct_episode(jobs, dataset, member)
    raw = jobs._load_base_config()["reward"]
    raw.update(source=source, rynnvalue=False)
    recipe = rewards.reward_derivation_config(raw)
    arrays = rewards._episode_timeline_arrays(episode)
    done = np.zeros(episode["recorded_action_count"], dtype=bool)
    if episode["terminal_step"] is not None:
        done[episode["terminal_step"]:] = True
    gamma, cumulative = recipe["gamma"], recipe["accumulate_primitive_steps"]
    annotation = None
    if source == "stage":
        path = Path(episode["trajectory_path"]).with_name("stage_annotation.json")
        if not path.is_file() or path.is_symlink():
            raise ValueError("缺少已保存的全局 Stage 关键帧")
        annotation = stage.validate_stage_annotation(json.loads(path.read_text()),
            run_id=episode["run_id"], trajectory_sha256=episode["trajectory_sha256"],
            done=arrays["environment_done"], success_consecutive_steps=header["success_consecutive_steps"])
        context = stage.stage_annotation_context(annotation, done=arrays["environment_done"],
            success_consecutive_steps=header["success_consecutive_steps"], exponent=recipe["stage_exponent"])
        scores = stage.stage_scores(context)
        final = [stage.stage_chunk_reward(scores, c["start"], c["length"], gamma, cumulative)
                 for c in episode["evaluation_chunks"]]
        arrays["stage_score"] = scores
        arrays["stage_chunk_reward"] = np.asarray(final, dtype=np.float32)
    else:
        final = [(rewards.sparse_primitive_return(done, c["start"], c["length"], gamma) if cumulative
                  else rewards.sparse_macro_reward(done, c["start"], c["length"]))
                 for c in episode["evaluation_chunks"]]
        arrays["sparse_reward"] = np.asarray(final, dtype=np.float32)
    arrays.update(boundary_steps=np.asarray(episode["reward_boundaries"], dtype=np.int64),
                  final_reward=np.asarray(final, dtype=np.float32), pbrs_chunk_reward=np.asarray(final, dtype=np.float32))
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    values = stream.getvalue()
    metadata = {"run_id": episode["run_id"], "source": source, "episode": episode, "prepared": header,
        "reward_config": recipe, "values_sha256": hashlib.sha256(values).hexdigest(),
        "trajectory_sha256": episode["trajectory_sha256"], "observations_sha256": episode["observations_sha256"],
        "entry": {**rewards._episode_reward_metadata(episode),
                  "derivation_implementation_sha256": rewards.reward_implementation_fingerprint(source)}}
    if annotation is not None:
        metadata["stage_annotation"] = annotation
        metadata["entry"]["stage_annotation_sha256"] = annotation["annotation_sha256"]
    return metadata, values


def validate_global_values(values: Path | bytes, episode: dict, source: str, offline_root: Path) -> None:
    with np.load(io.BytesIO(values) if isinstance(values, bytes) else values, allow_pickle=False) as arrays:
        boundaries = arrays["boundary_steps"]
        if not np.array_equal(boundaries, episode["reward_boundaries"]):
            raise ValueError("全局评价时间点与当前数据集 chunk 边界不一致，请使用相同成功阈值；不会自动重新评价")
        final = arrays["final_reward"] if "final_reward" in arrays else arrays["pbrs_chunk_reward"]
        if final.shape != (len(boundaries) - 1,) or not np.isfinite(final).all():
            raise ValueError("全局奖励数组不完整或包含无效数值")
        if source == "stage" and (arrays["stage_score"].shape != (episode["recorded_action_count"] + 1,)
                                   or not np.isfinite(arrays["stage_score"]).all()):
            raise ValueError("全局 Stage 奖励未覆盖完整轨迹")
        if source == "rynnvalue":
            _, _, rewards, _ = reward_core(offline_root)
            rewards.validate_official_outputs({name: arrays[name] for name in rewards.OFFICIAL_OUTPUT_KEYS},
                                                len(boundaries))
