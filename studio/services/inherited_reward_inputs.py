"""CPU-only adapters for global labels whose old Prepare directory is gone."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
from pathlib import Path

from studio.storage.paths import storage_path

import numpy as np


CORE_MODULES = {
    "config": "core.config", "config_sources": "core.config_sources", "data": "core.data",
    "execution": "core.execution", "io": "core.io", "run_layout": "core.run_layout",
    "models": "core.model_config", "base_models": "core.registry", "model_assets": "core.model_assets",
    "model_artifacts": "core.artifacts", "pi05_assets": "core.pi05_assets",
    "image_transforms": "core.image_transforms", "model_storage": "core.model_storage",
    "checkpoint_assets": "core.checkpoint_assets", "training": "train.trainer",
    "vla_adapter": "train.models.vla_adapter", "pi05": "train.models.pi05",
    "rewards": "rewards.manifest", "fusion_rewards": "rewards.fusion",
}


def core_module(part: str):
    """Torch-free core and reward modules, addressed by their short historical names."""
    return importlib.import_module(CORE_MODULES[part])


def reconstruct_episode(jobs, dataset: dict, member: dict) -> tuple[dict, dict]:
    """Rebuild indices from frozen control data, never from evaluator predictions."""
    from core import config, data
    raw = jobs._effective_config(dataset)
    artifacts = member["artifacts"]
    for artifact in artifacts.values():
        path = storage_path(artifact["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["size"]:
            raise ValueError(f"冻结源文件缺失或已改变：{path.name}")
    episode = data._load_run(storage_path(artifacts["manifest"]["path"]),
        config.LoadedConfig(jobs.base_config_path, raw),
        frozen_observations_sha256=artifacts["observations"]["sha256"])
    if episode is None or episode["run_id"] != member["run_id"]:
        raise ValueError("冻结轨迹不是有效的训练轨迹（身份或任务不匹配、未完成，或为策略相机消融运行）")
    for name, field in (("manifest", "source_manifest"), ("trajectory", "trajectory"),
                        ("observations", "observations")):
        path_field = "source_manifest" if name == "manifest" else f"{field}_path"
        if (storage_path(episode[path_field]).resolve() != storage_path(artifacts[name]["path"]).resolve()
                or episode[f"{field}_sha256"] != artifacts[name]["sha256"]):
            raise ValueError(f"冻结 {name} 与全局评价源不匹配")
    if (episode["recorded_action_count"] != member["end_step"]
            or int(episode.get("resume_step") or 0) != int(member.get("resume_step") or 0)):
        raise ValueError("冻结轨迹长度或接管边界已改变")
    episode["split"] = member["split"]
    data.add_episode_chunks(episode)
    header = {"schema_version": data.MANIFEST_SCHEMA_VERSION, "replay_policy": data.REPLAY_POLICY,
              "chunking": data.CHUNKING_RULE,
              **{key: raw["data"][key] for key in ("action_dim", "proprio_dim",
                                                  "control_hz", "success_consecutive_steps")}}
    return episode, header


def direct_global_reward(jobs, dataset: dict, member: dict, source: str, *,
                         use_dataset_labels: bool = True) -> tuple[dict, bytes]:
    """Global Stage labels and environment outcomes need no evaluation job."""
    from rewards import manifest as rewards
    from rewards.fusion import step_reward_arrays
    from rewards.stage import keyframes, reward as stage_reward
    episode, header = reconstruct_episode(jobs, dataset, member)
    raw = jobs._load_base_config()["reward"]
    raw.update(source=source, rynnvalue=False)
    recipe = rewards.reward_derivation_config(raw)
    arrays = rewards._episode_timeline_arrays(episode)
    annotation = None
    if source == "stage":
        if use_dataset_labels:
            with jobs.datasets.lock:
                _, _, label = jobs.datasets.stage_labels.member(dataset["id"], member["run_id"])
        else:
            label = jobs.datasets.stage_labels._global(member)
        if label.get("annotation") is None:
            if label.get("error"):
                raise ValueError(label["error"])
            raise FileNotFoundError("数据集与全局均缺少已保存的 Stage 关键帧标注"
                if use_dataset_labels else "缺少已保存的全局 Stage 关键帧标注")
        annotation = keyframes.validate_stage_annotation(label["annotation"],
            run_id=episode["run_id"], trajectory_sha256=episode["trajectory_sha256"],
            done=arrays["environment_done"], success_consecutive_steps=header["success_consecutive_steps"])
        context = stage_reward.stage_annotation_context(annotation, done=arrays["environment_done"],
            success_consecutive_steps=header["success_consecutive_steps"], exponent=recipe["stage_exponent"])
        arrays["stage_score"] = stage_reward.stage_scores(context)
    arrays.update(step_reward_arrays(episode, recipe, arrays))
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
    count = int(episode["recorded_action_count"])
    with np.load(io.BytesIO(values) if isinstance(values, bytes) else values, allow_pickle=False) as arrays:
        final = arrays["final_reward"]
        if final.shape != (count,) or not np.isfinite(final).all():
            raise ValueError("全局奖励未逐步覆盖完整轨迹或包含无效数值，请在数据集配置中重新生成评价／奖励；不会自动重跑模型")
        if source == "stage" and (arrays["stage_score"].shape != (episode["recorded_action_count"] + 1,)
                                   or not np.isfinite(arrays["stage_score"]).all()):
            raise ValueError("全局 Stage 奖励未覆盖完整轨迹")
        if source == "rynnvalue":
            from rewards.rynnvalue.outputs import OFFICIAL_OUTPUT_KEYS, validate_official_outputs
            validate_official_outputs({name: arrays[name] for name in OFFICIAL_OUTPUT_KEYS}, count + 1)
