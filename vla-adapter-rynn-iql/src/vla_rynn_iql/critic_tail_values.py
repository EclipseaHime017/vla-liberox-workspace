"""Read-only endpoint V estimates from an existing diagnostic's frozen value network."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .base_models import model_contract
from .data import load_manifest, replay_chunks
from .frame_cache import FrameCache
from .io import sha256_file, stable_hash
from .iql import ValueNetwork
from .models import replay_stats
from .replay import ActionDataset
from .rewards import reward_derivation_config
from .storage_paths import storage_path, with_dataset_lease


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((str(array.dtype), array.shape)).encode() + array.tobytes()).hexdigest()


@with_dataset_lease
def export_tail_values(config, result: Path, output: Path, *, critic_sha256: str,
                       split: str = "all", device: str = "cpu"):
    if split not in {"train", "validation", "all"}:
        raise ValueError("Invalid tail split")
    result, output = Path(result), Path(output)
    checkpoint_path, summary_path, csv_path = (result / n for n in ("critic.pt", "summary.json", "qv.csv"))
    if output.exists():
        raise FileExistsError("Tail output already exists; use a new file")
    if len(critic_sha256) != 64 or sha256_file(checkpoint_path) != critic_sha256:
        raise ValueError("Frozen critic SHA256 mismatch")
    manifest_path = Path(config.section("paths")["work_dir"]) / "dataset_manifest.json"
    reward_path = Path(config.section("reward")["manifest_path"])
    paths = [checkpoint_path, summary_path, csv_path, manifest_path, reward_path]
    before = {str(p): sha256_file(p) for p in paths}
    manifest = load_manifest(config)
    diagnostic = json.loads(summary_path.read_text())
    recipe = reward_derivation_config(config.section("reward"))
    if (diagnostic.get("kind") != "iql_critic_diagnostic"
            or diagnostic["training_config_sha256"] != config.digest
            or diagnostic["dataset_sha256"] != manifest["dataset_sha256"]
            or diagnostic["reward_config"] != recipe
            or diagnostic["reward_manifest_sha256"] != before[str(reward_path)]
            or config.section("reward")["manifest_sha256"] != before[str(reward_path)]
            or diagnostic["model_config"] != config.section("model")
            or diagnostic["model_contract"] != model_contract(config.section("model"))):
        raise ValueError("Frozen diagnostic configuration/dataset/reward/model identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "iql_critic_diagnostic"
            or checkpoint["reward_config"] != recipe or checkpoint["iql_config"] != config.section("iql")
            or checkpoint["proprio_dim"] != manifest["proprio_dim"]
            or checkpoint["action_dim"] != manifest["action_dim"]
            or checkpoint["horizon"] != diagnostic["critic_horizon"]
            or checkpoint["seed"] != diagnostic["seed"]):
        raise ValueError("Frozen critic metadata mismatch")
    with csv_path.open(newline="") as stream:
        references = {}
        for row in csv.DictReader(stream):
            key = row["split"], row["run_id"], int(row["start"]), int(row["end"])
            if key in references:
                raise ValueError("Duplicate diagnostic row")
            references[key] = float(row["v"])
    failures = {e["run_id"]: e for e in manifest["episodes"]
                if e.get("terminal_step") is None and (split == "all" or e["split"] == split)}
    if len({e["run_id"] for e in manifest["episodes"]}) != len(manifest["episodes"]):
        raise ValueError("Duplicate dataset run ID")
    for episode in failures.values():
        for name in ("trajectory", "observations"):
            path = storage_path(episode[f"{name}_path"])
            actual = sha256_file(path)
            if actual != episode[f"{name}_sha256"]:
                raise ValueError(f"Changed {name}: {episode['run_id']}")
            before[str(path)] = actual
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (target.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested tail inference device is unavailable")
    action_stats, proprio_stats = replay_stats(config.section("model"))
    rows = []
    # Only V is instantiated; no actor, optimizer, updates, or reward materialization.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        value = ValueNetwork(proprio_dim=manifest["proprio_dim"])
        value.load_state_dict({k.removeprefix("value."): v for k, v in checkpoint["critic"]["model"].items()
                               if k.startswith("value.")}, strict=True)
        value.to(target).eval().requires_grad_(False)
        for label in ("train", "validation"):
            if split not in ("all", label) or not any(e["split"] == label for e in failures.values()):
                continue
            replay = ActionDataset(config, action_stats, proprio_stats, label)
            replay.image_size = int(config.section("iql")["critic_image_size"])
            replay._frames = FrameCache(None)  # Decode in memory; never write the existing dataset/cache.
            for index, (episode, chunk_index, _) in enumerate(replay.items):
                rid = episode["run_id"]
                chunk = replay.chunks[rid][chunk_index]
                if rid not in failures or chunk["end"] != episode["recorded_action_count"]:
                    continue
                if chunk != replay_chunks(episode)[-1] or chunk["start"] >= chunk["end"]:
                    raise ValueError(f"Not the recorded final transition: {rid}")
                replay._frame_indices[rid] = {chunk["start"], chunk["end"]}
                sample = replay._sample(index, transitions=True)
                current = float(value(sample["pixels"].unsqueeze(0).to(target),
                                      sample["proprio"].unsqueeze(0).to(target)).item())
                reference = references.get((label, rid, chunk["start"], chunk["end"]))
                if reference is None or not np.isclose(current, reference, rtol=1e-4, atol=1e-5):
                    raise ValueError(f"Frozen V/current preprocessing disagrees with original qv.csv: {rid}")
                tail = float(value(sample["next_pixels"].unsqueeze(0).to(target),
                                   sample["next_proprio"].unsqueeze(0).to(target)).item())
                if not np.isfinite(tail):
                    raise ValueError(f"Nonfinite endpoint V: {rid}")
                rows.append({"run_id": rid, "split": label, "root_run_id": episode["root_run_id"],
                    "parent_run_id": episode.get("parent_run_id"), "observation_step": chunk["end"],
                    "last_chunk_start": chunk["start"], "tail_v": tail,
                    "trajectory_sha256": episode["trajectory_sha256"],
                    "observations_sha256": episode["observations_sha256"],
                    "next_pixels_sha256": tensor_digest(sample["next_pixels"]),
                    "next_proprio_sha256": tensor_digest(sample["next_proprio"]),
                    "reference_start_v": reference, "recomputed_start_v": current})
    if {r["run_id"] for r in rows} != set(failures) or len(rows) != len(failures):
        raise ValueError("Replay deduplication omitted or duplicated a failure endpoint")
    if before != {name: sha256_file(Path(name)) for name in before}:
        raise ValueError("Frozen input changed during tail export")
    payload = {"schema_version": 1, "kind": "frozen_critic_tail_values", "split": split,
        "dataset_sha256": manifest["dataset_sha256"], "dataset_manifest_sha256": before[str(manifest_path)],
        "reward_manifest_sha256": before[str(reward_path)], "critic_sha256": critic_sha256,
        "training_config_sha256": config.digest, "summary_sha256": before[str(summary_path)],
        "gamma": recipe["gamma"], "primitive_discount": recipe["accumulate_primitive_steps"],
        "model_contract": diagnostic["model_contract"], "chunking": manifest["chunking"],
        "normalization_sha256": stable_hash({"action": action_stats, "proprio": proprio_stats}),
        "critic_image_size": replay.image_size if rows else config.section("iql")["critic_image_size"],
        "input_sha256": before, "rows": rows,
        "caveat": "Frozen learned V, possibly biased by terminal truncation; not an infinite-return truth or proof of W5."}
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload
