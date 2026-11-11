"""Read-only final IQL critic census over the exact deduplicated training replay."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .base_models import model_contract
from .config import LoadedConfig, validate_train_config
from .config_sources import read_source
from .data import iter_unique_replay_chunks, replay_chunks, training_replay_policy
from .io import sha256_file, stable_hash
from .iql import QNetwork, ValueNetwork
from .model_artifacts import validate_parent
from .models import model_signature
from .rewards import reward_manifest_digest
from .server_replay import validate_cache


def _json(path):
    return json.loads(Path(path).read_text())


def _hashes(paths):
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular immutable input: {path}")
        result[str(path)] = sha256_file(path)
    return result


def _groups(episode, chunk):
    terminal = episode.get("terminal_step")
    result = []
    if episode.get("kind") != "branch" and terminal is None:
        result.append("failed_parent")
    if chunk["action_source"] == "human" and terminal is not None:
        result.append("human_post_success" if int(chunk["start"]) >= int(terminal) + 1
                      else "human_pre_success")
    return result


def _statistics(rows):
    if not rows:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None,
                "cap_fraction": None}
    weights = np.asarray([row["weight"] for row in rows], dtype=np.float64)
    return {"count": len(rows), "mean": float(weights.mean()),
            **{name: float(np.quantile(weights, q)) for name, q in
               (("p05", .05), ("p50", .5), ("p95", .95))},
            "cap_fraction": float(np.mean([row["at_cap"] for row in rows]))}


def export_critic(checkpoint: Path, prepared_manifest: Path, reward_manifest: Path,
                  replay_cache: Path, output: Path, *, expected_trainer_sha256: str,
                  expected_beta: float | None = None,
                  device: str = "cpu", batch_size: int = 32) -> dict:
    """Inspect a trusted torch checkpoint; never call training/actor/optimizer APIs.

    trainer.pt is a trusted local pickle, authenticated against an independently
    supplied SHA before deserializing. mmap avoids materializing actor optimizer
    tensor storage. Only q1/q2/value tensors are copied into inference modules.
    Existing inputs and output are never overwritten, nor is replay rebuilt.
    """
    checkpoint, replay_cache = Path(checkpoint).resolve(), Path(replay_cache).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = [checkpoint / name for name in ("checkpoint.json", "effective_config.yaml", "trainer.pt")]
    files += [Path(prepared_manifest).resolve(), Path(reward_manifest).resolve(), replay_cache / "cache.json"]
    before = _hashes(files)
    if before[str(checkpoint / "trainer.pt")] != expected_trainer_sha256:
        raise ValueError("trainer.pt SHA mismatch")
    metadata, manifest, reward = _json(files[0]), _json(files[3]), _json(files[4])
    raw = read_source(files[1])
    if LoadedConfig(files[1], raw).digest != metadata.get("config_sha256"):
        raise ValueError("Saved configuration SHA mismatch")
    config = validate_train_config(copy.deepcopy(raw), files[1])
    step = metadata.get("step")
    if (metadata.get("schema_version") != 2 or type(step) is not int or step < 1
            or checkpoint.name != f"step_{step:08d}"
            or step != config.section("training")["train_steps"]):
        raise ValueError("R5 requires the final schema-2 checkpoint, not an intermediate step")
    if metadata.get("algorithm") != "iql" or config.section("training")["method"] != "iql":
        raise ValueError("R5 requires IQL")
    if (step != 10000 or config.section("reward")["gamma"] != .99
            or config.section("reward")["source"] != "rynnvalue"
            or config.section("data").get("include_post_success", True)
            or not config.section("data").get("bootstrap_failed_end", False)
            or not config.section("reward").get("zero_success_potential", False)):
        raise ValueError("R5 requires final T2-D: 10000 steps, Rynn gamma=.99, no post-success, both W5 switches enabled")
    contract = model_contract(config.section("model"))
    parent = metadata.get("model_parent")
    validate_parent(parent, config.section("model"))
    expected = {
        "model_config": model_signature(config.raw), "model_contract": contract,
        "base_checkpoint": config.section("model")["base_checkpoint"], "stats_key": parent["stats_key"],
        "dataset_sha256": manifest["dataset_sha256"], "reward_sha256": reward_manifest_digest(reward),
        "training_layout": {"chunking": manifest.get("chunking"),
                            "actor_horizon": contract["io"]["replay_horizon"],
                            "transition_horizon": manifest["action_horizon"]},
        "replay_policy": training_replay_policy(config.section("data").get("include_post_success", True)),
        "bootstrap_failed_end": config.section("data").get("bootstrap_failed_end", False),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Checkpoint {name} mismatch")
    if reward.get("complete") is False or reward["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Reward manifest incomplete or from another dataset")
    cache = validate_cache(replay_cache, hashes=True)
    identity = cache["identity"]
    if cache["fingerprint"] != stable_hash(identity):
        raise ValueError("Replay fingerprint mismatch")
    cache_expected = {"dataset": manifest["dataset_sha256"], "reward": expected["reward_sha256"],
                      "model_contract": contract, "transitions": True,
                      "include_post_success": config.section("data").get("include_post_success", True),
                      "bootstrap_failed_end": expected["bootstrap_failed_end"],
                      "image_size": config.section("iql")["critic_image_size"]}
    for name, value in cache_expected.items():
        if identity.get(name) != value:
            raise ValueError(f"Replay identity {name} mismatch")
    for name, digest in identity["implementation"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Replay implementation mismatch: {name}")
    selected = list(iter_unique_replay_chunks(manifest["episodes"], split="train",
                    include_post_success=identity["include_post_success"]))
    if not selected or len(selected) != cache["count"]:
        raise ValueError("Replay is not the complete deduplicated train split")
    chunks = [(episode, index, replay_chunks(episode)[index]) for episode, index in selected]
    for record, (episode, _, chunk) in zip(cache["records"], chunks):
        for name, value in {"run_id": episode["run_id"], "start": chunk["start"],
                            "action_source": chunk["action_source"]}.items():
            if record.get(name) != value:
                raise ValueError(f"Replay train ordering/member mismatch: {name}")
    # No cache builder, VLM, model backend, actor or optimizer is invoked.
    names = ("pixels", "proprio", "actions", "action_mask", "chunk_length")
    arrays = {name: np.load(replay_cache / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
    if not np.array_equal(arrays["chunk_length"], [int(c["length"]) for _, _, c in chunks]):
        raise ValueError("Replay chunk length mismatch")
    beta = float(config.section("iql")["beta"])
    cap = float(config.section("iql")["max_advantage_weight"])
    if not math.isfinite(beta) or beta <= 0 or cap != 20:
        raise ValueError("R5 T2-D requires a finite positive saved actual beta and cap=20")
    if expected_beta is not None and (
            isinstance(expected_beta, bool) or not math.isfinite(expected_beta)
            or expected_beta <= 0 or beta != expected_beta):
        raise ValueError(f"Saved actual beta {beta} differs from independently expected beta {expected_beta}")
    trainer = torch.load(files[2], map_location="cpu", weights_only=False, mmap=True)
    if (trainer.get("schema_version"), trainer.get("step"), trainer.get("algorithm")) != (2, step, "iql"):
        raise ValueError("trainer.pt metadata mismatch")
    state = trainer["algorithm_state"]["model"]
    target = torch.device(device)
    devices = [target.index or 0] if target.type == "cuda" else []
    rows = []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        dims = (manifest["action_horizon"], manifest["action_dim"], manifest["proprio_dim"])
        modules = {"q1": QNetwork(*dims), "q2": QNetwork(*dims), "value": ValueNetwork(dims[2])}
        for name, module in modules.items():
            module.load_state_dict({key[len(name) + 1:]: value for key, value in state.items()
                                    if key.startswith(name + ".")}, strict=True)
            module.to(target).eval().requires_grad_(False)
        for start in range(0, len(chunks), batch_size):
            batch = {name: torch.from_numpy(np.array(array[start:start + batch_size], copy=True)).to(target)
                     for name, array in arrays.items()}
            qargs = [batch[name] for name in ("pixels", "proprio", "actions", "action_mask")]
            q1, q2 = modules["q1"](*qargs), modules["q2"](*qargs)
            value = modules["value"](batch["pixels"], batch["proprio"])
            advantage = torch.minimum(q1, q2) - value
            # Exactly the training float32 exp/clamp, not normalized weights.
            weights = torch.exp(beta * advantage).clamp(max=cap)
            if not all(torch.isfinite(v).all() for v in (q1, q2, value, advantage, weights)):
                raise ValueError("Non-finite frozen critic output")
            for offset in range(len(value)):
                episode, index, chunk = chunks[start + offset]
                a = float(advantage[offset])
                rows.append({"run_id": episode["run_id"], "root_run_id": episode.get("root_run_id"),
                    "parent_run_id": episode.get("parent_run_id"), "split": "train", "kind": episode["kind"],
                    "chunk_index": index, "start": int(chunk["start"]), "end": int(chunk["end"]),
                    "action_source": chunk["action_source"], "groups": _groups(episode, chunk),
                    "q1": float(q1[offset]), "q2": float(q2[offset]), "q": float(torch.minimum(q1, q2)[offset]),
                    "v": float(value[offset]), "advantage": a, "weight": float(weights[offset]),
                    "at_cap": beta * a >= math.log(cap)})
    if _hashes(files) != before or validate_cache(replay_cache, hashes=True) != cache:
        raise ValueError("Input changed during critic census")
    group_stats = {name: _statistics([row for row in rows if name in row["groups"]]) for name in
                   ("failed_parent", "human_pre_success", "human_post_success")}
    failed, human = group_stats["failed_parent"], group_stats["human_pre_success"]
    complete = failed["count"] > 0 and human["count"] > 0
    result = {"schema_version": 1, "kind": "final_iql_train_critic_census", "source_sha256": before,
              "checkpoint_metadata": metadata, "config_sha256": config.digest,
              "seed": config.section("training")["seed"], "step": step, "split": "train",
              "gamma": config.section("reward")["gamma"], "beta": beta, "max_weight": cap,
              "expected_beta": expected_beta,
              "beta_source": "checkpoint/effective_config.yaml:iql.beta",
              "source": "T2-D", "deduplicated": True, "checkpoint_step": step,
              "cache_identity": identity, "cache_fingerprint": cache["fingerprint"],
              "normalization_sha256": stable_hash({key: identity[key] for key in ("action_stats", "proprio_stats")}),
              "model_parent_sha256": parent["sha256"],
              "tool_sha256": sha256_file(Path(__file__)),
              "cache_arrays": cache["arrays"], "count": len(rows), "statistics": _statistics(rows),
              "groups": group_stats, "d2_second_condition_evaluable": complete,
              "d2_second_condition": failed["mean"] <= .8 * human["mean"] if complete else None,
              "note": "One final seed critic; not P4, not a pooled-seed D2 decision.", "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result
