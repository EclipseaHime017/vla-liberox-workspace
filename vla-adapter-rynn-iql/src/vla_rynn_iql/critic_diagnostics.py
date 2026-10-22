"""Actor-free IQL critic fitting and read-only rollout diagnostics."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from .base_models import model_contract
from .config import LoadedConfig, reward_source
from .data import load_manifest, replay_end_step
from .iql import PixelIQL, chunk_bellman_target
from .io import atomic_json, sha256_file
from .models import model_config, replay_stats
from .replay import ActionDataset, ReplayDataset
from .rewards import load_pinned_reward_index, reward_derivation_config, TRAINING_REDUCTION_CONFIG_KEYS
from .storage_paths import storage_path, with_dataset_lease


SCIENTIFIC_CAVEATS = [
    "IQL V is an expectile of dataset-action Q values, not the selected policy's V^pi.",
    "Q and V are learned estimates, not ideal or ground-truth values; fitting does not prove policy quality.",
    "Training curves are in-sample; validation and rollout curves are separate probes, not certified OPE.",
    "Observed returns stop at the saved replay endpoint, including time limits; they are finite-trajectory references.",
    "Rollout probes have no reward labels: no TD target or observed return is generated for them.",
    "Probe actions and chunk lengths outside the training support can have unreliable Q estimates.",
]
VALUE_COLUMNS = ("q1", "q2", "min_q", "v", "advantage")
REWARD_COLUMNS = ("reward", "bootstrap_mask", "bellman_target", "td_error", "observed_return")
CSV_COLUMNS = ("split", "run_id", "start", "end", "chunk_length", "action_source", *VALUE_COLUMNS, *REWARD_COLUMNS)


def _validate_sources(manifest: dict) -> dict[str, str]:
    hashes = {}
    for episode in manifest["episodes"]:
        for key in ("trajectory", "observations"):
            path = storage_path(episode[f"{key}_path"])
            expected = episode[f"{key}_sha256"]
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Prepared {key} changed for run {episode['run_id']}: {path}")
            hashes[str(path)] = expected
    return hashes


def _existing_rewards(config: LoadedConfig) -> dict:
    """Reject changed reductions before the regular loader can materialize them."""
    reward = config.section("reward")
    if not reward.get("manifest_path") or not reward.get("manifest_sha256"):
        raise ValueError("Critic diagnostics require an existing pinned dataset reward version")
    path = Path(reward["manifest_path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != reward["manifest_sha256"]:
        raise ValueError("Pinned reward manifest is missing or changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recipe = reward_derivation_config(reward)
    if payload.get("reward_config") != recipe:
        raise ValueError("Critic diagnostics must use the saved reward configuration without recomputation")
    if payload.get("binding_kind") == "global_trajectory_snapshots":
        for entry in payload.get("episodes", []):
            stored = entry.get("saved_reward_config", {})
            if any(stored.get(key) != recipe.get(key) for key in TRAINING_REDUCTION_CONFIG_KEYS):
                raise ValueError("Saved global rewards have different discounts or reductions; cannot reuse unchanged")
    return load_pinned_reward_index(config)


def _batch(samples: list[dict], horizon: int, device: torch.device) -> dict:
    normalized = []
    for sample in samples:
        size = sample["actions"].shape[0]
        if size > horizon:
            raise ValueError("Critic horizon must cover every recorded action chunk")
        sample = dict(sample)
        sample["actions"] = F.pad(sample["actions"], (0, 0, 0, horizon - size))
        sample["action_mask"] = F.pad(sample["action_mask"], (0, horizon - size))
        normalized.append({key: value for key, value in sample.items() if torch.is_tensor(value)})
    return {key: value.to(device) for key, value in default_collate(normalized).items()}


def _finite_returns(dataset: ReplayDataset, discount: float, cumulative: bool) -> dict:
    result = {}
    episodes = {episode["run_id"]: (episode, annotation) for episode, _, annotation in dataset.items}
    for run_id, (episode, annotation) in episodes.items():
        with np.load(annotation, allow_pickle=False) as source:
            rewards = source["final_reward"] if "final_reward" in source else source["pbrs_chunk_reward"]
        end = replay_end_step(episode, dataset.include_post_success)
        total = 0.0
        for index in reversed(range(len(dataset.chunks[run_id]))):
            chunk = dataset.chunks[run_id][index]
            if int(chunk["end"]) > end:
                continue
            factor = discount ** (int(chunk["length"]) if cumulative else 1)
            total = float(rewards[index]) + factor * total
            result[(run_id, int(chunk["start"]))] = total
    return result


@torch.no_grad()
def _evaluate(agent: PixelIQL, dataset: ActionDataset, split: str, horizon: int,
              batch_size: int, device: torch.device,
              stop_requested: Callable[[], bool] | None = None) -> list[dict]:
    has_rewards = isinstance(dataset, ReplayDataset)
    returns = _finite_returns(dataset, agent.discount, agent.accumulate_primitive_steps) if has_rewards else {}
    order = sorted(range(len(dataset)), key=lambda index: (
        dataset.items[index][0]["run_id"],
        int(dataset.chunks[dataset.items[index][0]["run_id"]][dataset.items[index][1]]["start"])))
    rows = []
    for offset in range(0, len(order), batch_size):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Critic diagnostics canceled at an evaluation batch boundary")
        samples = [(dataset[index] if has_rewards else dataset._sample(index, transitions=True))
                   for index in order[offset:offset + batch_size]]
        batch = _batch(samples, horizon, device)
        q1 = agent.q1(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q2 = agent.q2(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        value = agent.value(batch["pixels"], batch["proprio"])
        minimum = torch.minimum(q1, q2)
        values = torch.stack((q1, q2, minimum, value, minimum - value), dim=1).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Critic produced non-finite diagnostic values")
        targets = None
        if has_rewards:
            targets = chunk_bellman_target(batch["reward"], agent.value(batch["next_pixels"], batch["next_proprio"]),
                batch["chunk_length"], batch["bootstrap_mask"], agent.discount,
                agent.accumulate_primitive_steps).cpu().numpy()
        for index, sample in enumerate(samples):
            start, length = int(sample["start"]), int(sample["chunk_length"])
            row = {"split": split, "run_id": sample["run_id"], "start": start, "end": start + length,
                   "chunk_length": length, "action_source": sample["action_source"],
                   **dict(zip(VALUE_COLUMNS, map(float, values[index]))),
                   **dict.fromkeys(REWARD_COLUMNS)}
            if has_rewards:
                row.update(reward=float(sample["reward"]), bootstrap_mask=float(sample["bootstrap_mask"]),
                    bellman_target=float(targets[index]), td_error=float(minimum[index]) - float(targets[index]),
                    observed_return=returns[(sample["run_id"], start)])
            rows.append(row)
    return rows


def _plots(output: Path, rows: list[dict], metrics: list[dict]) -> None:
    # Agg figures do not require a GUI and do not mutate pyplot's global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["run_id"])].append(row)
    figure = Figure(figsize=(15, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3, squeeze=False)
    for line, split in enumerate(("train", "validation", "rollout")):
        for column, key in enumerate(("min_q", "v", "advantage")):
            axis = axes[line, column]
            for (group, run_id), series in grouped.items():
                if group != split:
                    continue
                axis.plot([row["start"] for row in series], [row[key] for row in series], label=run_id)
                if key != "advantage" and series[0]["observed_return"] is not None:
                    axis.plot([row["start"] for row in series], [row["observed_return"] for row in series],
                              linestyle=":", alpha=.5, label=f"{run_id} observed return")
            axis.set(title=f"{split}: {key}", xlabel="Observation step", ylabel="Estimated value")
            axis.grid(alpha=.2)
            if axis.lines and len(axis.lines) <= 8:
                axis.legend(fontsize=7)
    figure.savefig(output / "qv.png", dpi=140)
    figure.clear()
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, key in zip(axes, ("q_loss", "value_loss")):
        axis.plot([item["train_step"] for item in metrics], [item[key] for item in metrics])
        axis.set(title=key, xlabel="Critic optimizer step", ylabel="Loss")
        axis.grid(alpha=.2)
    figure.savefig(output / "training.png", dpi=140)
    figure.clear()


@with_dataset_lease
def run_critic_diagnostic(training_config: LoadedConfig, probe_config: LoadedConfig, output: Path,
                          *, train_steps: int, batch_size: int, seed: int, device: str,
                          progress: Callable[[dict], None] | None = None,
                          stop_requested: Callable[[], bool] | None = None) -> dict:
    """Fit only Q/V on saved dataset rewards; score a reward-free rollout probe."""
    @with_dataset_lease
    def probe_locked(config):
        return _run(training_config, config, output, train_steps=train_steps, batch_size=batch_size,
                    seed=seed, device=device, progress=progress, stop_requested=stop_requested)
    return probe_locked(probe_config)


def _run(training_config, probe_config, output, *, train_steps, batch_size, seed, device, progress, stop_requested):
    for name, value, minimum in (("train_steps", train_steps, 1), ("batch_size", batch_size, 1), ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (target.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Critic device is unavailable: {device}")
    train_manifest, probe_manifest = load_manifest(training_config), load_manifest(probe_config)
    tasks = {episode["task_id"] for manifest in (train_manifest, probe_manifest)
             for episode in manifest["episodes"]}
    if len(tasks) != 1:
        raise ValueError("The pixel critic requires training data and rollout probes from one task")
    train_contract = model_contract(training_config.section("model"))
    if train_contract != model_contract(probe_config.section("model")):
        raise ValueError("Training replay and rollout probe must use the same selected model contract")
    train_model = model_config(training_config.raw, partial=True)
    probe_model = model_config(probe_config.raw, partial=True)
    if any(train_model[key] != probe_model[key] for key in
           ("family", "base_id", "base_checkpoint", "stats_key", "base_revision")):
        raise ValueError("Training replay and rollout probe must use the same selected model normalization identity")
    for manifest in (train_manifest, probe_manifest):
        for key in ("action_dim", "proprio_dim"):
            if manifest[key] != train_contract["io"][key]:
                raise ValueError(f"Diagnostic prepared {key} does not match the selected model")
    source_hashes = {**_validate_sources(train_manifest), **_validate_sources(probe_manifest)}
    reward_index = _existing_rewards(training_config)
    reward_hashes = {str(Path(training_config.section("reward")["manifest_path"])):
                     training_config.section("reward")["manifest_sha256"]}
    reward_hashes.update({str(storage_path(item["reward_path"])): item["reward_sha256"]
                          for item in reward_index["episodes"]})
    action_stats, proprio_stats = replay_stats(training_config.section("model"))
    train = ReplayDataset(training_config, action_stats, proprio_stats, "train", reward_index=reward_index)
    validation = ReplayDataset(training_config, action_stats, proprio_stats, "validation", reward_index=reward_index)
    rollout = ActionDataset(probe_config, action_stats, proprio_stats, "train")
    if not len(rollout):
        raise ValueError("Rollout probe contains no action chunks")
    horizon = max(train_manifest["action_horizon"], probe_manifest["action_horizon"])
    cfg, recipe = training_config.section("iql"), training_config.section("reward")
    derivation = reward_derivation_config(recipe)
    rollout.image_size = int(cfg["critic_image_size"])
    output = Path(output)
    artifacts = ("summary.json", "qv.csv", "qv.npz", "qv.png", "training.png", "metrics.jsonl", "critic.pt")
    if any((output / name).exists() for name in artifacts):
        raise ValueError("Diagnostic result artifacts already exist; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if target.type == "cuda":
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
        agent = PixelIQL(horizon=horizon, action_dim=train_manifest["action_dim"],
            proprio_dim=train_manifest["proprio_dim"], discount=derivation["gamma"], expectile=cfg["expectile"],
            tau=cfg["target_tau"], accumulate_primitive_steps=derivation["accumulate_primitive_steps"],
            **{key: cfg[key] for key in ("critic_lr", "value_lr", "critic_optimizer", "critic_weight_decay",
                "value_optimizer", "value_weight_decay", "critic_max_grad_norm", "value_max_grad_norm")}).to(target)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        metrics = []
        agent.train()
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(1, train_steps + 1):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("Critic diagnostics canceled at a batch boundary")
                indices = torch.randint(len(train), (batch_size,), generator=generator).tolist()
                values = vars(agent.update(_batch([train[index] for index in indices], horizon, target)))
                if not all(np.isfinite(value) for value in values.values()):
                    raise ValueError(f"Non-finite critic metrics at optimizer step {step}")
                item = {"train_step": step, "train_steps": train_steps, **values}
                metrics.append(item)
                stream.write(json.dumps(item, allow_nan=False) + "\n")
                stream.flush()
                if progress is not None:
                    progress(item)
        agent.eval()
        rows = sum((_evaluate(agent, dataset, split, horizon, batch_size, target, stop_requested)
                    for split, dataset in (("train", train), ("validation", validation), ("rollout", rollout))), [])
        torch.save({"schema_version": 1, "kind": "iql_critic_diagnostic", "critic": agent.checkpoint(),
                    "horizon": horizon, "action_dim": train_manifest["action_dim"],
                    "proprio_dim": train_manifest["proprio_dim"], "seed": seed,
                    "iql_config": cfg, "reward_config": derivation}, output / "critic.pt")
    with (output / "qv.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "qv.npz", **{
        key: np.asarray([np.nan if row[key] is None else row[key] for row in rows]) for key in CSV_COLUMNS})
    _plots(output, rows, metrics)
    for name, digest in {**source_hashes, **reward_hashes}.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Diagnostic input changed during execution: {name}")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "rollout")}
    training_episodes = {episode["run_id"]: episode for episode, _, _ in train.items}
    training_hashes = {episode["trajectory_sha256"] for episode in training_episodes.values()}
    overlapping = [episode["run_id"] for episode in probe_manifest["episodes"]
                   if episode["run_id"] in training_episodes or episode["trajectory_sha256"] in training_hashes]
    lengths = {int(train.chunks[episode["run_id"]][index]["length"]) for episode, index, _ in train.items}
    summary = {"schema_version": 1, "kind": "iql_critic_diagnostic", "actor_loaded_for_training": False,
        "actor_updated": False, "reward_generated": False, "train_steps": train_steps,
        "batch_size": batch_size, "seed": seed, "device": str(target), "critic_horizon": horizon,
        "dataset_sha256": train_manifest["dataset_sha256"], "probe_dataset_sha256": probe_manifest["dataset_sha256"],
        "training_config_sha256": training_config.digest, "probe_config_sha256": probe_config.digest,
        "reward_manifest_sha256": training_config.section("reward")["manifest_sha256"],
        "reward_source": reward_source(recipe), "reward_config": derivation,
        "model_config": training_config.section("model"), "model_contract": train_contract,
        "counts": counts, "in_sample_splits": ["train"], "scientific_caveats": SCIENTIFIC_CAVEATS,
        "rollout_overlaps_training_runs": sorted(set(overlapping)),
        "unsupported_probe_chunk_lengths": sorted({row["chunk_length"] for row in rows
            if row["split"] == "rollout" and row["chunk_length"] not in lengths}),
        "run_ids": {split: sorted({row["run_id"] for row in rows if row["split"] == split}) for split in counts},
        "artifacts": list(artifacts)}
    atomic_json(output / "summary.json", summary)
    return summary
