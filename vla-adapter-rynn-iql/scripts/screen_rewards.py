"""Read-only chunk-return diagnostics; only NumPy and the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def discounted_returns(rewards, gamma, *, lengths=None, tail_value=0.0):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rewards must be a finite vector")
    if not 0 <= gamma <= 1 or not np.isfinite(tail_value):
        raise ValueError("Invalid gamma or tail value")
    exponents = np.ones(len(values)) if lengths is None else np.asarray(lengths)
    if exponents.shape != values.shape or not np.isfinite(exponents).all() or (exponents <= 0).any():
        raise ValueError("Discount lengths must be positive and aligned")
    result = np.empty_like(values)
    running = float(tail_value)
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + gamma ** exponents[index] * running
        result[index] = running
    return result


def expectile(values, tau):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < tau < 1:
        raise ValueError("Expectile needs finite values and 0 < tau < 1")
    lo, hi = float(values.min()), float(values.max())
    for _ in range(100):
        middle = (lo + hi) / 2
        residual = values - middle
        derivative = np.sum(np.where(residual >= 0, tau, 1 - tau) * residual)
        if derivative > 0:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2


def median_handoff_reference(grouped, series):
    """PLAN v4 R11: a declared 10+1 group heuristic, never same-state V(s)."""
    if len(grouped) != 5 or any(len(rows) != 10 for rows in grouped.values()):
        raise ValueError("PLAN v4 reference requires all 50 branches in 5 groups of 10")
    groups = []
    for parent, rows in sorted(grouped.items()):
        steps = [row["handoff_step"] for row in rows]
        middle = float(np.median(steps))
        chunks, _, returns = series[parent]
        indices = [i for i, chunk in enumerate(chunks) if chunk["start"] <= middle < chunk["end"]]
        if len(indices) != 1:
            raise ValueError("Median handoff does not identify exactly one parent chunk")
        index = indices[0]
        values = np.asarray([row["human_return"] for row in rows] + [float(returns[index])], dtype=np.float64)
        value = float(values.mean())
        for iteration in range(1000):
            coefficients = np.where(values >= value, .8, .2)
            updated = float(np.dot(coefficients, values) / coefficients.sum())
            if abs(updated - value) <= 1e-12:
                value = updated
                break
            value = updated
        else:
            raise ValueError("Weighted-mean expectile iteration did not converge")
        weight = np.exp(np.minimum(3. * (values - value), np.log(20.)))
        groups.append({"parent": parent, "branches": [r["branch"] for r in rows],
            "branch_count": 10, "handoff_steps": steps, "median_handoff_step": middle,
            "parent_chunk_index": index, "parent_chunk_start": chunks[index]["start"],
            "parent_chunk_end": chunks[index]["end"], "parent_offset": middle - chunks[index]["start"],
            "human_returns": values[:-1].tolist(), "policy_return": float(values[-1]),
            "expectile": value, "expectile_iterations": iteration + 1,
            "human_weights": weight[:-1].tolist(), "policy_weight": float(weight[-1])})
    return {"definition": "plan_v4_median_handoff_10_plus_1", "split": "all", "branch_count": 50,
        "parent_count": 5, "beta": 3., "tau": .8, "cap": 20., "groups": groups,
        "human_weights": summary([v for group in groups for v in group["human_weights"]]),
        "policy_weights": summary([group["policy_weight"] for group in groups]),
        "interpretation": "Group-level return heuristic, NOT same-state V(s) or trained critic advantage.",
        "p3_comparisons_changed": False}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))}


def full_chunks(episode):
    chunks = episode["chunks"]
    if not chunks or chunks[-1]["end"] != episode["recorded_action_count"]:
        complete = episode.get("evaluation_chunks", [])
        if len(chunks) > len(complete) or any(
            any(a[key] != b[key] for key in ("start", "end", "length", "action_source"))
            for a, b in zip(chunks, complete)
        ):
            raise ValueError(f"Misaligned historical replay: {episode['run_id']}")
        chunks = complete
    end = 0
    for chunk in chunks:
        if chunk["start"] != end or chunk["length"] <= 0 or chunk["end"] - end != chunk["length"]:
            raise ValueError(f"Invalid chunk partition: {episode['run_id']}")
        end = chunk["end"]
    if not chunks or end != episode["recorded_action_count"]:
        raise ValueError(f"Incomplete replay: {episode['run_id']}")
    return chunks


def validated_tail_values(payload, dataset, dataset_path, reward_path, *, gamma, split,
                          primitive_discount, critic_sha256):
    if not isinstance(payload, dict) or payload.get("kind") != "frozen_critic_tail_values":
        raise ValueError("Explicit terminal V requires an identity-bound frozen critic export")
    if (not critic_sha256 or len(critic_sha256) != 64 or payload.get("critic_sha256") != critic_sha256
            or payload.get("schema_version") != 1 or payload.get("dataset_sha256") != dataset["dataset_sha256"]
            or payload.get("dataset_manifest_sha256") != digest(dataset_path)
            or payload.get("reward_manifest_sha256") != digest(reward_path)
            or payload.get("chunking") != dataset.get("chunking")
            or payload.get("gamma") != gamma or payload.get("primitive_discount") is not primitive_discount
            or payload.get("split") not in {split, "all"}):
        raise ValueError("Tail critic/dataset/reward/gamma/split identity mismatch")
    for key in ("training_config_sha256", "summary_sha256", "normalization_sha256"):
        if not isinstance(payload.get(key), str) or len(payload[key]) != 64:
            raise ValueError(f"Tail export missing identity: {key}")
    expected = {e["run_id"]: e for e in dataset["episodes"] if e.get("terminal_step") is None
                and (payload["split"] == "all" or e["split"] == payload["split"])}
    values = {}
    for row in payload.get("rows", []):
        rid = row["run_id"]
        if rid not in expected or rid in values:
            raise ValueError("Unexpected/duplicate tail endpoint")
        episode = expected[rid]
        if (row.get("observation_step") != episode["recorded_action_count"]
                or row.get("last_chunk_start") != full_chunks(episode)[-1]["start"]
                or any(row.get(k) != episode.get(k) for k in
                       ("split", "root_run_id", "parent_run_id", "trajectory_sha256", "observations_sha256"))):
            raise ValueError(f"Tail endpoint identity mismatch: {rid}")
        for key in ("trajectory_sha256", "observations_sha256", "next_pixels_sha256", "next_proprio_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError(f"Tail endpoint missing evidence: {key}")
        value = float(row["tail_v"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite tail V")
        values[rid] = value
    if set(values) != set(expected):
        raise ValueError("Missing failure tail endpoint")
    return values


def analyze(dataset_path, reward_path, *, gamma, split="train", include_post_success=True,
            beta=3.0, tau=.8, primitive_discount=False, bootstrap_failures=False, tail_values=None,
            path_map=None, tail_critic_sha256=None, plan_v4_reference=False):
    dataset_path, reward_path = Path(dataset_path), Path(reward_path)
    dataset, manifest = json.loads(dataset_path.read_text()), json.loads(reward_path.read_text())
    if manifest.get("complete") is False or not dataset.get("dataset_sha256") or dataset["dataset_sha256"] != manifest.get("dataset_sha256"):
        raise ValueError("Incomplete reward version or dataset identity mismatch")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if plan_v4_reference and (split != "all" or beta != 3. or tau != .8 or primitive_discount
                              or bootstrap_failures or not include_post_success):
        raise ValueError("PLAN v4 reference requires all, beta3, tau.8, macro discount, original full recording/tail0")
    tail_evidence = None
    if bootstrap_failures:
        tail_evidence = tail_values
        tail_values = validated_tail_values(tail_values, dataset, dataset_path, reward_path,
            gamma=gamma, split=split, primitive_discount=primitive_discount, critic_sha256=tail_critic_sha256)
    episodes = {item["run_id"]: item for item in dataset["episodes"]}
    rewards = {item["run_id"]: item for item in manifest["episodes"]}
    if len(episodes) != len(dataset["episodes"]) or len(rewards) != len(manifest["episodes"]):
        raise ValueError("Duplicate episode ID")
    series, scales, inputs = {}, [], []
    for run_id, episode in episodes.items():
        if split != "all" and episode["split"] != split:
            continue
        entry = rewards[run_id]
        original_path = entry.get("reward_path") or entry["annotation_path"]
        if path_map is not None:
            if original_path not in path_map:
                raise ValueError(f"Reward path missing from explicit path map: {original_path}")
            path = Path(path_map[original_path])
        else:
            path = Path(original_path)
            if not path.is_absolute():
                path = reward_path.parent / path
        expected = entry.get("reward_sha256") or entry.get("annotation_sha256")
        actual = digest(path)
        if not expected or actual != expected:
            raise ValueError(f"Reward checksum mismatch/missing: {run_id}")
        chunks = full_chunks(episode)
        with np.load(path, allow_pickle=False) as source:
            boundaries = [chunks[0]["start"], *(c["end"] for c in chunks)]
            if "boundary_steps" in source and not np.array_equal(source["boundary_steps"], boundaries):
                raise ValueError(f"Reward boundaries differ from dataset chunks: {run_id}")
            key = "final_reward" if "final_reward" in source else "pbrs_chunk_reward"
            values = np.asarray(source[key], dtype=np.float64)
            scale = np.asarray(source["final_reward_scale"] if "final_reward_scale" in source else 1.)
            if scale.size != 1 or not np.isfinite(scale).all() or float(scale.reshape(-1)[0]) <= 0:
                raise ValueError(f"Invalid reward scale: {run_id}")
            scales.append(float(scale.reshape(-1)[0]))
        if values.shape != (len(chunks),) or not np.isfinite(values).all():
            raise ValueError(f"Unaligned/nonfinite rewards: {run_id}")
        terminal = episode.get("terminal_step")
        end = episode["recorded_action_count"] if include_post_success or terminal is None else terminal + 1
        selected = [c for c in chunks if c["end"] <= end]
        if not selected or selected[-1]["end"] != end:
            raise ValueError(f"Missing confirmed-success boundary: {run_id}")
        tail = 0.
        if bootstrap_failures and terminal is None:
            if tail_values is None or run_id not in tail_values:
                raise ValueError(f"Explicit terminal V required for truncated failure: {run_id}")
            tail = float(tail_values[run_id])
        values = values[:len(selected)]
        returns = discounted_returns(values, gamma, lengths=[c["length"] for c in selected] if primitive_discount else None, tail_value=tail)
        series[run_id] = (selected, values, returns)
        inputs.append({"run_id": run_id, "reward_sha256": actual, "tail_value": tail})

    comparisons, grouped = [], {}
    for run_id, (chunks, _, returns) in series.items():
        episode = episodes[run_id]
        if episode.get("kind") != "branch":
            continue
        candidates = [i for i, chunk in enumerate(chunks) if not chunk.get("copied_prefix", False)]
        if not candidates:
            continue
        index = candidates[0]
        start = chunks[index]["start"]
        if start != episode["resume_step"]:
            raise ValueError(f"First noncopy chunk differs from resume_step: {run_id}")
        parent_id = episode["parent_run_id"]
        if parent_id not in series:
            raise ValueError(f"Parent absent from selected split: {parent_id}")
        parent_chunks, _, parent_returns = series[parent_id]
        parent_index = next((i for i, c in enumerate(parent_chunks) if c["start"] <= start < c["end"]), None)
        if parent_index is None:
            raise ValueError(f"No parent chunk at takeover: {run_id}")
        human, policy = float(returns[index]), float(parent_returns[parent_index])
        row = {"branch": run_id, "parent": parent_id, "handoff_step": start,
               "parent_chunk_start": parent_chunks[parent_index]["start"],
               "human_return": human, "policy_return": policy, "difference": human - policy}
        comparisons.append(row)
        grouped.setdefault(parent_id, []).append(row)
    weights, ambiguities, parent_return_groups = [], [], []
    for parent, rows in grouped.items():
        policy_values = [r["policy_return"] for r in rows]
        parent_return_groups.append({"parent": parent, "branch_count": len(rows),
            "handoff_steps": sorted({r["handoff_step"] for r in rows}),
            "parent_chunk_starts": sorted({r["parent_chunk_start"] for r in rows}),
            "human_returns": summary([r["human_return"] for r in rows]),
            "paired_covering_policy_returns": summary(policy_values),
            "unique_policy_return_count": len(set(policy_values)),
            "interpretation": "Group-level return reference, not same-state V or an advantage proof."})
        if len(set(policy_values)) != 1:
            ambiguities.append({"parent": parent,
                "reason": "multiple covering-chunk policy returns; no prescribed single policy return for 10+1 pooling",
                "policy_return_count": len(set(policy_values))})
            continue
        policy = rows[0]["policy_return"]
        human = [r["human_return"] for r in rows]
        value = expectile(human + [policy], tau)
        weight = np.exp(np.minimum(beta * (np.asarray(human + [policy]) - value), np.log(20.)))
        weights.append({"parent": parent, "branch_count": len(rows), "value": value,
                        "interpretation": "group-level return expectile, not same-state V(s)",
                        "human_weights": weight[:-1].tolist(), "policy_weight": float(weight[-1])})

    # Exclude copied prefixes represented by their original producer, matching replay selection.
    def key(episode, chunk):
        origin = chunk.get("origin_run_id") or (episode["root_run_id"] if chunk.get("copied_prefix", False) else episode["run_id"])
        return origin, chunk["start"], chunk["end"], chunk["action_source"]
    produced = {key(episodes[rid], c) for rid, (chunks, _, _) in series.items() for c in chunks if not c.get("copied_prefix", False)}
    seen, categories = set(), {name: [] for name in ("human_pre_success", "human_post_success", "failed_parent", "other")}
    for rid in sorted(series, key=lambda rid: (episodes[rid].get("kind") == "branch", rid)):
        episode = episodes[rid]
        chunks, values, _ = series[rid]
        for chunk, reward in zip(chunks, values):
            identity = key(episode, chunk)
            if (chunk.get("copied_prefix", False) and identity in produced) or identity in seen:
                continue
            seen.add(identity)
            terminal = episode.get("terminal_step")
            if chunk["action_source"] == "human":
                name = "human_post_success" if terminal is not None and chunk["start"] >= terminal + 1 else "human_pre_success"
            elif episode.get("kind") != "branch" and terminal is None:
                name = "failed_parent"
            else:
                name = "other"
            categories[name].append(float(reward))
    differences = [r["difference"] for r in comparisons]
    return {"schema_version": 1, "dataset_sha256": dataset["dataset_sha256"],
            "dataset_manifest_sha256": digest(dataset_path), "reward_manifest_sha256": digest(reward_path),
            "settings": {"gamma": gamma, "split": split, "include_post_success": include_post_success,
                         "beta": beta, "expectile": tau, "primitive_discount": primitive_discount,
                         "bootstrap_failures": bootstrap_failures},
            "inputs": inputs, "comparisons": comparisons,
            "tail_evidence": None if tail_evidence is None else {
                "critic_sha256": tail_evidence["critic_sha256"],
                "summary_sha256": tail_evidence["summary_sha256"],
                "payload_sha256": hashlib.sha256(json.dumps(tail_evidence, sort_keys=True).encode()).hexdigest(),
                "caveat": tail_evidence.get("caveat")},
            "handoff_summary": {"count": len(comparisons), "human_higher_fraction": float(np.mean(np.asarray(differences) > 0)) if differences else None,
                                "difference": summary(differences), "human_return": summary([r["human_return"] for r in comparisons]),
                                "policy_return": summary([r["policy_return"] for r in comparisons])},
            "expectile_groups": weights, "expectile_ambiguities": ambiguities,
            "parent_return_groups": parent_return_groups,
            "human_weights": summary([w for group in weights for w in group["human_weights"]]),
            "policy_weights": summary([group["policy_weight"] for group in weights]),
            "plan_v4_reference": median_handoff_reference(grouped, series) if plan_v4_reference else None,
            "reward_groups": {name: summary(values) for name, values in categories.items()},
            "normalization_scales": summary(scales)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--reward-manifest", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="train")
    parser.add_argument("--exclude-post-success", action="store_true")
    parser.add_argument("--primitive-discount", action="store_true")
    parser.add_argument("--bootstrap-failures", action="store_true")
    parser.add_argument("--plan-v4-reference", action="store_true", help="Fixed all50 median-handoff 10+1 reference, beta3/tau.8")
    parser.add_argument("--tail-values", type=Path, help="Identity-bound export_critic_tail_values.py JSON")
    parser.add_argument("--tail-critic-sha256", help="Independently recorded frozen critic checksum")
    parser.add_argument("--path-map", type=Path, help="Read-only source path mapping; relative targets resolve from this JSON file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path_map = None
    if args.path_map:
        path_map = json.loads(args.path_map.read_text())
        if not isinstance(path_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in path_map.items()):
            raise ValueError("Path map must contain source/target strings")
        path_map = {key: (args.path_map.parent / value).resolve() for key, value in path_map.items()}
    result = analyze(args.dataset_manifest, args.reward_manifest, gamma=args.gamma, split=args.split,
                     include_post_success=not args.exclude_post_success, primitive_discount=args.primitive_discount,
                     bootstrap_failures=args.bootstrap_failures,
                     tail_values=json.loads(args.tail_values.read_text()) if args.tail_values else None,
                     path_map=path_map, tail_critic_sha256=args.tail_critic_sha256,
                     plan_v4_reference=args.plan_v4_reference)
    if args.path_map:
        result["path_map_sha256"] = digest(args.path_map)
    with args.output.open("x") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
