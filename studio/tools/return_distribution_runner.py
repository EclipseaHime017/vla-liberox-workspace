"""Roll out a Value Flows policy recording every critic-scored candidate, then plot the
return distributions its exported critic assigns them (``train.return_distribution``)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from studio.storage.files import atomic_write_json
from studio.tools.critic_runner import frozen_catalog, simulate

ROLLOUT_KEYS = {"max_steps", "open_loop_steps", "seed", "init_state_index", "control_hz",
                "success_consecutive_steps"}
STORED_ORIENTATION = "libero_raw"  # frames as the platform recorder stores them
DISPLAY_ORIENTATION = "rotate_180"  # upright, as in the platform's videos
PANEL = (800, 368)  # rendered histogram/timeline panel (width, height), multiples of 16


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    keys = {"schema_version", "job_id", "policy_snapshot", "task_snapshot", "rollout", "analysis",
            "artifacts_dir"}
    if not isinstance(spec, dict) or spec.keys() != keys or spec["schema_version"] != 1:
        raise ValueError("Invalid return distribution configuration")
    root = path.resolve().parent
    if spec["job_id"] != root.name:
        raise ValueError("Return distribution job identity mismatch")
    target = Path(spec["artifacts_dir"])
    if target.is_symlink() or not target.is_absolute() or not target.resolve().is_relative_to(root):
        raise ValueError("Return distribution artifacts must belong to its job directory")
    rollout = spec["rollout"]
    if not isinstance(rollout, dict) or rollout.keys() != ROLLOUT_KEYS:
        raise ValueError("Invalid return distribution rollout configuration")
    for name, value in rollout.items():
        if type(value) is not int or value < (0 if name in {"seed", "init_state_index"} else 1):
            raise ValueError(f"Invalid return distribution rollout {name}")
    analysis = spec["analysis"]
    if (not isinstance(analysis, dict) or analysis.keys() != {"return_samples", "seed"}
            or type(analysis["return_samples"]) is not int or not 16 <= analysis["return_samples"] <= 4096
            or type(analysis["seed"]) is not int or analysis["seed"] < 0):
        raise ValueError("Invalid return distribution analysis configuration")
    return spec


def rollout(spec: dict, directory: Path, *, stop_requested=lambda: False) -> None:
    """Simulate the policy, keeping each query's candidates, critic scores and chosen index."""
    queries = []

    def query(provider, observation, prompt):
        actions, selection = provider.predict_with_selection(observation, prompt, ())
        queries.append(selection)
        return actions

    recorder, result, _ = simulate(spec, directory, stop_requested=stop_requested, query=query)
    steps = np.asarray(recorder.inference_query_steps, dtype=np.int64)
    if len(steps) != len(queries) or not queries:
        raise ValueError("Every policy query must report its critic selection")
    chosen = np.asarray([item["chosen"] for item in queries], dtype=np.int64)
    candidates = np.stack([item["candidates"] for item in queries]).astype(np.float32)
    for index, chunk in enumerate(recorder.inference_action_chunks):
        if not np.array_equal(candidates[index, chosen[index]], chunk):
            raise ValueError("Recorded chunks differ from the reported critic selection")
    output = directory / "rollout"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **recorder.arrays())
    np.savez_compressed(output / "observations.npz", **recorder.observation_arrays())
    np.savez_compressed(output / "selection.npz", query_step=steps, candidates=candidates, chosen=chosen,
                        scores=np.stack([item["scores"] for item in queries]).astype(np.float32))
    atomic_write_json(Path(spec["artifacts_dir"]) / "progress.json", {
        "phase": "rollout_complete", "rollout_step": recorder.action_count,
        "rollout_steps": spec["rollout"]["max_steps"], "success": bool(result.success)})


def _histograms(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros((*samples.shape[:2], len(edges) - 1), dtype=np.int32)
    for query in range(samples.shape[0]):
        for candidate in range(samples.shape[1]):
            counts[query, candidate] = np.histogram(samples[query, candidate], bins=edges)[0]
    return counts


def _panels(data: dict, edges: np.ndarray) -> list[np.ndarray]:
    """One rendered histogram + timeline panel per query (RGB arrays of size ``PANEL``)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    returns, chosen, scores = data["returns"], data["chosen"], data["scores"]
    times, reference = data["query_time"], data["reference"]
    rows = np.arange(len(chosen))
    picked = returns[rows, chosen]
    low, middle, high = (np.quantile(picked, q, axis=1) for q in (0.1, 0.5, 0.9))
    width, height = PANEL
    figure, (histogram, timeline) = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=100,
                                                 gridspec_kw={"width_ratios": [1.15, 1]})
    # One density scale for the whole video, so heights compare across queries.
    peak = max(float(np.histogram(returns[q, c], bins=edges, density=True)[0].max())
               for q in rows for c in range(returns.shape[1])) * 1.08 or 1.
    panels = []
    for query in rows:
        histogram.clear(); timeline.clear()
        others = [candidate for candidate in range(returns.shape[1]) if candidate != chosen[query]]
        for order, candidate in enumerate(others):
            histogram.hist(returns[query, candidate], bins=edges, density=True, histtype="step",
                           color="#9aa4b2", linewidth=0.8, alpha=0.7,
                           label="other candidates" if order == 0 else None)
        histogram.hist(returns[query, chosen[query]], bins=edges, density=True, histtype="stepfilled",
                       color="#d9534f", alpha=0.65, label="chosen chunk")
        histogram.axvline(scores[query, chosen[query]], color="#d9534f", linestyle=":", linewidth=1.2,
                          label="selection score")
        if reference is not None:
            histogram.axvline(reference[query], color="#222", linestyle="--", linewidth=1.2, label="realized return")
        histogram.set_xlim(edges[0], edges[-1]); histogram.set_ylim(0, peak)
        histogram.set_xlabel("return"); histogram.set_ylabel("density")
        histogram.set_title(f"query {query + 1}/{len(rows)} · step {int(data['query_step'][query])}", fontsize=9)
        histogram.legend(fontsize=7, loc="upper left")
        timeline.fill_between(times, low, high, color="#d9534f", alpha=0.2, label="chosen 10–90%")
        timeline.plot(times, middle, color="#d9534f", linewidth=1.4, label="chosen median")
        if reference is not None:
            timeline.plot(times, reference, color="#222", linestyle="--", linewidth=1.2, label="realized return")
        timeline.axvline(times[query], color="#555", linewidth=0.8)
        timeline.set_ylim(edges[0], edges[-1]); timeline.set_xlabel("time (s)"); timeline.set_ylabel("return")
        timeline.legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        figure.canvas.draw()
        panels.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return panels


def _write_videos(artifacts: Path, frames: np.ndarray, steps: np.ndarray, panels: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    side = PANEL[1]
    query = np.searchsorted(steps, np.arange(len(frames)), side="right") - 1
    with imageio.get_writer(artifacts / "rollout.mp4", fps=fps, macro_block_size=16) as plain, \
            imageio.get_writer(artifacts / "returns.mp4", fps=fps, macro_block_size=16) as combined:
        for index, frame in enumerate(frames):
            image = np.asarray(Image.fromarray(frame).resize((side, side), Image.BILINEAR))
            plain.append_data(image)
            panel = panels[max(int(query[index]), 0)]
            combined.append_data(np.concatenate((image, panel), axis=1))


def analyze(spec: dict, directory: Path, *, stop_requested=lambda: False) -> dict:
    """Sample each candidate's return distribution with the policy's exported critic."""
    from core.data import confirmed_terminal_step
    from core.image_transforms import image_orientation, policy_view
    from core.model_config import model_backend, replay_stats
    from core.registry import model_observation
    from core.io import sha256_file
    from train.methods.critics import ValueFlowScorer
    from train.methods.selectors import load_critic
    from train.return_distribution import candidate_returns, histogram_edges, realized_returns

    artifacts = Path(spec["artifacts_dir"])
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib"))
    entry = frozen_catalog(spec, directory).entry(spec["policy_snapshot"]["policy_id"])
    critic = entry.artifacts.get("critic")
    if critic is None or sha256_file(critic) != entry.component_sha256.get("critic"):
        raise ValueError("The policy's exported critic is missing or changed")
    scorer = load_critic(critic, "cpu")
    if not isinstance(scorer, ValueFlowScorer) or not {"num_flow_steps", "ret_agg", "discount", "reward_config"} <= scorer.settings.keys():
        raise ValueError("The policy does not export a Value Flows critic with its return semantics; re-export it")
    with np.load(directory / "rollout" / "trajectory.npz", allow_pickle=False) as source:
        trajectory = {key: source[key] for key in source.files}
    with np.load(directory / "rollout" / "observations.npz", allow_pickle=False) as source:
        observations = {key: source[key] for key in source.files}
    with np.load(directory / "rollout" / "selection.npz", allow_pickle=False) as source:
        selection = {key: source[key] for key in source.files}
    if stop_requested():
        raise InterruptedError("Return distribution canceled before analysis")
    action_stats, proprio_stats = replay_stats(entry.settings)
    returns = candidate_returns(
        scorer, trajectory, observations, selection, stored_orientation=STORED_ORIENTATION,
        image_orientation=image_orientation(model_observation(entry.settings)),
        action_stats=action_stats, proprio_stats=proprio_stats,
        process_action=model_backend({"model": entry.settings}).process_action,
        samples=spec["analysis"]["return_samples"], seed=spec["analysis"]["seed"])
    reference, reason = realized_returns(trajectory["done"], scorer.settings, selection["query_step"])
    edges = histogram_edges(returns, reference)
    fps = spec["rollout"]["control_hz"]
    data = {"returns": returns, "chosen": selection["chosen"], "scores": selection["scores"],
            "query_step": selection["query_step"], "query_time": selection["query_step"] / fps,
            "reference": reference}
    np.savez_compressed(artifacts / "returns.npz", returns=returns, scores=selection["scores"],
                        chosen=selection["chosen"], query_step=selection["query_step"], edges=edges,
                        **({} if reference is None else {"realized_return": reference}))
    counts = _histograms(returns, edges)
    quantiles = np.quantile(returns, (0.1, 0.5, 0.9), axis=2)
    atomic_write_json(artifacts / "returns.json", {
        "schema_version": 1, "fps": fps, "action_count": int(len(trajectory["done"])),
        "edges": edges.tolist(), "samples": int(returns.shape[2]),
        "reference": {"available": reference is not None, "reason": reason,
                      "values": None if reference is None else reference.tolist()},
        "queries": [{"step": int(step), "chosen": int(chosen), "scores": scores.tolist(),
                     "mean": returns[index].mean(axis=1).tolist(), "q10": quantiles[0, index].tolist(),
                     "median": quantiles[1, index].tolist(), "q90": quantiles[2, index].tolist(),
                     "histograms": counts[index].tolist()}
                    for index, (step, chosen, scores) in enumerate(
                        zip(selection["query_step"], selection["chosen"], selection["scores"]))]})
    frames = np.stack([policy_view(frame, STORED_ORIENTATION, DISPLAY_ORIENTATION)
                       for frame in observations["agentview_image"]])
    _write_videos(artifacts, frames, selection["query_step"], _panels(data, edges), fps)
    rows = np.arange(len(selection["chosen"]))
    summary = {"schema_version": 1, "kind": "value_flow_return_distribution",
               "policy_id": spec["policy_snapshot"]["policy_id"],
               "policy_content_sha256": spec["policy_snapshot"]["content_sha256"],
               "task_id": spec["task_snapshot"]["task_id"], "rollout": spec["rollout"],
               "analysis": spec["analysis"], "queries": int(len(rows)), "candidates": int(returns.shape[1]),
               "success": confirmed_terminal_step(trajectory["done"],
                                                  spec["rollout"]["success_consecutive_steps"]) is not None,
               "discount": scorer.settings["discount"], "reward_config": scorer.settings["reward_config"],
               "return_bounds": scorer.settings["return_bounds"],
               # How often the one-sample selection picked the candidate with the highest mean return.
               "chosen_best_mean_fraction": float(np.mean(returns[rows, selection["chosen"]].mean(axis=1)
                                                          >= returns.mean(axis=2).max(axis=1) - 1e-6)),
               "reference_available": reference is not None, "reference_reason": reason}
    atomic_write_json(artifacts / "summary.json", summary)
    atomic_write_json(artifacts / "progress.json", {"phase": "completed",
        "rollout_step": int(len(trajectory["done"])), "rollout_steps": spec["rollout"]["max_steps"]})
    return summary

