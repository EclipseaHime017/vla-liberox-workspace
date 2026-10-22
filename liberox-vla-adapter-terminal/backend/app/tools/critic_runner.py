"""Collect a frozen-policy probe, then fit critics in a separate process."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import yaml

from ..storage.files import atomic_write_json, atomic_write_yaml


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    keys = {"schema_version", "job_id", "dataset_id", "dataset_name", "training_config",
            "policy_snapshot", "task_snapshot", "rollout", "critic", "artifacts_dir"}
    if not isinstance(spec, dict) or spec.keys() != keys or spec["schema_version"] != 1:
        raise ValueError("Invalid critic diagnostic configuration")
    root = path.resolve().parent
    if spec["job_id"] != root.name:
        raise ValueError("Critic diagnostic job identity mismatch")
    for name in ("training_config", "artifacts_dir"):
        target = Path(spec[name])
        if target.is_symlink() or not target.is_absolute() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Diagnostic {name} must belong to its job directory")
    rollout = spec["rollout"]
    required = {"max_steps", "open_loop_steps", "seed", "init_state_index", "control_hz", "success_consecutive_steps"}
    if not isinstance(rollout, dict) or rollout.keys() != required:
        raise ValueError("Invalid diagnostic rollout configuration")
    for name, value in rollout.items():
        if type(value) is not int or value < (0 if name in {"seed", "init_state_index"} else 1):
            raise ValueError(f"Invalid diagnostic rollout {name}")
    if not isinstance(spec["critic"], dict) or spec["critic"].keys() != {"train_steps", "batch_size", "seed"}:
        raise ValueError("Invalid critic training configuration")
    for name, value in spec["critic"].items():
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid critic {name}")
    return spec


def collect_episode(*, simulator, provider, env, initial_state, prompt: str,
                    settings: dict, stop_requested: Callable[[], bool], progress=None):
    import eval_pickplace_direct as direct
    from simulation_core import run_control_loop
    from trajectory_utils import TrajectoryRecorder

    observation = simulator.observations(env, simulator.restore(env, initial_state))
    class ObservedRecorder(TrajectoryRecorder):
        def _append_state(self, env, observation):
            # LIBERO's control env disables camera observations; render the raw
            # pair here before the N+1 state is persisted, not after recording.
            super()._append_state(env, simulator.observations(env, observation))

    recorder = ObservedRecorder(control_hz=settings["control_hz"], capture_images=True)
    recorder.policy_action_codec = provider.metadata()["recording_action_codec"]
    recorder.record_initial(env, observation)

    def query(observation, _step):
        observation = simulator.observations(env, observation)
        return observation, provider.predict(observation, prompt, ())

    def transition(_observation, step, _success):
        if progress and (step == 1 or step % settings["open_loop_steps"] == 0 or step == settings["max_steps"]):
            progress({"phase": "rollout", "rollout_step": step, "rollout_steps": settings["max_steps"]})

    result = run_control_loop(
        env=env, recorder=recorder, initial_observation=observation,
        target_action_count=settings["max_steps"],
        rate_limiter=direct.RealTimeControlLimiter(settings["control_hz"], False),
        action_source="policy", open_loop_steps=settings["open_loop_steps"],
        policy_query=query, policy_action_transform=provider.process_action,
        stop_requested=stop_requested, on_transition=transition,
        stop_on_success=False,
    )
    if stop_requested() or result.stopped_reason == "user_stop":
        raise InterruptedError("Critic diagnostic rollout canceled")
    return recorder, result


def _catalog(spec: dict, directory: Path):
    from ..policies.snapshots import catalog_from_snapshot
    return catalog_from_snapshot(spec["policy_snapshot"], directory / ".empty-policy-registry")


def rollout(spec: dict, directory: Path, *, stop_requested=lambda: False) -> Path:
    import eval_pickplace_direct as direct
    from simulation_core import confirmed_success_step
    from trajectory_utils import save_trajectory_bundle
    from ..policies.router import PolicyProvider
    from ..services.task_registry import TaskSnapshot
    from ..simulators.libero_x import LiberoXSimulator
    from vla_rynn_iql.config import LoadedConfig, load_train_config
    from vla_rynn_iql.data import prepare_dataset

    policy, task, settings = spec["policy_snapshot"], spec["task_snapshot"], spec["rollout"]
    if stop_requested():
        raise InterruptedError("Critic diagnostic canceled before rollout")
    catalog = _catalog(spec, directory)
    base = direct.load_config(direct.DEFAULT_CONFIG_PATH.resolve())
    config = replace(base, checkpoint=policy["base_checkpoint"], stats_key=policy["stats_key"],
        level=task["level"], task_name=task["task_name"], max_steps=settings["max_steps"],
        open_loop_steps=settings["open_loop_steps"], seed=settings["seed"],
        control_hz=settings["control_hz"], realtime_control=False, disabled_policy_cameras=(),
        headless=True, no_video=True, save_trajectory=True, save_observation_images=True,
        trajectory_plot=False)
    direct.apply_runtime_environment(config)
    direct.add_repo_paths(config.vla_root, config.liberox_root)
    runtime = direct.load_runtime()
    if "task_revision_sha256" in task:
        frozen = TaskSnapshot.load(runtime, directory / "task_snapshot" / "task.json")
        if frozen.manifest["task_revision_sha256"] != task["task_revision_sha256"]:
            raise ValueError("Diagnostic task snapshot changed after registration")
        bddl, init = frozen.paths(task["task_id"])
        if (Path(task["bddl_path"]), Path(task["init_path"])) != (bddl, init):
            raise ValueError("Diagnostic task paths differ from their frozen snapshot")
    else:
        bddl, init = Path(task["bddl_path"]), Path(task["init_path"])
    states = direct.load_initial_states(runtime, init)
    if settings["init_state_index"] >= len(states):
        raise ValueError("Diagnostic init state is out of range")
    simulator = LiberoXSimulator(runtime)
    provider = PolicyProvider(runtime, config, catalog)
    env = None
    artifacts = Path(spec["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    def progress(value):
        atomic_write_json(artifacts / "progress.json", value)

    try:
        provider.load(settings["open_loop_steps"], policy["policy_id"],
                      expected_content_sha256=policy["content_sha256"])
        set_seed = getattr(runtime, "set_seed_everywhere", None)
        if callable(set_seed):
            set_seed(settings["seed"])
        provider.seed(settings["seed"])
        env = simulator.create(bddl, config, max_steps=settings["max_steps"], seed=settings["seed"])
        recorder, result = collect_episode(simulator=simulator, provider=provider, env=env,
            initial_state=states[settings["init_state_index"]], prompt=task["prompt"],
            settings=settings, stop_requested=stop_requested, progress=progress)
        metadata = provider.metadata()
    finally:
        try:
            provider.unload()
        finally:
            if env is not None:
                direct.close_env(env)
    _catalog(spec, directory)  # No fallback if weights changed during rollout.
    training = load_train_config(Path(spec["training_config"]))
    identifier = f"{spec['job_id']}_rollout"
    run_dir = directory / "probe" / "runs" / identifier
    episode = run_dir / "episodes" / "episode_000"
    record = {"id": identifier, "project_id": training.section("data")["project_id"],
        "kind": "original", "root_session_id": identifier, "parent_session_id": None,
        "resume_step": None, "task_id": task["task_id"], "task": task["prompt"],
        "task_name": task["task_name"], "level": task["level"], "status": "COMPLETED",
        "error": None, "control_mode": "policy", "open_loop_steps": settings["open_loop_steps"],
        "seed": settings["seed"], "init_state_index": settings["init_state_index"],
        "action_count": recorder.action_count, "state_count": recorder.state_count,
        "success": result.success, "success_confirmed_step": confirmed_success_step(
            recorder.dones, settings["success_consecutive_steps"]),
        "success_confirm_steps": settings["success_consecutive_steps"],
        "trajectory_segments": [{"start": 0, "end": recorder.action_count,
            "origin_run_id": identifier, "open_loop_steps": settings["open_loop_steps"]}],
        "policy_id": policy["policy_id"], "policy_family": policy["family"], **metadata}
    save_trajectory_bundle(recorder, episode / "trajectory", record,
                           save_observations=True, create_plot=False, save_metadata_json=False)
    atomic_write_json(run_dir / "run.json", record)
    raw = copy.deepcopy(training.raw)
    raw["paths"].update(dataset_sources=[str(run_dir)], work_dir=str(directory / "probe" / "work"),
                        annotation_cache=str(directory / "probe" / "cache"))
    raw["data"].update(selection_manifest=None, stage_annotations_manifest=None,
                       task_ids=[task["task_id"]], validation_fraction=0.,
                       include_post_success=True, allow_no_success=True)
    probe = directory / "probe" / "config.yaml"
    atomic_write_yaml(probe, raw)
    prepare_dataset(LoadedConfig(probe, raw))  # No reward materialization for the probe.
    progress({"phase": "rollout_complete", "rollout_step": recorder.action_count,
              "rollout_steps": settings["max_steps"]})
    return probe


def fit(spec: dict, directory: Path, *, stop_requested=lambda: False) -> dict:
    from vla_rynn_iql.config import load_train_config
    from vla_rynn_iql.critic_diagnostics import run_critic_diagnostic

    _catalog(spec, directory)
    output = Path(spec["artifacts_dir"])
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib"))
    training = load_train_config(Path(spec["training_config"]))
    probe = load_train_config(directory / "probe" / "config.yaml")
    last_progress = 0.

    def progress(value):
        nonlocal last_progress
        now = time.monotonic()
        if (now - last_progress >= 1. or value.get("train_step") in {1, spec["critic"]["train_steps"]}):
            atomic_write_json(output / "progress.json", value)
            last_progress = now

    summary = run_critic_diagnostic(training, probe, output, **spec["critic"],
        device=training.section("training")["device"], stop_requested=stop_requested,
        progress=progress)
    _catalog(spec, directory)
    summary.update(policy_id=spec["policy_snapshot"]["policy_id"],
                   policy_content_sha256=spec["policy_snapshot"]["content_sha256"],
                   dataset_id=spec["dataset_id"], dataset_name=spec["dataset_name"],
                   task_id=spec["task_snapshot"]["task_id"], rollout=spec["rollout"])
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "progress.json", {"phase": "completed",
        "train_step": spec["critic"]["train_steps"], "train_steps": spec["critic"]["train_steps"]})
    return summary
