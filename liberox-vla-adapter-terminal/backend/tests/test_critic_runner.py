from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import eval_pickplace_direct as direct
from backend.app.tools import critic_runner as runner


def _observation(step):
    return {"robot0_eef_pos": np.asarray([step / 100., 0., 0.]),
            "robot0_eef_quat": np.asarray([0., 0., 0., 1.]),
            "robot0_gripper_qpos": np.asarray([.02, .02])}


def _image(step):
    return np.arange(12, dtype=np.uint8).reshape(2, 2, 3) + step


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.actions = []

    def get_sim_state(self):
        return np.asarray([float(self.steps), 0.])

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        # The real platform uses use_camera_obs=False: env.step has no pixels.
        return _observation(self.steps), 0., self.steps <= 5, {}


class FakeSimulator:
    def __init__(self):
        self.env = FakeEnv()
        self.creation = None

    def create(self, bddl, config, **kwargs):
        self.creation = (bddl, config, kwargs)
        return self.env

    @staticmethod
    def restore(env, initial_state):
        assert np.array_equal(initial_state, [1., 2.])
        env.steps = 0
        return _observation(0)

    @staticmethod
    def observations(env, observation):
        return {"agentview_image": _image(env.steps).copy(),
                "robot0_eye_in_hand_image": (_image(env.steps) + 50).copy(), **observation}


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.queries = []
        self.unloaded = False

    def metadata(self):
        return {"recording_action_codec": "libero_env_v1"}

    def predict(self, observation, prompt, disabled):
        self.queries.append((copy.deepcopy(observation), prompt, disabled))
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, -1] = -1.
        return actions

    @staticmethod
    def process_action(action):
        return action

    def load(self, steps, policy_id, **kwargs):
        self.calls.append((steps, policy_id, kwargs))

    def seed(self, seed):
        self.seed_value = seed

    def unload(self):
        self.unloaded = True


@pytest.fixture
def settings():
    return {"max_steps": 11, "open_loop_steps": 4, "seed": 7, "init_state_index": 0,
            "control_hz": 20, "success_consecutive_steps": 5}


def test_collection_records_raw_n_plus_one_images_and_real_inference_boundaries(settings):
    simulator, provider = FakeSimulator(), FakeProvider()
    progress = []
    recorder, result = runner.collect_episode(simulator=simulator, provider=provider, env=simulator.env,
        initial_state=np.asarray([1., 2.]), prompt="place the bowl", settings=settings,
        stop_requested=lambda: False, progress=progress.append)
    assert simulator.env.steps == settings["max_steps"]
    assert result.success is True and result.stopped_reason == "horizon"
    assert recorder.action_count == 11 and recorder.state_count == 12
    assert recorder.inference_query_steps == [0, 4, 8]
    assert [len(chunk) for chunk in recorder.inference_action_chunks] == [8, 8, 8]
    for step, (agent, wrist) in enumerate(zip(recorder.agentview_images, recorder.wrist_images)):
        np.testing.assert_array_equal(agent, _image(step))
        np.testing.assert_array_equal(wrist, _image(step) + 50)
    assert len(provider.queries) == 3
    for step, (observation, prompt, disabled) in zip((0, 4, 8), provider.queries):
        np.testing.assert_array_equal(observation["agentview_image"], _image(step))
        assert prompt == "place the bowl" and disabled == ()
    assert recorder.dones[:5] == [True] * 5 and recorder.dones[5:] == [False] * 6
    assert [item["rollout_step"] for item in progress] == [1, 4, 8, 11]


def test_collection_cancellation_does_not_return_a_successful_probe(settings):
    simulator, provider = FakeSimulator(), FakeProvider()
    with pytest.raises(InterruptedError, match="canceled"):
        runner.collect_episode(simulator=simulator, provider=provider, env=simulator.env,
            initial_state=np.asarray([1., 2.]), prompt="task", settings=settings,
            stop_requested=lambda: simulator.env.steps >= 6)
    assert simulator.env.steps == 6


@pytest.fixture
def worker_spec(tmp_path, settings, monkeypatch):
    from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
    import backend.app.policies.router as providers
    import backend.app.services.task_registry as tasks
    import backend.app.simulators.libero_x as simulators

    directory = tmp_path / "critic_test"
    directory.mkdir()
    artifacts = directory / "artifacts"
    raw = load_train_config(DEFAULT_TRAIN_CONFIG).raw
    raw["paths"].update(work_dir=str(directory / "train/work"),
                        output_dir=str(directory / "train/output"),
                        annotation_cache=str(directory / "train/cache"))
    training = directory / "training.yaml"
    training.write_text(yaml.safe_dump(raw))
    snapshot = directory / "task_snapshot"
    snapshot.mkdir()
    bddl, init = snapshot / "task.bddl", snapshot / "initial_states.init"
    bddl.write_text("fake task")
    init.write_text("fake states")
    task = {"task_id": "LEVEL1::task", "task_name": "task", "level": "LEVEL1", "prompt": "place the bowl",
            "bddl_path": str(bddl), "init_path": str(init), "task_revision_sha256": "f" * 64}
    spec = {"schema_version": 1, "job_id": directory.name, "dataset_id": "dataset-test",
            "dataset_name": "diagnostic data", "training_config": str(training),
            "policy_snapshot": {"policy_id": "selected-model", "family": "vla_adapter",
                "base_checkpoint": "selected-weights", "stats_key": "selected-stats", "content_sha256": "a" * 64},
            "task_snapshot": task, "rollout": dict(settings),
            "critic": {"train_steps": 2, "batch_size": 3, "seed": 12}, "artifacts_dir": str(artifacts)}
    simulator, provider = FakeSimulator(), FakeProvider()
    monkeypatch.setattr(providers, "PolicyProvider", lambda *_: provider)
    monkeypatch.setattr(simulators, "LiberoXSimulator", lambda *_: simulator)
    monkeypatch.setattr(direct, "apply_runtime_environment", lambda _: None)
    monkeypatch.setattr(direct, "add_repo_paths", lambda *_: None)
    runtime = SimpleNamespace(seeds=[])
    runtime.set_seed_everywhere = runtime.seeds.append
    provider.runtime = runtime
    monkeypatch.setattr(direct, "load_runtime", lambda: runtime)
    monkeypatch.setattr(direct, "load_initial_states", lambda *_: [np.asarray([1., 2.])])
    monkeypatch.setattr(tasks.TaskSnapshot, "load", lambda *_: SimpleNamespace(
        paths=lambda _: (bddl, init), manifest={"task_revision_sha256": task["task_revision_sha256"]}))
    catalog_calls = []
    monkeypatch.setattr(runner, "_catalog", lambda *args: catalog_calls.append(args) or "catalog")
    closed = []
    monkeypatch.setattr(direct, "close_env", closed.append)
    return spec, directory, simulator, provider, catalog_calls, closed


def test_rollout_prepares_private_reward_free_probe_and_unloads_selected_model(worker_spec, monkeypatch):
    spec, directory, simulator, provider, catalog_calls, closed = worker_spec
    from vla_rynn_iql import rewards
    from vla_rynn_iql.config import load_train_config
    from vla_rynn_iql.data import load_manifest

    monkeypatch.setattr(rewards, "materialize_reward_manifest", lambda *a, **k: pytest.fail("probe must have no reward"))
    path = runner.rollout(spec, directory)
    assert path == directory / "probe/config.yaml"
    assert provider.calls == [(4, "selected-model", {"expected_content_sha256": "a" * 64})]
    assert provider.seed_value == 7 and provider.unloaded is True
    assert provider.runtime.seeds == [7]
    assert closed == [simulator.env]
    assert len(catalog_calls) == 2
    _, config, kwargs = simulator.creation
    assert config.checkpoint == "selected-weights" and config.stats_key == "selected-stats"
    assert config.realtime_control is False and config.disabled_policy_cameras == ()
    assert kwargs == {"max_steps": 11, "seed": 7}
    probe = load_train_config(path)
    assert probe.section("data")["include_post_success"] is True
    assert probe.section("data")["selection_manifest"] is None
    manifest = load_manifest(probe)
    assert manifest["episode_count"] == 1
    episode = manifest["episodes"][0]
    assert episode["inference_query_steps"] == [0, 4, 8]
    assert episode["recorded_action_count"] == 11
    assert episode["observation_orientation"] == "libero_raw"
    assert [(c["start"], c["end"]) for c in episode["chunks"]] == [(0, 4), (4, 5), (5, 8), (8, 11)]
    with np.load(episode["observations_path"], allow_pickle=False) as arrays:
        assert arrays["agentview_image"].shape == (12, 2, 2, 3)
        for step, image in enumerate(arrays["agentview_image"]):
            np.testing.assert_array_equal(image, _image(step))
    run_path = next((directory / "probe/runs").rglob("run.json"))
    run = json.loads(run_path.read_text())
    assert run["success_confirmed_step"] == 4 and run["action_count"] == 11
    assert not list(directory.rglob("reward_manifest.json"))
    assert not list(directory.rglob("policy.yaml"))
    assert not list(directory.rglob("catalog.sqlite3"))


def test_rollout_cancellation_unloads_and_does_not_publish_probe(worker_spec):
    spec, directory, simulator, provider, _, closed = worker_spec
    with pytest.raises(InterruptedError):
        runner.rollout(spec, directory, stop_requested=lambda: simulator.env.steps >= 6)
    assert provider.unloaded is True and closed == [simulator.env]
    assert not list(directory.rglob("run.json"))
    assert not (directory / "probe/config.yaml").exists()


def test_rollout_load_failure_unloads_and_never_uses_another_policy(worker_spec, monkeypatch):
    spec, directory, _, provider, _, closed = worker_spec
    def fail(*args, **kwargs):
        raise ValueError("Selected policy weights changed")
    monkeypatch.setattr(provider, "load", fail)
    with pytest.raises(ValueError, match="changed"):
        runner.rollout(spec, directory)
    assert provider.unloaded is True and closed == []
    assert provider.queries == []
    assert not list(directory.rglob("run.json"))


def test_rollout_rechecks_identity_before_publishing_probe(worker_spec, monkeypatch):
    spec, directory, simulator, provider, _, closed = worker_spec
    checks = []
    def check(*args):
        checks.append(True)
        if len(checks) == 2:
            raise ValueError("Selected policy weights changed after rollout")
        return "catalog"
    monkeypatch.setattr(runner, "_catalog", check)
    with pytest.raises(ValueError, match="changed after rollout"):
        runner.rollout(spec, directory)
    assert provider.unloaded is True and closed == [simulator.env]
    assert not list(directory.rglob("run.json"))
    assert not (directory / "probe/config.yaml").exists()


def test_fit_calls_only_the_actor_free_core_with_frozen_configs(worker_spec, monkeypatch):
    spec, directory, _, provider, catalog_calls, _ = worker_spec
    from vla_rynn_iql import critic_diagnostics
    runner.rollout(spec, directory)
    catalog_calls.clear()
    called = []
    stopping = lambda: False

    def core(training, probe, output, **kwargs):
        called.append((training, probe, output, kwargs))
        kwargs["progress"]({"train_step": 1, "q_loss": .2})
        return {"actor_updated": False, "reward_generated": False}

    monkeypatch.setattr(critic_diagnostics, "run_critic_diagnostic", core)
    summary = runner.fit(spec, directory, stop_requested=stopping)
    assert len(called) == 1 and len(catalog_calls) == 2
    training, probe, output, kwargs = called[0]
    assert training.path == Path(spec["training_config"])
    assert probe.path == directory / "probe/config.yaml"
    assert output == Path(spec["artifacts_dir"])
    assert kwargs["train_steps"] == 2 and kwargs["batch_size"] == 3 and kwargs["seed"] == 12
    assert kwargs["stop_requested"] is stopping
    assert kwargs["device"] == training.section("training")["device"]
    assert summary["policy_id"] == "selected-model" and summary["policy_content_sha256"] == "a" * 64
    assert summary["dataset_id"] == "dataset-test"
    assert json.loads((output / "summary.json").read_text()) == summary
    assert json.loads((output / "progress.json").read_text())["phase"] == "completed"
    assert len(provider.calls) == 1  # fit does not load the actor again.


def test_rollout_prepare_fit_combination_uses_saved_rewards_and_actual_cpu_critics(worker_spec, monkeypatch):
    import torch
    from PIL import Image
    from trajectory_utils import save_trajectory_bundle
    from vla_rynn_iql import config as configuration, models, rewards, training as actor_training
    from vla_rynn_iql.config import load_train_config
    from vla_rynn_iql.data import load_manifest, prepare_dataset
    from vla_rynn_iql.io import sha256_file

    spec, directory, simulator, provider, _, _ = worker_spec
    validate_cuda = configuration._cuda_device
    # Ordinary actor training requires CUDA; this actor-free integration uses CPU.
    monkeypatch.setattr(configuration, "_cuda_device", lambda value, name:
        None if value == "cpu" and name == "training.device" else validate_cuda(value, name))
    training_path = Path(spec["training_config"])
    raw = yaml.safe_load(training_path.read_text())
    norm = directory / "normalization"
    norm.mkdir()
    stats = lambda size: {"q01": [-1.] * size, "q99": [1.] * size}
    (norm / "dataset_statistics.json").write_text(json.dumps({raw["model"]["stats_key"]: {
        "action": stats(7), "proprio": stats(8)}}))
    raw["model"].update(base_checkpoint=str(norm), base_revision=None)
    raw["training"]["device"] = "cpu"
    raw["iql"]["critic_image_size"] = 16
    raw["data"].update(validation_fraction=0., task_ids=[spec["task_snapshot"]["task_id"]])
    raw["reward"].update(source="sparse", rynnvalue=False, gamma=.8, accumulate_primitive_steps=False)
    source = directory / "training-source" / "runs" / "train-only"
    raw["paths"]["dataset_sources"] = [str(source)]
    train_simulator = FakeSimulator()
    recorder, result = runner.collect_episode(simulator=train_simulator, provider=FakeProvider(),
        env=train_simulator.env, initial_state=np.asarray([1., 2.]), prompt="place the bowl",
        settings=spec["rollout"], stop_requested=lambda: False)
    record = {"id": "train-only", "project_id": raw["data"]["project_id"], "kind": "original",
        "root_session_id": "train-only", "parent_session_id": None, "resume_step": None,
        "task_id": spec["task_snapshot"]["task_id"], "task": "place the bowl", "task_name": "task",
        "status": "COMPLETED", "error": None, "control_mode": "policy", "open_loop_steps": 4,
        "success": result.success}
    save_trajectory_bundle(recorder, source / "episodes/episode_000/trajectory", record,
        save_observations=True, create_plot=False, save_metadata_json=False)
    (source / "run.json").write_text(json.dumps(record))
    training_path.write_text(yaml.safe_dump(raw))
    training = load_train_config(training_path)
    prepared = prepare_dataset(training)
    saved = rewards.load_reward_index(training)  # Fixture creation, before the diagnostic begins.
    reward_manifest = Path(saved["episodes"][0]["reward_path"]).parent / "reward_manifest.json"
    raw["reward"].update(manifest_path=str(reward_manifest), manifest_sha256=sha256_file(reward_manifest),
                         version_id="saved-test-reward")
    training_path.write_text(yaml.safe_dump(raw))
    protected = [training_path, prepared.manifest, reward_manifest,
                 Path(saved["episodes"][0]["reward_path"]), source / "episodes/episode_000/trajectory.npz",
                 source / "episodes/episode_000/trajectory_observations.npz"]
    before = {path: sha256_file(path) for path in protected}

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not generate rewards or load/train an actor")

    monkeypatch.setattr(rewards, "materialize_reward_manifest", forbidden)
    monkeypatch.setattr(models, "model_backend", forbidden)
    monkeypatch.setattr(actor_training, "train", forbidden)
    monkeypatch.setenv("MPLCONFIGDIR", str(directory / "matplotlib"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        probe_path = runner.rollout(spec, directory)
        summary = runner.fit(spec, directory)
    finally:
        torch.set_num_threads(previous_threads)
    assert provider.runtime.seeds == [7] and provider.unloaded is True
    assert simulator.env.steps == 11 and len(provider.calls) == 1
    assert summary["actor_updated"] is False and summary["reward_generated"] is False
    assert summary["train_steps"] == 2 and summary["batch_size"] == 3
    assert summary["run_ids"]["train"] == ["train-only"]
    assert summary["run_ids"]["rollout"] == [f"{spec['job_id']}_rollout"]
    artifacts = Path(spec["artifacts_dir"])
    assert set(summary["artifacts"]) == {"summary.json", "qv.csv", "qv.npz", "qv.png", "training.png", "metrics.jsonl", "critic.pt"}
    for name in summary["artifacts"]:
        assert (artifacts / name).stat().st_size > 0
    with Image.open(artifacts / "qv.png") as image:
        assert np.asarray(image).std() > 0
    with np.load(artifacts / "qv.npz", allow_pickle=False) as arrays:
        rollout_rows = arrays["split"] == "rollout"
        assert rollout_rows.any() and (arrays["split"] == "train").any()
        assert np.isfinite(arrays["q1"]).all() and np.isfinite(arrays["v"]).all()
        assert np.isnan(arrays["reward"][rollout_rows]).all()
        assert np.isnan(arrays["observed_return"][rollout_rows]).all()
    checkpoint = torch.load(artifacts / "critic.pt", weights_only=False)
    assert set(checkpoint["critic"]) == {"model", "q_optimizer", "value_optimizer"}
    assert before == {path: sha256_file(path) for path in before}
    assert not list((directory / "probe").rglob("reward_manifest.json"))
    assert not list(directory.rglob("policy.yaml"))
    assert not list(directory.rglob("catalog.sqlite3"))
    probe_manifest = load_manifest(load_train_config(probe_path))
    assert probe_manifest["episode_count"] == 1
    assert probe_manifest["episodes"][0]["observation_orientation"] == "libero_raw"


@pytest.mark.parametrize("change", ["job", "outside", "relative", "symlink", "extra", "bool", "critic"])
def test_spec_rejects_unsafe_paths_and_invalid_configuration(worker_spec, change, tmp_path):
    spec, directory, *_ = worker_spec
    if change == "job":
        spec["job_id"] = "not-the-directory"
    elif change == "outside":
        spec["artifacts_dir"] = str(tmp_path / "elsewhere")
    elif change == "relative":
        spec["training_config"] = "training.yaml"
    elif change == "symlink":
        link = directory / "linked.yaml"
        link.symlink_to(spec["training_config"])
        spec["training_config"] = str(link)
    elif change == "extra":
        spec["unknown"] = True
    elif change == "bool":
        spec["rollout"]["seed"] = True
    else:
        spec["critic"]["batch_size"] = 0
    path = directory / "effective_config.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        runner.load_spec(path)


def test_spec_round_trip(worker_spec):
    spec, directory, *_ = worker_spec
    path = directory / "effective_config.yaml"
    path.write_text(yaml.safe_dump(spec))
    assert runner.load_spec(path) == spec


def test_catalog_rejects_changed_snapshot_content(tmp_path):
    from backend.app.policies.catalog import PolicyCatalog
    from backend.app.policies.snapshots import snapshot
    from test_pi05_policy import checkpoint

    settings, _ = checkpoint(tmp_path)
    catalog = PolicyCatalog(tmp_path / "registry", base_models={settings["base_id"]: settings})
    selected = snapshot(catalog.select(settings["base_id"]))
    runner._catalog({"policy_snapshot": selected}, tmp_path)
    selected["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        runner._catalog({"policy_snapshot": selected}, tmp_path)


def test_cli_cancellation_returns_130_and_invokes_one_phase(worker_spec, monkeypatch):
    spec, directory, *_ = worker_spec
    script = Path(__file__).resolve().parents[2] / "scripts/run_critic_diagnostic.py"
    definition = importlib.util.spec_from_file_location("critic_diagnostic_cli_test", script)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    config = directory / "effective_config.yaml"
    calls = []
    monkeypatch.setattr(module, "load_spec", lambda path: spec)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module, "rollout", lambda *a, **k: pytest.fail("wrong phase"))
    def canceled(*args, **kwargs):
        calls.append((args, kwargs))
        raise InterruptedError("canceled")
    monkeypatch.setattr(module, "fit", canceled)
    monkeypatch.setattr(module.sys, "argv", [str(script), "--config", str(config), "--phase", "fit"])
    assert module.main() == 130
    assert len(calls) == 1 and calls[0][0] == (spec, directory)
