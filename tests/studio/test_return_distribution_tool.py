"""Value Flows return distribution tool: selection reporting, rollout capture, analysis, queue and API."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config import LoadedConfig, load_train_config
from core.io import sha256_file
from studio.api import tools
from studio.policies.pi05 import Pi05PolicyProvider
from studio.policies.registry import configured_models
from studio.policies.router import PolicyProvider
from studio.services.return_distribution_service import ReturnDistributionService, flow_critic_policy
from studio.tools import return_distribution_runner as runner
from train.methods.base import build_algorithm, save_critic
from train.methods.selectors import CRITIC_FILE

QUANTILE = {"q01": [-1.0] * 8, "q99": [1.0] * 8, "codec": "openpi_quantile"}
OBSERVATION = {"agentview_image": np.zeros((4, 4, 3), np.uint8),
               "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
               "robot0_eef_pos": [0, 0, 0], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0, 0]}


def sampling_provider(monkeypatch, reply, selector=True):
    selected = SimpleNamespace(family="pi05", policy_id="vf", label="VF", content_sha256="a" * 64,
                               selector={"kind": "argmax", "num_samples": 3} if selector else None,
                               settings=configured_models()["pi05-liberox-base"])
    provider = Pi05PolicyProvider(None, None, SimpleNamespace(select=lambda _: selected))
    provider.process = SimpleNamespace(poll=lambda: None)
    provider.current_policy_entry = SimpleNamespace(content_sha256=selected.content_sha256)
    provider.load(10, "vf", expected_content_sha256=selected.content_sha256)
    captured = {}
    def send(request):
        captured.update(request)
        return reply
    monkeypatch.setattr(provider, "_send", send)
    unloaded = []
    monkeypatch.setattr(provider, "unload", lambda: unloaded.append(True))
    return provider, captured, unloaded


def selection_reply(chosen=1):
    candidates = np.random.default_rng(0).uniform(-1, 1, (3, 10, 7)).astype(np.float32)
    return {"actions": candidates[chosen].tolist(),
            "selection": {"candidates": candidates.tolist(), "scores": [0.1, 0.5, 0.2], "chosen": chosen}}


def test_provider_reports_every_candidate_and_its_score(monkeypatch):
    reply = selection_reply()
    provider, captured, _ = sampling_provider(monkeypatch, reply)
    actions, selection = provider.predict_with_selection(OBSERVATION, "task")
    assert captured["selection_details"] is True
    np.testing.assert_array_equal(actions, selection["candidates"][1])
    assert selection["scores"].tolist() == pytest.approx([0.1, 0.5, 0.2]) and selection["chosen"] == 1
    # Ordinary inference never asks the worker for the candidates.
    provider.predict(OBSERVATION, "task")
    assert "selection_details" not in captured or captured == {**captured}


@pytest.mark.parametrize("change", ["chosen_mismatch", "count", "missing"])
def test_inconsistent_selection_is_rejected_and_unloads(monkeypatch, change):
    reply = selection_reply()
    if change == "chosen_mismatch":
        reply["selection"]["chosen"] = 0
    elif change == "count":
        reply["selection"]["scores"] = [0.1, 0.5]
    else:
        reply.pop("selection")
    provider, _, unloaded = sampling_provider(monkeypatch, reply)
    with pytest.raises(ValueError, match="selection"):
        provider.predict_with_selection(OBSERVATION, "task")
    assert unloaded


def test_policies_without_a_critic_cannot_report_selections(monkeypatch):
    provider, captured, _ = sampling_provider(monkeypatch, selection_reply(), selector=False)
    with pytest.raises(ValueError, match="no critic selection"):
        provider.predict_with_selection(OBSERVATION, "task")
    assert captured == {}
    router = PolicyProvider.__new__(PolicyProvider)
    router.family, router.provider = "vla_adapter", SimpleNamespace(predict=lambda *_: None)
    with pytest.raises(ValueError, match="do not report critic selections"):
        router.predict_with_selection(OBSERVATION, "task", ())


class FakeRecorder:
    def __init__(self, steps=12):
        self.inference_query_steps, self.inference_action_chunks = [], []
        self.action_count, self.steps = 0, steps

    def arrays(self):
        rng = np.random.default_rng(1)
        done = np.zeros(self.steps, dtype=bool)
        done[8:] = True
        return {"eef_position": rng.normal(size=(self.steps + 1, 3)).astype(np.float32),
                "eef_axis_angle": rng.normal(size=(self.steps + 1, 3)).astype(np.float32),
                "gripper_qpos": rng.normal(size=(self.steps + 1, 2)).astype(np.float32), "done": done}

    def observation_arrays(self):
        frames = np.zeros((self.steps + 1, 16, 16, 3), np.uint8)
        frames[:, :8] = np.arange(self.steps + 1, dtype=np.uint8)[:, None, None, None] * 10
        return {"agentview_image": frames, "wrist_image": frames.copy()}


def spec(tmp_path, samples=16):
    directory = tmp_path / "returns_job"
    directory.mkdir(exist_ok=True)
    return {"schema_version": 1, "job_id": directory.name,
            "policy_snapshot": {"policy_id": "vf", "content_sha256": "c" * 64}, "task_snapshot": {"task_id": "t"},
            "rollout": {"max_steps": 12, "open_loop_steps": 10, "seed": 3, "init_state_index": 0,
                        "control_hz": 20, "success_consecutive_steps": 3},
            "analysis": {"return_samples": samples, "seed": 3},
            "artifacts_dir": str((directory / "artifacts").resolve())}, directory


def fake_simulate(recorder, replies):
    def simulate(spec, directory, *, stop_requested, query):
        Path(spec["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
        provider = SimpleNamespace(predict_with_selection=lambda *_: replies.pop(0))
        for step in (0, 10):
            chunk = query(provider, {}, "task")
            recorder.inference_query_steps.append(step)
            recorder.inference_action_chunks.append(np.asarray(chunk, dtype=np.float32))
        recorder.action_count = recorder.steps
        return recorder, SimpleNamespace(success=True), {}
    return simulate


def selections(chosen=(1, 2)):
    rng = np.random.default_rng(2)
    result = []
    for index in chosen:
        candidates = rng.uniform(-1.2, 1.2, (3, 10, 7)).astype(np.float32)
        result.append((candidates[index], {"candidates": candidates, "scores": np.asarray([-1., -.5, -2.], np.float32),
                                           "chosen": index}))
    return result


def test_rollout_keeps_every_query_selection_and_matches_recorded_chunks(tmp_path, monkeypatch):
    value, directory = spec(tmp_path)
    replies = selections()
    expected = [item[1] for item in replies]
    monkeypatch.setattr(runner, "simulate", fake_simulate(FakeRecorder(), replies))
    runner.rollout(value, directory)
    with np.load(directory / "rollout" / "selection.npz") as saved:
        assert saved["query_step"].tolist() == [0, 10] and saved["chosen"].tolist() == [1, 2]
        np.testing.assert_array_equal(saved["candidates"][1], expected[1]["candidates"])
    assert json.loads((directory / "artifacts" / "progress.json").read_text())["phase"] == "rollout_complete"
    tampered = selections()
    tampered[0] = (tampered[0][1]["candidates"][0], tampered[0][1])
    monkeypatch.setattr(runner, "simulate", fake_simulate(FakeRecorder(), tampered))
    with pytest.raises(ValueError, match="differ from the reported critic selection"):
        runner.rollout(value, directory)


def test_analysis_writes_distributions_reference_and_videos(tmp_path, monkeypatch):
    value, directory = spec(tmp_path)
    monkeypatch.setattr(runner, "simulate", fake_simulate(FakeRecorder(), selections()))
    runner.rollout(value, directory)
    raw = load_train_config(method="vf", family="pi05").raw
    raw["reward"].update(source="sparse")
    raw["data"]["success_consecutive_steps"] = 3  # the critic's own success rule defines its rewards
    raw["vf"]["critic_image_size"] = 16
    config = LoadedConfig(tmp_path / "vf.yaml", raw)
    algorithm = build_algorithm(config, torch.device("cpu"))
    algorithm.bind_reward_range(-1.0, 0.0)
    save_critic(algorithm, config, tmp_path)
    critic = tmp_path / CRITIC_FILE
    entry = SimpleNamespace(artifacts={"critic": critic}, component_sha256={"critic": sha256_file(critic)},
                            settings=configured_models()["pi05-liberox-base"])
    monkeypatch.setattr(runner, "frozen_catalog", lambda *_: SimpleNamespace(entry=lambda _: entry))
    monkeypatch.setattr("core.model_config.replay_stats", lambda _: (QUANTILE, QUANTILE))
    summary = runner.analyze(value, directory)
    artifacts = directory / "artifacts"
    data = json.loads((artifacts / "returns.json").read_text())
    assert [query["step"] for query in data["queries"]] == [0, 10]
    assert len(data["queries"][0]["histograms"]) == 3 and len(data["edges"]) == 49
    assert all(sum(counts) == 16 for counts in data["queries"][1]["histograms"])
    # Sparse rewards: success confirmed at action 10 of 12 with γ = 0.99.
    gamma = config.section("reward")["gamma"]
    assert data["reference"]["available"]
    assert data["reference"]["values"] == pytest.approx([-sum(gamma ** k for k in range(10)), 0.0])
    with np.load(artifacts / "returns.npz") as saved:
        assert saved["returns"].shape == (2, 3, 16)
        low, high = saved["edges"][0], saved["edges"][-1]
        assert low <= saved["returns"].min() and saved["returns"].max() <= high
    for name in ("returns.mp4", "rollout.mp4"):
        assert (artifacts / name).stat().st_size > 0
    assert summary["queries"] == 2 and summary["candidates"] == 3 and summary["success"] is True
    assert summary["reward_config"]["source"] == "sparse" and summary["reference_available"]
    entry.component_sha256["critic"] = "0" * 64
    with pytest.raises(ValueError, match="missing or changed"):
        runner.analyze(value, directory)


class VfCatalog:
    def __init__(self):
        self.entries = {"vf": SimpleNamespace(algorithm="vf", is_base=False),
                        "ifql": SimpleNamespace(algorithm="ifql", is_base=False),
                        "base": SimpleNamespace(algorithm="bc", is_base=True)}

    def entry(self, policy_id):
        return self.entries[policy_id]

    def list_policies(self):
        return [{"policy_id": key, "algorithm": value.algorithm, "kind": "base" if value.is_base else "overlay"}
                for key, value in self.entries.items()]


@pytest.fixture
def tool(tmp_path, monkeypatch):
    from test_critic_tools import setup_jobs
    from test_evaluation_api import FakeManager
    from studio.services.offline_job_service import OfflineJobService

    jobs, _ = setup_jobs(tmp_path)
    jobs._new_job = OfflineJobService._new_job.__get__(jobs)
    jobs.manager = FakeManager(tmp_path)
    jobs.manager.policy_catalog = VfCatalog()

    def freeze(catalog, task_id, destination):
        destination.mkdir()
        return SimpleNamespace(paths=lambda _: (destination / "task.bddl", destination / "init"),
                               manifest={"task_revision_sha256": "f" * 64})

    monkeypatch.setattr("studio.services.task_registry.freeze_task", freeze)
    monkeypatch.setattr("studio.policies.snapshots.validate_snapshot", lambda value: value)
    monkeypatch.setattr(jobs, "preview_evaluation", lambda request: {
        "policy_snapshot": {"policy_id": request["policy_id"], "label": "VF", "content_sha256": "b" * 64},
        "task_snapshot": {"task_id": request["task_id"], "prompt": "pick"},
        "config": {"open_loop_steps": 10, "control_hz": 20, "success_streak": 5}})
    request = {"task_id": "LEVEL1::pick", "policy_id": "vf", "max_steps": 40, "seed": 2,
               "init_state_index": 1, "return_samples": 64}
    return ReturnDistributionService(jobs), jobs, request


def test_only_trained_flow_critic_policies_are_offered_and_accepted(tool):
    service, jobs, request = tool
    assert service.defaults() == {"return_samples": 256, "policy_ids": ["vf"]}
    assert flow_critic_policy({"algorithm": "vf", "kind": "overlay"})
    for policy_id in ("ifql", "base"):
        with pytest.raises(ValueError, match="Value Flows critic"):
            service.start({**request, "policy_id": policy_id})
    assert service.list()["jobs"] == []


def test_start_queues_rollout_then_cpu_analysis_with_a_frozen_spec(tool):
    service, jobs, request = tool
    job = service.start(request)
    assert job["kind"] == "return_distribution" and job["status"] == "QUEUED"
    _, stored = jobs._load_job(job["id"])
    assert [stage["id"] for stage in stored["stages"]] == ["rollout", "analyze"]
    assert all(stage["argv"][:3] == ["python", "-m", "studio.commands.return_distribution"]
               for stage in stored["stages"])
    assert job["parameters"]["requires_gpu"] is True and job["parameters"]["policy_label"] == "VF"
    config = jobs.jobs_root / job["id"] / "effective_config.yaml"
    value = runner.load_spec(config)
    assert value["rollout"] == {"max_steps": 40, "open_loop_steps": 10, "seed": 2, "init_state_index": 1,
                                "control_hz": 20, "success_consecutive_steps": 5}
    assert value["analysis"] == {"return_samples": 64, "seed": 2}
    with pytest.raises(ValueError):
        service.start({**request, "return_samples": 8})
    raw = yaml.safe_load(config.read_text())
    raw["analysis"]["return_samples"] = 8
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="analysis"):
        runner.load_spec(config)


def test_api_routes_for_return_distributions(tool):
    service, jobs, request = tool
    app = FastAPI()
    app.state.return_distribution_service = service
    app.include_router(tools.returns_router)
    with TestClient(app) as client:
        assert client.get("/api/tools/returns/defaults").json()["policy_ids"] == ["vf"]
        assert client.post("/api/tools/returns", json={**request, "reward_gamma": .5}).status_code == 422
        created = client.post("/api/tools/returns", json=request)
        assert created.status_code == 201, created.text
        identifier = created.json()["id"]
        assert client.get("/api/tools/returns").json()["jobs"][0]["id"] == identifier
        artifacts = jobs.jobs_root / identifier / "artifacts"
        artifacts.mkdir()
        (artifacts / "returns.json").write_text(json.dumps({"schema_version": 1}))
        (artifacts / "summary.json").write_text(json.dumps({"queries": 2}))
        detail = client.get(f"/api/tools/returns/{identifier}").json()
        assert detail["return_summary"] == {"queries": 2}
        assert detail["artifacts"]["returns.json"] == f"/api/tools/returns/{identifier}/artifacts/returns.json"
        assert client.get(f"/api/tools/returns/{identifier}/artifacts/job.json").status_code == 422
        assert client.post(f"/api/tools/returns/{identifier}/stop").status_code == 200
