from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from backend.app.api import evaluations
from backend.app.api.models import DeleteEvaluationRequest, EvaluationRequest
from backend.app.evaluation.batch import load_effective_config
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.storage.database import SCHEMA_VERSION, connect, migrate


class FakeTaskCatalog:
    def __init__(self, root: Path):
        self.bddl = root / "task.bddl"
        self.init = root / "task.init"
        self.bddl.write_text("task", encoding="utf-8")
        self.init.write_bytes(b"states")

    def metadata(self, task_id: str):
        if task_id != "LEVEL1::pick":
            raise ValueError("unknown task")
        return {
            "task_id": task_id,
            "level": "LEVEL1",
            "task_name": "pick_the_bowl",
            "prompt": "pick the bowl",
            "init_state_count": 3,
        }

    def paths(self, task_id: str):
        self.metadata(task_id)
        return self.bddl, self.init

    def initial_state_count(self, task_id: str):
        self.metadata(task_id)
        return 3


class FakePolicyCatalog:
    def refresh(self):
        pass

    def entry(self, policy_id: str):
        if policy_id != "base":
            raise ValueError("unknown policy")
        return SimpleNamespace(
            policy_id="base",
            label="Base policy",
            base_checkpoint="VLA-Adapter/LIBERO-Object-Pro",
            stats_key="libero_object",
            manifest=None,
            action_head=None,
            proprio_projector=None,
            training_step=None,
            compatibility_sha256=None,
        )


class FakeManager:
    def __init__(self, root: Path):
        self.catalog = FakeTaskCatalog(root)
        self.policy_catalog = FakePolicyCatalog()
        self.eval_config = SimpleNamespace(control_hz=20)
        self.active_session_id = None
        self.draft = None
        self.provider = SimpleNamespace(unload=lambda: None)

    def controller_status(self):
        return {"state": "READY"}


def make_service(tmp_path: Path) -> OfflineJobService:
    project_root = tmp_path / "dataset-root" / "projects" / "test"
    config = SimpleNamespace(
        project_root=project_root,
        catalog_path=tmp_path / "dataset-root" / "catalog.sqlite3",
        project_id="test",
        offline_rl_root=tmp_path,
        train_environment="vla-liberox",
        reward_environment="rynnvalue-reward",
        tensorboard_host="127.0.0.1",
        tensorboard_port=6006,
    )
    datasets = SimpleNamespace()
    return OfflineJobService(config, FakeManager(tmp_path), datasets)


def evaluation_request(**changes):
    value = {
        "task_id": "LEVEL1::pick",
        "policy_id": "base",
        "trials": 7,
        "max_steps": 100,
        "open_loop_steps": 4,
        "realtime": True,
        "init_state_indices": None,
        "base_seed": 11,
        "seed_count": None,
        "schedule_seed": 19,
    }
    value.update(changes)
    return value


def test_evaluation_request_is_strict_and_validates_state_pool():
    valid = EvaluationRequest.model_validate(evaluation_request())
    assert valid.trials == 7
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(evaluation_request(trials=True))
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(evaluation_request(init_state_indices=[]))
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(
            evaluation_request(init_state_indices=[0, 0])
        )
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate({**evaluation_request(), "command": "unsafe"})


def test_preview_uses_task_bounds_and_returns_frontend_shape(tmp_path: Path):
    service = make_service(tmp_path)
    first = service.preview_evaluation(evaluation_request())
    second = service.preview_evaluation(evaluation_request())
    assert first["schedule"] == second["schedule"]
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["config"]["task_id"] == "LEVEL1::pick"
    assert first["config"]["policy_id"] == "base"
    assert first["config"]["init_state_indices"] == [0, 1, 2]
    assert first["config"]["seed_count"] == 3
    assert first["config"]["control_hz"] == 20
    assert len(first["schedule"]) == 7
    assert set(first["schedule"][0]) == {
        "trial_index", "init_state_index", "seed",
    }
    assert first["estimated_duration_seconds"] == 35.0
    with pytest.raises(ValueError, match="invalid values"):
        service.preview_evaluation(
            evaluation_request(init_state_indices=[0, 3])
        )
    with pytest.raises(ValueError, match="exceeds"):
        service.preview_evaluation(
            evaluation_request(base_seed=2147483647, seed_count=2)
        )


def test_launch_history_mapping_and_safe_persistent_delete(tmp_path: Path):
    service = make_service(tmp_path)
    captured = {}

    def fake_new_job(**kwargs):
        captured.update(kwargs)
        return {
            "id": kwargs["config_path"].parent.name,
            "kind": "evaluation",
            "status": "STARTING",
            "dataset_id": None,
            "parameters": kwargs["parameters"],
            "evaluation_summary": {"attempted_trials": 0},
        }

    service._new_job = fake_new_job
    preview = service.preview_evaluation(evaluation_request())
    job = service._launch_evaluation(preview)
    evaluation_id = job["id"]
    effective = load_effective_config(captured["config_path"])
    assert effective["evaluation_id"] == evaluation_id
    assert effective["schedule"] == preview["schedule"]
    assert effective["config"]["control_hz"] == 20
    result_dir = Path(captured["output_path"])
    assert [path.name for path in result_dir.iterdir()] == ["evaluation.json"]
    live_start = service._public_job({
        "id": evaluation_id,
        "kind": "evaluation",
        "status": "RUNNING",
        "output_path": str(result_dir),
    })["evaluation_summary"]
    assert live_start["current_trial"] == 1
    assert live_start["init_state_index"] == preview["schedule"][0]["init_state_index"]
    assert live_start["seed"] == preview["schedule"][0]["seed"]

    result_path = result_dir / "evaluation.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({
        "status": "COMPLETED",
        "started_at": result["created_at"],
        "completed_at": result["created_at"],
        "task": result.pop("task_snapshot"),
        "policy": result.pop("policy_snapshot"),
        "trials": [{
            "trial_index": 0,
            "init_state_index": 1,
            "seed": 11,
            "success": True,
            "error": None,
            "executed_steps": 100,
            "first_success_step": 40,
            "max_done_streak": 8,
            "final_done": False,
            "policy_queries": 25,
            "inference_latency_mean_ms": 12.5,
            "measured_control_hz": 20.0,
            "deadline_misses": 0,
            "wall_seconds": 5.0,
        }],
        "aggregate": {
            "scheduled_trials": 7,
            "attempted": 1,
            "successes": 1,
            "failures": 0,
            "errors": 0,
            "success_rate": 1.0,
            "wilson_95": [0.2, 1.0],
            "completion_coverage": 1 / 7,
            "by_init_state": {"1": {
                "attempted": 1, "successes": 1, "failures": 0,
                "errors": 0, "success_rate": 1.0,
            }},
            "by_seed": {},
            "by_combination": {},
            "first_success_step": {"mean": 40.0},
            "policy_queries": {"mean": 25.0},
            "inference_latency_ms": {"mean": 12.5},
            "control_hz": {"mean": 20.0},
            "wall_seconds": {"mean": 5.0},
        },
        "timing": {
            "model_load_seconds": 3.0,
            "wall_seconds": 5.0,
            "simulated_seconds": 5.0,
        },
    })
    result_path.write_text(json.dumps(result), encoding="utf-8")

    public = service.get_evaluation(evaluation_id)
    assert public["task_id"] == "LEVEL1::pick"
    assert public["policy_id"] == "base"
    assert public["aggregate"]["total_trials"] == 7
    assert public["aggregate"]["attempted_trials"] == 1
    assert public["aggregate"]["wilson_lower"] == 0.2
    assert public["aggregate"]["by_init_state"]["1"]["trials"] == 1
    assert public["trials"][0]["steps"] == 100
    assert public["trials"][0]["inference_latency_ms"] == 12.5
    assert public["wall_time_seconds"] == 5.0
    public_job = service._public_job({
        "id": evaluation_id,
        "kind": "evaluation",
        "status": "COMPLETED",
        "output_path": str(result_dir),
    })
    live = public_job["evaluation_summary"]
    assert live["attempted_trials"] == 1
    assert live["progress_percent"] == pytest.approx(100 / 7)
    assert live["estimated_remaining_seconds"] == 30.0
    assert live["init_state_index"] == 1
    assert live["seed"] == 11
    assert service.list_evaluations(task_id="LEVEL1::pick")[0]["id"] == evaluation_id

    with pytest.raises(ValueError, match="exactly match"):
        service.delete_evaluation(evaluation_id, "wrong")
    deleted = service.delete_evaluation(evaluation_id, evaluation_id)
    assert deleted["deleted"] == evaluation_id
    assert not result_dir.exists()
    assert not captured["config_path"].parent.exists()
    assert service.list_evaluations() == []
    with pytest.raises(KeyError):
        service.get_evaluation(evaluation_id)


def test_active_evaluation_cannot_be_deleted(tmp_path: Path):
    service = make_service(tmp_path)
    service._new_job = lambda **kwargs: {
        "id": kwargs["config_path"].parent.name,
        "kind": "evaluation",
        "status": "STARTING",
    }
    job = service._launch_evaluation(
        service.preview_evaluation(evaluation_request(trials=1))
    )
    with pytest.raises(Exception, match="must be stopped"):
        service.delete_evaluation(job["id"], job["id"])


def test_evaluation_api_routes_delegate_and_translate_errors():
    calls = []

    class Stub:
        def preview_evaluation(self, body):
            calls.append(("preview", body))
            return {"schedule": []}

        def start_evaluation(self, body):
            calls.append(("start", body))
            return {"id": "eval", "kind": "evaluation"}

        def list_evaluations(self, **filters):
            calls.append(("list", filters))
            return []

        def get_evaluation(self, evaluation_id):
            if evaluation_id == "missing":
                raise KeyError(evaluation_id)
            return {"id": evaluation_id}

        def stop_evaluation(self, evaluation_id):
            return {"id": evaluation_id, "status": "STOPPING"}

        def delete_evaluation(self, evaluation_id, confirmation):
            calls.append(("delete", (evaluation_id, confirmation)))
            return {"deleted": evaluation_id}

    app = FastAPI()
    app.state.offline_job_service = Stub()
    request = Request({"type": "http", "app": app})
    body = EvaluationRequest.model_validate(evaluation_request())

    async def exercise():
        assert await evaluations.preview(body, request) == {"schedule": []}
        assert (await evaluations.create(body, request))["kind"] == "evaluation"
        assert await evaluations.history(
            request, "task", "base", "COMPLETED", "2026-01-01", "2026-12-31"
        ) == []
        assert (await evaluations.detail("eval", request))["id"] == "eval"
        assert (await evaluations.stop("eval", request))["status"] == "STOPPING"
        assert await evaluations.delete(
            "eval", DeleteEvaluationRequest(confirm_evaluation_id="eval"), request
        ) == {"deleted": "eval"}
        with pytest.raises(HTTPException) as missing:
            await evaluations.detail("missing", request)
        assert missing.value.status_code == 404

    asyncio.run(exercise())
    assert calls[0][0] == "preview"
    assert calls[-1] == ("delete", ("eval", "eval"))


def test_catalog_schema_indexes_evaluations(tmp_path: Path):
    database = tmp_path / "catalog.sqlite3"
    migrate(database)
    with connect(database) as connection:
        version = connection.execute("SELECT version FROM schema_info").fetchone()[0]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_runs'"
        ).fetchone()
    assert version == SCHEMA_VERSION == 4
    assert table[0] == "evaluation_runs"


def test_catalog_migrates_v2_to_evaluation_index(tmp_path: Path):
    database = tmp_path / "catalog-v2.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_info (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_info(version) VALUES (2)")
    migrate(database)
    with connect(database) as connection:
        version = connection.execute("SELECT version FROM schema_info").fetchone()[0]
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(evaluation_runs)")
        }
    assert version == SCHEMA_VERSION == 4
    assert {"id", "task_id", "policy_id", "result_path"} <= columns
