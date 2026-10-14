"""Platform/model boundary tests with an opaque substitute family, never real weights."""
from copy import deepcopy
import io
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

import test_model as cli
from backend.app.api.models import CreateTrainingDatasetRequest, DraftRequest, EvaluationRequest, TrainingRunRequest


OPTIONS = ["--model", "family-variant"]


class Platform:
    def __init__(self):
        self.calls = []
        self.created_jobs = set()
        self.deleted = []
        self.base = {"policy_id": "family-variant", "kind": "base", "family": "family",
                     "inference": {"open_loop_steps": 8, "max_open_loop_steps": 8},
                     "io": {"replay_horizon": 8}, "model_config": {}, "content_sha256": "a" * 64,
                     "base_checkpoint": "/weights/base", "base_revision": "a" * 64, "stats_key": "stats"}
        self.child = {**self.base, "policy_id": "exact-child", "kind": "overlay",
                      "parent_model_id": self.base["policy_id"], "algorithm": "bc", "training_step": 2,
                      "content_sha256": "b" * 64, "components": [{"name": "actor"}],
                      "manifest": {"dataset_sha256": "dataset-hash"}}
        self.record = {"id": "owned-run", "work_job_id": "simulation", "status": "COMPLETED", "error": None,
                       "policy_id": self.base["policy_id"], "task_id": cli.TASK_ID,
                       "action_count": 40, "state_count": 41, "policy_queries": 5, "trajectory": "trajectory.npz",
                       "policy_content_sha256": self.base["content_sha256"],
                       "artifacts": {"episodes/episode_000/trajectory_observations.npz": "observations.npz"}}
        self.dataset = {"id": "owned-dataset", "task_id": cli.TASK_ID, "integrity_status": "HEALTHY",
                        "members": [{"run_id": self.record["id"]}]}
        self.defaults = {"algorithm": "bc", "model": {"model_base_id": self.base["policy_id"], "model_family": "family"}}
        self.jobs = {identifier: {"id": identifier, "kind": identifier, "status": "COMPLETED"}
                     for identifier in ("simulation", "dataset", "training", "evaluation")}
        self.jobs["dataset"]["result"] = {"dataset_id": self.dataset["id"]}
        self.jobs["training"].update({
            "parameters": {"model": self.base},
            "metrics": {"step": 2, "actor_loss": 0.0, "actor_grad_norm": 0.0, "actor_learning_rate": 1e-5},
            "training_summary": {"status": "completed", "steps": 2, "algorithm": "bc",
                                 "resumed_from_step": 0, "micro_batch_size": 1, "transitions_processed": 2,
                                 "dataset_sha256": "dataset-hash", "policy_overlay": "/registry/exact-child/policy.yaml"},
        })
        self.evaluation = {"aggregate": {"errors": 0, "completed_trials": 1, "successes": 0},
                           "policy_snapshot": self.child, "trials": [{"steps": 40}]}

    def request(self, method, path, payload=None):
        self.calls.append((method, path, deepcopy(payload)))
        if method == "GET":
            values = {"/api/models/family-variant": self.base, "/api/models/exact-child": self.child,
                      "/api/sessions/owned-run": self.record, "/api/training-datasets/owned-dataset": self.dataset,
                      "/api/evaluations/evaluation": self.evaluation}
            if path in values:
                return deepcopy(values[path])
            if path.startswith("/api/training/defaults?"):
                assert parse_qs(urlsplit(path).query) == {
                    "algorithm": ["bc"], "model_base_id": [self.base["policy_id"]], "model_family": [self.base["family"]]}
                return deepcopy(self.defaults)
            if "/logs?" in path:
                return {"text": "", "next_offset": 0}
            if path.startswith("/api/jobs/"):
                return deepcopy(self.jobs[path.rsplit("/", 1)[1]])
        if method == "POST":
            if path == "/api/sessions":
                DraftRequest.model_validate(payload)
                self.created_jobs.add("simulation")
                return deepcopy(self.record)
            kinds = {"/api/training-datasets": ("dataset", CreateTrainingDatasetRequest),
                     "/api/training-runs": ("training", TrainingRunRequest),
                     "/api/evaluations": ("evaluation", EvaluationRequest)}
            if path in kinds:
                kind, schema = kinds[path]
                schema.model_validate(payload)
                self.created_jobs.add(kind)
                return deepcopy(self.jobs[kind])
            if path.startswith("/api/jobs/") and path.endswith("/stop"):
                identifier = path.split("/")[-2]
                self.jobs[identifier]["status"] = "CANCELED"
                return deepcopy(self.jobs[identifier])
        if method == "PATCH" and path == "/api/datasets/runs/owned-run/labels":
            assert payload == {"is_test": True}
            return payload
        if method == "DELETE":
            identifier = path.rsplit("/", 1)[-1]
            if path.startswith("/api/jobs/"):
                assert identifier in self.created_jobs
                assert payload == {"confirm_job_id": identifier, "delete_exported_models": identifier == "training"}
            elif path == "/api/training-datasets/owned-dataset":
                assert "dataset" in self.created_jobs and payload == {"confirm_dataset_id": identifier}
            elif path == "/api/sessions/owned-run":
                assert "simulation" in self.created_jobs and payload == {"confirm_session_id": identifier}
            else:
                raise AssertionError((method, path))
            self.deleted.append(path)
            return {"deleted": identifier}
        raise AssertionError((method, path))


def test_fixed_training_settings_and_model_owned_contract():
    api = Platform()
    base, parameters = cli.prepare(api, cli.parser().parse_args(OPTIONS))
    assert parameters == {
        "algorithm": "bc", "model_family": "family", "model_base_id": api.base["policy_id"],
        "train_steps": 2, "micro_batch_size": 1, "gradient_accumulation_steps": 1,
        "checkpoint_interval": 2, "actor_lr_warmup_steps": 0, "seed": 0, "resume_checkpoint": None,
        "console_interval_steps": 1, "tensorboard": False, "wandb_enabled": False,
    }
    base["inference"]["open_loop_steps"] = 4
    assert base["io"]["replay_horizon"] == 8
    assert cli.simulation_request(base) == {
        "policy_id": api.base["policy_id"], "task_id": cli.TASK_ID, "trials": 1,
        "max_steps": 40, "open_loop_steps": 4, "realtime": False, "init_state_indices": [0],
        "base_seed": 0, "seed_count": 1, "schedule_seed": 0,
    }
    assert all(method == "GET" for method, *_ in api.calls)


@pytest.mark.parametrize("flag", ["--steps", "--max-steps", "--seed", "--init-state", "--method", "--task",
                                   "--dataset", "--url", "--no-wait", "--dry-run"])
def test_only_model_is_configurable(flag):
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args([*OPTIONS, flag, "1"])
    assert error.value.code == 2


@pytest.mark.parametrize("mutation", [
    lambda api: api.base.update(policy_id="different"),
    lambda api: api.base.update(kind="child"),
    lambda api: api.defaults["model"].update(model_base_id="different"),
    lambda api: api.defaults["model"].update(model_family="different"),
    lambda api: api.defaults.update(algorithm="iql"),
])
def test_preflight_never_falls_back_to_another_model(mutation):
    api = Platform()
    mutation(api)
    with pytest.raises(ValueError):
        cli.prepare(api, cli.parser().parse_args(OPTIONS))
    assert all(method == "GET" for method, *_ in api.calls)


def test_one_invocation_checks_entire_pipeline_via_queue(monkeypatch, capsys):
    api = Platform()
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == 0
    posts = [(path, body) for method, path, body in api.calls if method == "POST"]
    assert [path for path, _ in posts] == ["/api/sessions", "/api/training-datasets", "/api/training-runs", "/api/evaluations"]
    assert posts[0][1]["policy_id"] == "family-variant"
    assert posts[1][1]["selection"] == {"mode": "manual", "run_ids": ["owned-run"]}
    assert posts[1][1]["validation_fraction"] == 0.0
    assert posts[2][1]["dataset_id"] == "owned-dataset"
    assert posts[3][1]["policy_id"] == "exact-child"
    assert all("/draft" not in path for _, path, _ in api.calls)
    assert "PASS: family-variant -> exact-child" in capsys.readouterr().out
    assert api.deleted == ["/api/jobs/evaluation", "/api/jobs/training",
        "/api/training-datasets/owned-dataset", "/api/jobs/dataset",
        "/api/sessions/owned-run", "/api/jobs/simulation"]


@pytest.mark.parametrize("stage,submissions", [("simulation", 1), ("dataset", 2), ("training", 3), ("evaluation", 4)])
@pytest.mark.parametrize("status,expected", [("FAILED", 1), ("CANCELED", 130)])
def test_failure_or_cancellation_stops_later_stages(monkeypatch, stage, submissions, status, expected):
    api = Platform()
    api.jobs[stage]["status"] = status
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == expected
    assert sum(method == "POST" for method, *_ in api.calls) == submissions
    assert {path.rsplit("/", 1)[-1] for path in api.deleted if path.startswith("/api/jobs/")} == api.created_jobs


@pytest.mark.parametrize("mutation,submissions", [
    (lambda api: api.record.update(action_count=1), 1),
    (lambda api: api.record.update(state_count=1), 1),
    (lambda api: api.record.update(policy_queries=0), 1),
    (lambda api: api.record.update(artifacts={}), 1),
    (lambda api: api.record.update(policy_content_sha256="changed"), 1),
    (lambda api: api.dataset.update(members=[{"run_id": "unrelated"}]), 2),
    (lambda api: api.dataset.update(integrity_status="BROKEN"), 2),
    (lambda api: api.jobs["training"]["training_summary"].update(policy_overlay=""), 3),
    (lambda api: api.jobs["training"]["training_summary"].update(steps=1), 3),
    (lambda api: api.jobs["training"]["training_summary"].update(resumed_from_step=1), 3),
    (lambda api: api.jobs["training"]["metrics"].update(actor_loss=float("nan")), 3),
    (lambda api: api.jobs["training"]["metrics"].update(actor_grad_norm=float("inf")), 3),
    (lambda api: api.jobs["training"]["metrics"].update(actor_learning_rate=0), 3),
    (lambda api: api.jobs["training"]["parameters"].update(model={**api.base, "base_revision": "c" * 64}), 3),
    (lambda api: api.child.update(policy_id="another-child"), 3),
    (lambda api: api.child.update(parent_model_id="another-base"), 3),
    (lambda api: api.child.update(family="different"), 3),
    (lambda api: api.child.update(kind="base"), 3),
    (lambda api: api.child.update(algorithm="iql"), 3),
    (lambda api: api.child.update(training_step=1000), 3),
    (lambda api: api.child.update(base_revision="c" * 64), 3),
    (lambda api: api.child.update(components=[]), 3),
    (lambda api: api.child.update(manifest={"dataset_sha256": "other-dataset"}), 3),
    (lambda api: api.evaluation.update(policy_snapshot={**api.child, "content_sha256": "changed"}), 4),
    (lambda api: api.evaluation["aggregate"].update(errors=1), 4),
    (lambda api: api.evaluation.update(trials=[{"steps": 1}]), 4),
])
def test_invalid_artifacts_or_identity_stop_pipeline(monkeypatch, mutation, submissions):
    api = Platform()
    mutation(api)
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == 1
    assert sum(method == "POST" for method, *_ in api.calls) == submissions
    assert {path.rsplit("/", 1)[-1] for path in api.deleted if path.startswith("/api/jobs/")} == api.created_jobs


def test_interrupt_waits_for_stop_and_cleans_all_submitted_resources(monkeypatch):
    api = Platform()
    api.jobs["training"]["status"] = "RUNNING"
    monkeypatch.setattr(cli, "Client", lambda _: api)
    monkeypatch.setattr(cli.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(OPTIONS) == 130
    assert api.jobs["training"]["status"] == "CANCELED"
    assert "/api/jobs/training" in api.deleted
    assert "/api/training-datasets/owned-dataset" in api.deleted
    assert "/api/sessions/owned-run" in api.deleted
    assert "evaluation" not in api.created_jobs


def test_cleanup_failure_reports_owned_id_and_does_not_report_pass(monkeypatch, capsys):
    api = Platform()
    request = api.request
    def fail(method, path, payload=None):
        if method == "DELETE" and path == "/api/jobs/training":
            raise cli.PlatformError(500, "disk unavailable")
        return request(method, path, payload)
    api.request = fail
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == 1
    captured = capsys.readouterr()
    assert "CLEANUP ERROR: training" in captured.err and "disk unavailable" in captured.err
    assert "PASS:" not in captured.out
    assert "/api/sessions/owned-run" in api.deleted


def test_cleanup_retries_release_conflict_and_accepts_already_deleted(monkeypatch):
    calls = []
    def request(method, path, payload):
        calls.append(path)
        raise cli.PlatformError(409 if len(calls) == 1 else 404, "not ready")
    resources = cli.TestResources(type("Client", (), {"request": staticmethod(request)})())
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    resources._delete("/api/jobs/owned", {"confirm_job_id": "owned"})
    assert calls == ["/api/jobs/owned", "/api/jobs/owned"]


def test_interrupt_only_stops_submitted_job(monkeypatch):
    calls = []
    class Running:
        def request(self, method, path, payload=None):
            calls.append((method, path))
            return {"text": "", "next_offset": 0}
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.time, "sleep", interrupt)
    with pytest.raises(cli.JobFailed) as error:
        cli.wait_for_job(Running(), {"id": "owned", "status": "RUNNING"})
    assert error.value.exit_code == 130
    assert calls[-1] == ("POST", "/api/jobs/owned/stop")


def test_interrupt_during_first_status_read_stops_known_simulation():
    calls = []
    class Running:
        def request(self, method, path, payload=None):
            calls.append((method, path))
            if method == "GET":
                raise KeyboardInterrupt
    with pytest.raises(cli.JobFailed) as error:
        cli.wait_for_job(Running(), {"id": "owned"})
    assert error.value.exit_code == 130
    assert calls == [("GET", "/api/jobs/owned"), ("POST", "/api/jobs/owned/stop")]


def test_resolved_normalization_is_owned_by_model_module():
    api = Platform()
    api.child["stats_key"] = "resolved-by-adapter"
    child = cli.exported_model(api, api.jobs["training"], api.base, api.base)
    assert child["policy_id"] == "exact-child"


def test_failed_job_reports_reason_stage_and_location(monkeypatch, capsys):
    api = Platform()
    api.jobs["training"].update(status="FAILED", stage="train", error="Missing normalization file",
                                 config_path="/jobs/training/effective.yaml")
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == 1
    error = capsys.readouterr().err
    assert "training / train: Missing normalization file" in error
    assert "/jobs/training/effective.yaml" in error
    assert "/api/jobs/training/logs" in error


def test_local_validation_reports_source_location(monkeypatch, capsys):
    api = Platform()
    api.base["kind"] = "child"
    monkeypatch.setattr(cli, "Client", lambda _: api)
    assert cli.main(OPTIONS) == 1
    error = capsys.readouterr().err
    assert "registered base model" in error
    assert "test_model.py" in error and "line " in error


@pytest.mark.parametrize("error", [
    URLError("unavailable"), HTTPError("http://localhost", 409, "conflict", {}, io.BytesIO(b"busy")),
])
def test_http_errors_are_not_retried(monkeypatch, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise error
    monkeypatch.setattr(cli, "urlopen", fail)
    with pytest.raises(RuntimeError):
        cli.Client("http://localhost").request("POST", "/api/training-runs", {})
    assert len(calls) == 1
