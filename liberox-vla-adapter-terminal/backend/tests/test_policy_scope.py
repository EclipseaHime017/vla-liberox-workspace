import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.core.exceptions import ConflictError
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.services.policy_management_service import PolicyManagementService
from backend.app.storage.files import recorded_policy_family
from backend.app.policies.catalog import PolicyCatalog
from test_policy_overlay import _overlay, BASE_MODELS
from test_training_platform import make_run, service


def model_runs(tmp_path):
    values = [make_run(tmp_path, name, success=index % 2 == 0)
              for index, name in enumerate(("a", "b", "c", "d", "unknown"))]
    for run, family, identifier in zip(values, ("custom", "custom", "custom", "another", None),
                                      ("alpha", "alpha", "beta", "gamma", "old")):
        run.update(policy_family=family, policy_id=identifier, policy_label=identifier.upper())
    return values


def test_scope_is_applied_before_paging_and_statistics_with_historical_choices(tmp_path):
    current = service(tmp_path, model_runs(tmp_path))
    page = current.list_runs_page(policy_family="custom", policy_id="alpha", page=2, page_size=1)
    assert page["total"] == page["eligible_count"] == 2
    assert page["pages"] == 2 and page["success_count"] == 1
    assert len(page["items"]) == 1 and page["items"][0]["policy_id"] == "alpha"
    assert {item["policy_id"] for item in page["policy_options"]} == {"alpha", "beta", "gamma", "old"}
    assert current.list_runs_page(policy_family="another", policy_id="alpha")["total"] == 0
    unknown = current.list_runs_page(policy_family="__unknown__")
    assert [run["id"] for run in unknown["items"]] == ["unknown"]


@pytest.mark.parametrize("selection", [
    dict(mode="random", size=2, seed=7),
    dict(mode="sequential", size=2),
    dict(mode="manual", run_ids=["a", "b"]),
    dict(mode="rule", quotas=[dict(source_type="inference", outcome="success", count=1),
                             dict(source_type="inference", outcome="failure", count=1)]),
])
def test_every_packaging_mode_respects_model_scope(tmp_path, selection):
    current = service(tmp_path, model_runs(tmp_path))
    selection.update(policy_family="custom", policy_id="alpha")
    assert set(current.preview("LEVEL1::pick", selection)["run_ids"]) == {"a", "b"}
    dataset = current.create(name="Scoped", task_id="LEVEL1::pick", selection=selection)
    assert {member["policy_id"] for member in dataset["members"]} == {"alpha"}
    before = (current.root / dataset["id"] / "dataset.json").read_bytes()
    assert current.list(policy_family="another") == []
    assert [item["id"] for item in current.list(policy_family="custom", policy_id="alpha")] == [dataset["id"]]
    assert (current.root / dataset["id"] / "dataset.json").read_bytes() == before
    with pytest.raises(ValueError, match="unavailable"):
        current.preview("LEVEL1::pick", dict(mode="manual", run_ids=["d"], policy_id="alpha"))


def test_bulk_evaluation_cannot_include_a_hidden_model(tmp_path):
    current = service(tmp_path, model_runs(tmp_path))
    jobs = object.__new__(OfflineJobService)
    jobs.lock, jobs.datasets = threading.RLock(), current
    jobs.trajectory_evaluations = SimpleNamespace(exists=lambda _: False)
    jobs.robometer_evaluations = None
    calls = []
    jobs._launch_trajectory_evaluation = lambda **kwargs: calls.append(kwargs) or dict(id="test", status="QUEUED")
    jobs.start_trajectory_evaluation(task_id="LEVEL1::pick", run_ids=None, overwrite=False,
                                     policy_family="custom", policy_id="alpha")
    assert set(calls[0]["requested_run_ids"]) == {"a", "b"}
    with pytest.raises(ValueError, match="Unavailable"):
        jobs.start_trajectory_evaluation(task_id="LEVEL1::pick", run_ids=["d"], overwrite=False,
                                         policy_family="custom", policy_id="alpha")


def test_legacy_family_comes_from_recorded_configuration_not_a_model_name(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(policy=dict(provider="future-family", policy_id="same", content_sha256="saved"))))
    assert recorded_policy_family(tmp_path, {"policy_id": "same"}) == "future-family"
    assert recorded_policy_family(tmp_path, {"policy_id": "different"}) is None
    assert recorded_policy_family(tmp_path, {"policy_id": "same", "policy_content_sha256": "changed"}) is None
    assert recorded_policy_family(tmp_path, {"policy_family": "frozen"}) == "frozen"
    path.unlink()
    assert recorded_policy_family(tmp_path, {"policy_id": "pi05-whatever"}) is None


def test_training_records_match_actual_outputs_not_dataset_or_input_model(tmp_path):
    _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, base_models=BASE_MODELS)
    records = [
        dict(id="correct", kind="training", parameters={}, training_summary={"policy_overlay": str(tmp_path / "trained/policy.yaml")}),
        dict(id="wrong-output", kind="training", parameters={"dataset_sha256": "a" * 64},
             training_summary={"policy_overlay": str(tmp_path / "another/policy.yaml")}),
        dict(id="input-only", kind="training", parameters={"policy_id": "trained"}),
        dict(id="wrong-directory", kind="training", parameters={}, training_summary={
            "policy_overlay": str(tmp_path / "other-registry/trained/policy.yaml")}),
    ]
    jobs = SimpleNamespace(list=lambda: records, configuration=lambda identifier: {"filename": "effective_config.yaml", "text": identifier})
    management = PolicyManagementService(SimpleNamespace(policy_catalog=catalog), jobs)
    assert [job["id"] for job in management.detail("trained")["training_records"]] == ["correct"]
    assert management.configuration("trained", "correct")["text"] == "correct"
    with pytest.raises(KeyError):
        management.configuration("trained", "wrong-output")
    copied = management.copy("trained", "Copy")
    assert [job["id"] for job in copied["training_records"]] == ["correct"]
    assert (tmp_path / copied["policy_id"] / "provenance.json").is_file()


def test_model_training_source_must_be_unique_without_picking_a_record(tmp_path):
    _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, base_models=BASE_MODELS)
    records = [dict(id=identifier, kind="training", parameters={}, training_summary={
        "policy_overlay": str(tmp_path / "trained/policy.yaml")}) for identifier in ("first", "second")]
    jobs = SimpleNamespace(list=lambda: records, configuration=lambda _: pytest.fail("Ambiguous sources must not expose a YAML"))
    management = PolicyManagementService(SimpleNamespace(policy_catalog=catalog), jobs)
    for operation in (lambda: management.detail("trained"), lambda: management.configuration("trained", "first")):
        with pytest.raises(ConflictError) as error:
            operation()
        assert error.value.code == "AMBIGUOUS_MODEL_TRAINING_RECORD"
        assert error.value.context == {"policy_id": "trained", "job_ids": ["first", "second"]}
    records.reverse()
    with pytest.raises(ConflictError):
        management.configuration("trained", "second")


def test_yaml_reader_returns_historical_text_and_rejects_unsafe_or_missing_files(tmp_path):
    jobs = object.__new__(OfflineJobService)
    jobs.jobs_root = tmp_path / "jobs"
    directory = jobs.jobs_root / "train"
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps(dict(schema_version=1, id="train", kind="training", parameters={})))
    config = directory / "effective_config.yaml"
    text = "training:\n  seed: 17\n# exact historical configuration\n"
    config.write_text(text)
    assert jobs.configuration("train") == {"filename": "effective_config.yaml", "text": text}
    with pytest.raises(ValueError, match="Invalid job id"):
        jobs.configuration("../train")
    config.unlink()
    with pytest.raises(FileNotFoundError):
        jobs.configuration("train")
    outside = tmp_path / "other.yaml"
    outside.write_text("private: true")
    config.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        jobs.configuration("train")


@pytest.mark.parametrize("training_count", [0, 1, 2])
def test_base_model_has_no_training_records_even_when_used_by_training_jobs(tmp_path, training_count):
    catalog = PolicyCatalog(tmp_path, base_models=BASE_MODELS)
    base = catalog.entry("base")
    settings = dict(base_id=base.policy_id, family=base.family,
                    base_checkpoint=base.base_checkpoint, base_revision=base.base_revision)
    records = [dict(id=f"train-{index}", kind="training", parameters={"model": settings},
                    training_summary={"policy_overlay": str(tmp_path / f"trained-{index}" / "policy.yaml")})
               for index in range(training_count)]
    jobs = SimpleNamespace(list=lambda: records, configuration=lambda _: pytest.fail("Base models must not expose a training YAML"))
    management = PolicyManagementService(SimpleNamespace(policy_catalog=catalog), jobs)
    assert management.detail("base")["training_records"] == []
    with pytest.raises(KeyError, match="不属于所选模型"):
        management.configuration("base", "train-0")


def test_unrecorded_policy_id_is_not_assumed_to_be_base(tmp_path, monkeypatch):
    run = make_run(tmp_path, "unknown")
    current = service(tmp_path, [run])
    assert current.list_runs_page(policy_id="base")["total"] == 0
    page = current.list_runs_page(policy_family="__unknown__", policy_id="__unknown__")
    assert page["total"] == 1 and page["policy_options"][0]["label"] == "未记录模型"


def test_scoped_page_does_not_inspect_unrelated_trajectory_arrays(tmp_path, monkeypatch):
    current = service(tmp_path, model_runs(tmp_path))
    classify = current.classify
    checked = []
    def inspect(run):
        checked.append(run["id"])
        return classify(run)
    monkeypatch.setattr(current, "classify", inspect)
    assert current.list_runs_page(policy_id="alpha")["total"] == 2
    assert set(checked) == {"a", "b"}


def test_model_scope_and_yaml_are_wired_through_http_without_path_access(tmp_path):
    import asyncio
    import httpx
    from fastapi import FastAPI
    from backend.app.api import datasets, policies, training_datasets

    app = FastAPI()
    app.state.training_dataset_service = service(tmp_path, model_runs(tmp_path))
    requested = []
    app.state.policy_management_service = SimpleNamespace(configuration=lambda *args:
        requested.append(args) or {"filename": "effective_config.yaml", "text": "training: {seed: 7}"})
    for router in (datasets.router, policies.router, training_datasets.router):
        app.include_router(router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            page = await client.get("/api/datasets/runs?policy_family=custom&policy_id=alpha&page_size=1")
            assert page.status_code == 200 and page.json()["total"] == 2
            selection = await client.post("/api/training-datasets/preview", json={
                "task_id": "LEVEL1::pick", "selection": {"mode": "random", "size": 2, "policy_family": "custom", "policy_id": "alpha"}})
            assert selection.status_code == 200 and set(selection.json()["run_ids"]) == {"a", "b"}
            config = await client.get("/api/models/model-id/training/job-id/config")
            assert config.json()["text"] == "training: {seed: 7}"
            assert requested == [("model-id", "job-id")]
    asyncio.run(check())


def test_model_scoped_frozen_dataset_remains_compatible_with_training_prepare(tmp_path):
    from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
    from vla_rynn_iql.data import prepare_dataset, load_manifest

    current = service(tmp_path, model_runs(tmp_path))
    dataset = current.create(name="Trainable scope", task_id="LEVEL1::pick",
        selection=dict(mode="manual", run_ids=["a", "b"], policy_family="custom", policy_id="alpha"))
    raw = load_train_config(DEFAULT_TRAIN_CONFIG).raw
    raw["data"].update(project_id="test", task_ids=["LEVEL1::pick"],
                       selection_manifest=str(current.root / dataset["id"] / "dataset.json"))
    raw["paths"].update(work_dir=str(tmp_path / "prepared"), output_dir=str(tmp_path / "output"))
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_train_config(path)
    prepare_dataset(config)
    assert {episode["run_id"] for episode in load_manifest(config)["episodes"]} == {"a", "b"}
