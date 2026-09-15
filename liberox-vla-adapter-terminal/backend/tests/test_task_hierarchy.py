"""Purpose grouping is metadata; filtering preserves exact scene identities."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml
from fastapi import FastAPI

from backend.app.api import datasets as datasets_api
from backend.app.core.config import load_ui_config
from backend.app.services.task_catalog import ConfiguredTaskCatalog
from backend.app.services.offline_job_service import OfflineJobService
import backend.app.services.task_catalog as tasks_module
from test_training_platform import make_run, service
from test_ui_config_storage import VALID


@pytest.mark.parametrize("families", [
    {}, [{"family_id": "bowl", "label": "Bowl", "task_names": []}],
    [{"family_id": "bowl", "label": "Bowl", "task_names": ["pick", "pick"]}],
    [{"family_id": "bowl", "label": "Bowl", "task_names": ["pick"], "extra": True}],
    [{"family_id": "bowl", "label": "Bowl", "task_names": ["pick"]},
     {"family_id": "other", "label": "Other", "task_names": ["pick"]}],
    [{"family_id": "bowl", "label": "Bowl", "task_names": ["pick"]},
     {"family_id": "bowl", "label": "Other", "task_names": ["place"]}],
])
def test_rejects_ambiguous_family_configuration(tmp_path, families):
    path = tmp_path / "ui.yaml"
    path.write_text(VALID + yaml.safe_dump({"task_families": families}))
    with pytest.raises((TypeError, ValueError)):
        load_ui_config(path)


def test_default_catalog_has_three_purposes_and_thirteen_unique_level1_to_4_scenes(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    config = load_ui_config(root / "configs/ui_config.yaml")
    default = SimpleNamespace(level="LEVEL1", task_name=config.task_families[0].task_names[0])
    # Read the actual BDDL prompts without importing MuJoCo, VLA or GPU libraries.
    def parse(path):
        language = Path(path).read_text().split("(:language", 1)[1].split(")", 1)[0].strip()
        return {"language": language}
    monkeypatch.setattr(tasks_module.direct, "load_initial_states", lambda *_: np.zeros((10, 5)))
    catalog = ConfiguredTaskCatalog(SimpleNamespace(parse_bddl_file=parse), root / "LIBERO-X",
                                   default, config.additional_tasks, config.task_families)
    entries = catalog.list_tasks()
    assert len(entries) == 13
    assert all(entry["available"] for entry in entries)
    assert {entry["level"] for entry in entries} == {f"LEVEL{i}" for i in range(1, 5)}
    assert {entry["family_id"] for entry in entries} == {"bowl_on_stove", "open_top_drawer", "stack_bowls"}
    assert len({(e["family_id"], e["level"], e["prompt"]) for e in entries}) == 13
    bowl = [e for e in entries if e["family_id"] == "bowl_on_stove"]
    assert len(bowl) == 5 and len({e["family_label"] for e in bowl}) == 1
    assert len([e for e in bowl if e["level"] == "LEVEL4"]) == 2
    assert all(e["task_id"] == f"{e['level']}::{e['task_name']}" for e in entries)


def test_task_set_filters_before_pagination_counts_and_status_checks(tmp_path, monkeypatch):
    runs = [make_run(tmp_path, str(i), task_id=f"LEVEL{i % 3 + 1}::pick") for i in range(9)]
    current = service(tmp_path, runs)
    checked = []
    def status(run):
        checked.append(run["task_id"])
        return {"status": "READY"}
    current.evaluations = SimpleNamespace(status=status)
    ids = ["LEVEL1::pick", "LEVEL2::pick"]
    page = current.list_runs_page(task_ids=ids, page=2, page_size=5)
    assert page["total"] == page["evaluated_count"] == page["eligible_count"] == 6
    assert page["pages"] == 2 and len(page["items"]) == 1
    assert set(checked) == set(ids)
    assert len(current.list_runs("LEVEL1::pick", task_ids=ids)) == 3
    assert current.list_runs("LEVEL3::pick", task_ids=ids) == []
    assert current.list_runs_page(task_ids=[])["total"] == 0

    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr(datasets_api, "run_in_threadpool", inline)
    app = FastAPI()
    app.state.training_dataset_service = current
    app.include_router(datasets_api.router)
    async def request(params):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/datasets/runs", params=params)
            assert response.status_code == 200
            return response.json()
    assert asyncio.run(request([("task_ids", id) for id in ids]))["total"] == 6
    assert asyncio.run(request({"task_ids": ""}))["total"] == 0
    assert asyncio.run(request({"task_id": ids[0]}))["total"] == 3
    assert asyncio.run(request({}))["total"] == 9


def test_frozen_dataset_and_evaluation_history_task_set_filters(tmp_path):
    ids = ["LEVEL1::pick", "LEVEL2::pick", "LEVEL4::drawer"]
    runs = [make_run(tmp_path, str(i), task_id=task) for i, task in enumerate(ids)]
    current = service(tmp_path, runs)
    for run in runs:
        current.create(name=run["id"], task_id=run["task_id"],
                       selection={"mode": "manual", "run_ids": [run["id"]]})
    assert {d["task_id"] for d in current.list(task_ids=ids[:2])} == set(ids[:2])
    assert current.list(task_ids=[]) == []
    assert len(current.list(ids[0])) == 1

    jobs = object.__new__(OfflineJobService)
    jobs.evaluations_root = tmp_path / "evaluations"
    jobs._load_evaluation_manifest = lambda path: json.loads(path.read_text())
    jobs._synchronize_evaluation = lambda path, payload: payload
    jobs._public_evaluation = lambda payload, path, detail: payload
    for i, task in enumerate(ids):
        path = jobs.evaluations_root / str(i) / "date" / "run" / "evaluation.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"task_snapshot": {"task_id": task}}))
    assert len(jobs.list_evaluations(task_ids=ids[:2])) == 2
    assert jobs.list_evaluations(task_ids=[]) == []
    assert len(jobs.list_evaluations(task_id=ids[0])) == 1
