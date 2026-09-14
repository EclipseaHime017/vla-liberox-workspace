import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

from backend.app.api.datasets import get_stage_annotation, save_stage_annotation
from backend.app.api.models import StageAnnotationRequest
from backend.app.core.exceptions import ConflictError
from backend.app.services.stage_annotation_service import StageAnnotationService


OFFLINE_ROOT = Path(__file__).resolve().parents[3] / "vla-adapter-rynn-iql"


def fixture(tmp_path, done=None):
    trajectory = tmp_path / "episodes/episode_000/trajectory.npz"
    trajectory.parent.mkdir(parents=True)
    flags = np.zeros(20, dtype=bool) if done is None else done
    np.savez_compressed(trajectory, env_action=np.zeros((len(flags), 7)), done=flags,
                        time_seconds=np.arange(len(flags)+1)/20)
    run = dict(id="r", status="COMPLETED", trajectory=str(trajectory))
    service = StageAnnotationService(SimpleNamespace(get_run=lambda _: run), OFFLINE_ROOT)
    service._defaults = lambda: (5, 2.)
    return service, trajectory, run


def test_saved_keyframes_cover_full_record_and_auto_confirm_success(tmp_path):
    flags = np.zeros(20, dtype=bool)
    flags[7:12] = True
    service, path, _ = fixture(tmp_path, flags)
    original = path.read_bytes()
    before = service.detail("r")
    assert before["status"] == "missing" and before["success_step"] == 12
    assert before["action_count"] == 20 and len(before["time_seconds"]) == 21
    frames = [{"step": 3, "kind": "positive"}, {"step": 6, "kind": "negative"},
              {"step": 9, "kind": "positive"}]
    saved = service.save("r", frames, 2, before["revision"])
    assert saved["status"] == "ready" and saved["revision"] != before["revision"]
    np.testing.assert_allclose(np.asarray(saved["scores"])[[0,3,6,9,12,20]], [-1,-.5,-1,-.5,0,0])
    assert path.read_bytes() == original
    payload = json.loads((path.parent / "stage_annotation.json").read_text())
    assert payload["trajectory_sha256"] == hashlib.sha256(original).hexdigest()
    assert len(payload["keyframes"]) == 3  # success is generated, never double-counted


def test_empty_failed_annotation_is_explicit_and_negative_values_not_clipped(tmp_path):
    service, _, _ = fixture(tmp_path)
    first = service.detail("r")
    empty = service.save("r", [], 2, first["revision"])
    assert empty["status"] == "ready" and empty["scores"] == [-1.] * 21
    negative = service.save("r", [{"step": 10, "kind": "negative"}], 3, empty["revision"])
    assert negative["scores"][10:] == [-2.] * 11


def test_version_conflict_and_source_change_require_explicit_reload(tmp_path):
    service, path, _ = fixture(tmp_path)
    original = service.detail("r")
    current = service.save("r", [], 2, original["revision"])
    with pytest.raises(ConflictError):
        service.save("r", [], 3, original["revision"])
    np.savez_compressed(path, env_action=np.ones((20,7)), done=np.zeros(20,dtype=bool),
                        time_seconds=np.arange(21)/20)
    assert service.detail("r")["status"] == "stale"
    with pytest.raises(ConflictError):
        service.save("r", [], 2, current["revision"])


def test_repeat_reads_cache_control_npz_and_never_load_observations(tmp_path, monkeypatch):
    service, path, _ = fixture(tmp_path)
    (path.parent / "trajectory_observations.npz").write_bytes(b"must never be read")
    original = np.load
    calls = []
    def checked(filename, **kwargs):
        assert Path(filename) == path
        calls.append(filename)
        return original(filename, **kwargs)
    monkeypatch.setattr(np, "load", checked)
    initial = service.detail("r")
    service.save("r", [], 2, initial["revision"])
    for _ in range(10):
        assert service.detail("r")["status"] == "ready"
    assert len(calls) == 1


def test_preflight_fails_all_missing_stale_or_wrong_frozen_members(tmp_path):
    service, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="r: 缺少"):
        service.validate_members([{"run_id": "r"}], 5)
    service.save("r", [], 2, service.detail("r")["revision"])
    assert list(service.validate_members([{"run_id": "r"}], 5)) == ["r"]
    # Dataset-local success thresholds are recipe context, not label identity.
    assert service.validate_members([{"run_id": "r"}], 6) == service.validate_members([{"run_id": "r"}], 5)
    with pytest.raises(ValueError, match="冻结数据集不匹配"):
        service.validate_members([{"run_id": "r", "artifacts": {"trajectory": {"sha256": "0"*64}}}], 5)
    (path.parent / "stage_annotation.json").write_text("broken")
    assert service.detail("r")["status"] == "stale"


def test_preview_exponent_changes_without_relabeling(tmp_path):
    service, path, _ = fixture(tmp_path)
    frames = [{"step": 10, "kind": "positive"}]
    before = service.detail("r")
    p2 = service.save("r", frames, 2, before["revision"])
    labels_path = path.parent / "stage_annotation.json"
    labels = labels_path.read_bytes()
    assert "exponent" not in json.loads(labels)
    service._defaults = lambda: (5, 4.)
    p4 = service.detail("r")
    assert p4["revision"] == p2["revision"]
    assert p4["scores"][5] != p2["scores"][5]
    assert labels_path.read_bytes() == labels


def test_recipe_failure_does_not_discard_saved_keyframes(tmp_path):
    service, path, _ = fixture(tmp_path, np.ones(20, dtype=bool))
    # An invalid successful normalization is an evaluation failure, not corrupt labels.
    saved = service.save("r", [{"step": 2, "kind": "negative"}], 2,
                         service.detail("r")["revision"])
    assert saved["status"] == "ready"
    assert "denominator" in saved["derivation_error"]
    assert saved["scores"] == []
    assert json.loads((path.parent / "stage_annotation.json").read_text())["keyframes"] == saved["keyframes"]


def test_active_or_symlink_sidecars_cannot_be_written(tmp_path):
    service, path, run = fixture(tmp_path)
    current = service.detail("r")
    run["status"] = "RUNNING"
    with pytest.raises(RuntimeError, match="active"):
        service.save("r", [], 2, current["revision"])
    run["status"] = "COMPLETED"
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("untouched")
    (path.parent / "stage_annotation.json").symlink_to(unrelated)
    with pytest.raises(ValueError, match="symlink"):
        service.save("r", [], 2, current["revision"])
    assert unrelated.read_text() == "untouched"


def test_api_schema_and_roundtrip(tmp_path, monkeypatch):
    service, _, _ = fixture(tmp_path)
    app = FastAPI()
    app.state.stage_annotation_service = service
    request = Request({"type": "http", "app": app})
    # Exercise route wiring separately from AnyIO's platform thread scheduling.
    dispatches = []
    async def dispatch(function, *args, **kwargs):
        dispatches.append(function.__name__)
        return function(*args, **kwargs)
    monkeypatch.setattr("backend.app.api.datasets.run_in_threadpool", dispatch)
    async def exercise():
        current = await get_stage_annotation("r", request)
        saved = await save_stage_annotation("r", StageAnnotationRequest(
            keyframes=[], exponent=2., revision=current["revision"]), request)
        assert saved["status"] == "ready"
    asyncio.run(exercise())
    assert dispatches == ["detail", "save"]
    for values in ({"exponent": True}, {"exponent": float("nan")},
                   {"keyframes": [{"step": 1.5, "kind": "positive"}]}, {"unexpected": 1}):
        with pytest.raises(ValidationError):
            StageAnnotationRequest(**{"keyframes": [], **values})
