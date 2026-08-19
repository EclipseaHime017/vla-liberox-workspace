from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from backend.app import main as main_module
from backend.app.main import (
    CreateBranchRequest,
    DraftRequest,
    DeleteSessionRequest,
    UpdateDraftRequest,
    create_app,
)


class FakeManager:
    def __init__(self, tmp_path: Path):
        self.lock = threading.RLock()
        self.active = None
        self.draft = None
        self.file = tmp_path / "summary.json"
        self.file.write_text("{}\n", encoding="utf-8")
        self.records = {
            "root": {
                "id": "root", "kind": "original", "status": "COMPLETED",
                "control_mode": "policy", "branchable": True, "legacy": False,
                "action_count": 10, "state_count": 11,
                "artifacts": {"summary.json": str(self.file)},
            }
        }
        self.sessions = {}

    def bootstrap(self):
        return {"ok": True}

    def list_sessions(self):
        return list(self.records.values())

    def controller_status(self):
        return {"state": "READY", "connected": True, "calibrated": True}

    def calibrate_controller(self):
        return {"state": "CALIBRATING", "connected": True, "calibrated": False}

    def get_public(self, session_id):
        if session_id not in self.records:
            raise KeyError(session_id)
        return self.records[session_id]

    def get_draft(self):
        return self.draft

    def create_draft(self, task_id, max_steps, open_loop_steps, policy_id="base"):
        if self.active:
            raise RuntimeError("active")
        self.draft = {
            "id": "draft", "task_id": task_id, "max_steps": max_steps,
            "open_loop_steps": open_loop_steps, "preview_status": "READY",
            "policy_id": policy_id, "policy_label": policy_id,
            "preview_ready": True, "preview_available": True,
        }
        return self.draft

    def update_draft(self, **patch):
        if self.draft is None:
            raise KeyError("draft")
        self.draft.update({key: value for key, value in patch.items() if value is not None})
        return self.draft

    def discard_draft(self):
        existed = self.draft is not None
        self.draft = None
        return existed

    def draft_preview(self):
        if self.draft is None:
            raise KeyError("draft")
        return b"\xff\xd8\xff\xd9"

    def start_draft(self):
        if self.draft is None:
            raise KeyError("draft")
        self.active = "new"
        value = {
            "id": "new", "kind": "original", "status": "RUNNING",
            "control_mode": "policy", "branchable": False, "legacy": False,
            "max_steps": self.draft["max_steps"],
            "open_loop_steps": self.draft["open_loop_steps"],
            "action_count": 0, "state_count": 1, "artifacts": {},
        }
        self.draft = None
        self.records["new"] = value
        return value

    def stop(self, session_id):
        value = self.get_public(session_id)
        value["status"] = "STOPPING"
        return value

    def create_branch(
        self,
        session_id,
        resume_step,
        control_mode,
        open_loop_steps,
        *,
        translation_gain=None,
        rotation_gain=None,
    ):
        self.get_public(session_id)
        value = {
            "id": "branch", "kind": "branch", "status": "RUNNING",
            "control_mode": control_mode, "branchable": False, "legacy": False,
            "manual_source": "spacemouse" if control_mode == "manual" else None,
            "manual_translation_gain": translation_gain,
            "manual_rotation_gain": rotation_gain,
            "resume_step": resume_step, "open_loop_steps": open_loop_steps,
            "action_count": resume_step, "state_count": resume_step + 1, "artifacts": {},
        }
        self.records["branch"] = value
        return value

    def frame_state(self, session_id, step):
        self.get_public(session_id)
        return {"step": step}

    def render_frame(self, session_id, step):
        self.get_public(session_id)
        return b"\xff\xd8\xff\xd9"

    def artifact_path(self, session_id, name):
        self.get_public(session_id)
        if name != self.file.name:
            raise FileNotFoundError(name)
        return self.file

    def delete_session(self, session_id, confirm_session_id):
        if session_id != confirm_session_id:
            raise ValueError("confirmation mismatch")
        self.records.pop(session_id)


def test_api_create_conflict_stop_branch_and_artifact(tmp_path: Path):
    manager = FakeManager(tmp_path)
    app = create_app(ui_config=object(), eval_config=object(), manager=manager)
    app.state.manager = manager
    request = Request({"type": "http", "app": app})

    async def exercise():
        routes = []
        for route in app.routes:
            included = getattr(route, "original_router", None)
            routes.extend(included.routes if included is not None else [route])

        def endpoint(path, method="GET"):
            return next(
                route.endpoint for route in routes
                if getattr(route, "path", None) == path
                and method in getattr(route, "methods", set())
            )
        get_session = next(
            route.endpoint for route in routes
            if getattr(route, "path", None) == "/api/sessions/{run_id}"
            and "GET" in getattr(route, "methods", set())
        )
        delete_session = next(
            route.endpoint for route in routes
            if getattr(route, "path", None) == "/api/sessions/{run_id}"
            and "DELETE" in getattr(route, "methods", set())
        )
        assert await endpoint("/api/bootstrap")(request) == {"ok": True}
        assert (await endpoint("/api/controller")(request))["state"] == "READY"
        assert (await endpoint("/api/controller/calibrate", "POST")(request))["state"] == "CALIBRATING"
        body = DraftRequest(
            task_id="LEVEL1::task", policy_id="trained", max_steps=20,
            open_loop_steps=4,
        )
        draft = await endpoint("/api/draft", "POST")(body, request)
        assert draft["preview_ready"] is True
        assert draft["policy_id"] == "trained"
        assert "new" not in manager.records
        updated = await endpoint("/api/draft", "PATCH")(
            UpdateDraftRequest(open_loop_steps=2), request
        )
        assert updated["open_loop_steps"] == 2
        preview = await endpoint("/api/draft/preview.jpg")(request)
        assert preview.body.startswith(b"\xff\xd8")
        created = await endpoint("/api/draft/start", "POST")(request)
        assert created["status"] == "RUNNING"
        with pytest.raises(HTTPException) as conflict:
            await endpoint("/api/draft", "POST")(body, request)
        assert conflict.value.status_code == 409
        stopped = await endpoint("/api/sessions/{run_id}/stop", "POST")("new", request)
        assert stopped["status"] == "STOPPING"
        branch_body = CreateBranchRequest(
            resume_step=5,
            control_mode="manual",
            open_loop_steps=1,
            translation_gain=0.25,
            rotation_gain=0.08,
        )
        branch = await endpoint("/api/sessions/{run_id}/branches", "POST")(
            "root",
            branch_body,
            request,
        )
        assert branch["resume_step"] == 5
        assert branch["manual_source"] == "spacemouse"
        assert branch["manual_rotation_gain"] == 0.08
        state = await endpoint("/api/sessions/{run_id}/frames/{step}/state")(
            "root",
            3,
            request,
        )
        assert state == {"step": 3}
        assert endpoint("/api/sessions/{run_id}/frames/{step}")
        assert manager.render_frame("root", 3).startswith(b"\xff\xd8")
        assert manager.artifact_path("root", "summary.json") == manager.file
        artifact_response = await endpoint(
            "/api/sessions/{run_id}/artifacts/{name:path}"
        )("root", "summary.json", request)
        assert artifact_response.headers["content-disposition"].startswith("inline;")
        assert artifact_response.headers["accept-ranges"] == "bytes"
        deleted = await delete_session(
            "root",
            DeleteSessionRequest(confirm_session_id="root"),
            request,
        )
        assert deleted == {"deleted": "root"}
        with pytest.raises(HTTPException) as missing:
            await get_session("missing", request)
        assert missing.value.status_code == 404

    asyncio.run(exercise())


def test_request_validation(tmp_path: Path):
    del tmp_path
    with pytest.raises(ValidationError):
        DraftRequest(task_id="task", max_steps=0, open_loop_steps=9)
    with pytest.raises(ValidationError):
        UpdateDraftRequest()
    with pytest.raises(ValidationError):
        CreateBranchRequest(
            resume_step=0,
            control_mode="invalid",
            open_loop_steps=1,
        )
    with pytest.raises(ValidationError):
        CreateBranchRequest(
            resume_step=0,
            control_mode="manual",
            open_loop_steps=1,
            translation_gain=0.01,
            rotation_gain=0.25,
        )
    with pytest.raises(ValidationError):
        CreateBranchRequest.model_validate({
            "resume_step": 0,
            "control_mode": "manual",
            "open_loop_steps": 1,
            "manual_source": "spacemouse",
        })


def test_frontend_entry_is_never_served_from_browser_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text("<html></html>\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "FRONTEND_DIST", frontend_dist)
    app = create_app(ui_config=object(), eval_config=object(), manager=FakeManager(tmp_path))
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/{path:path}"
    )

    response = asyncio.run(route.endpoint(""))

    assert response.headers["cache-control"] == "no-store"


def test_build_info_endpoint_reports_the_served_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frontend_root = tmp_path / "frontend"
    (frontend_root / "src").mkdir(parents=True)
    (frontend_root / "src" / "main.tsx").write_text("export {};\n", encoding="utf-8")
    (frontend_root / "dist").mkdir()
    (frontend_root / "dist" / "index.html").write_text("<html></html>\n", encoding="utf-8")
    fingerprint = main_module.frontend_build_info(frontend_root)["source_fingerprint"]
    (frontend_root / "dist" / ".source-fingerprint").write_text(
        str(fingerprint) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "FRONTEND_ROOT", frontend_root)
    app = create_app(ui_config=object(), eval_config=object(), manager=FakeManager(tmp_path))
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/api/build-info"
    )

    result = asyncio.run(route.endpoint())

    assert result["dist_fingerprint"] == fingerprint
    assert result["current"] is True
