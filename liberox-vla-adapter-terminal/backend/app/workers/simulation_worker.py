"""Simulation session state machine and background workers."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

import eval_pickplace_direct as direct
from simulation_core import run_control_loop
from trajectory_utils import (
    TrajectoryRecorder,
    load_trajectory,
)

from ..core.config import UIConfig
from ..devices.spacemouse import SpaceMouseInput, SpaceMouseSnapshot, load_spacemouse_config
from ..domain.run import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    SimulationDraft,
    SimulationSession,
    utc_now,
)
from ..evaluation.libero_evaluator import LiberoEvaluator
from ..policies.vla_adapter import VLAAdapterPolicyProvider
from ..policies.catalog import PolicyCatalog
from ..recording.episode_recorder import EpisodeRecorderFactory
from ..services.controller_service import SpaceMouseControllerService
from ..services.task_catalog import ConfiguredTaskCatalog
from ..simulators.libero_x import LiberoXSimulator
from ..storage.files import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    safe_artifacts,
    scan_trajectories,
)
from ..storage.repositories import RunRepository


LOGGER = logging.getLogger("liberox.backend.worker")
SESSION_MANIFEST_SCHEMA_VERSION = 2
SESSION_SUMMARY_SCHEMA_VERSION = 2


class PreviewService:
    """One task-aware read-only MuJoCo renderer shared by drafts and sessions."""

    def __init__(self, manager: "SimulationManager"):
        self.manager = manager
        self.states: queue.Queue[
            tuple[SimulationSession | SimulationDraft, np.ndarray, str, int | None]
        ] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="ui-preview")
        self.thread.start()

    def submit(
        self,
        target: SimulationSession | SimulationDraft,
        state: Any,
        revision: int | None = None,
    ) -> None:
        item = (target, np.asarray(state).copy(), target.task_id, revision)
        try:
            self.states.put_nowait(item)
        except queue.Full:
            try:
                self.states.get_nowait()
            except queue.Empty:
                pass
            try:
                self.states.put_nowait(item)
            except queue.Full:
                pass

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            LOGGER.warning("Preview service did not stop within 30 seconds")

    def _run(self) -> None:
        config = self.manager.eval_config
        env = None
        env_task_id: str | None = None
        period = 1.0 / self.manager.ui_config.preview_fps
        try:
            last_render = 0.0
            while not self.stop_event.is_set():
                try:
                    target, state, task_id, revision = self.states.get(timeout=0.1)
                except queue.Empty:
                    continue
                delay = period - (time.monotonic() - last_render)
                if delay > 0 and self.stop_event.wait(delay):
                    break
                while True:
                    try:
                        target, state, task_id, revision = self.states.get_nowait()
                    except queue.Empty:
                        break
                try:
                    if env is None or env_task_id != task_id:
                        if env is not None:
                            self.manager.simulator.close(env)
                            env = None
                        bddl, _ = self.manager.catalog.paths(task_id)
                        env = self.manager.simulator.create(
                            bddl,
                            config,
                            max_steps=10000,
                            seed=config.seed,
                        )
                        env_task_id = task_id
                    self.manager.simulator.restore(env, state)
                    frame = self.manager.simulator.render_operator_preview(
                        env,
                        self.manager.ui_config.preview_width,
                        self.manager.ui_config.preview_height,
                    )
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, self.manager.ui_config.jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError("Preview JPEG encoding failed")
                    with self.manager.lock:
                        if isinstance(target, SimulationDraft):
                            if (
                                self.manager.draft is target
                                and target.preview_revision == revision
                            ):
                                target.latest_jpeg = encoded.tobytes()
                                target.preview_status = "READY"
                                target.error = None
                        else:
                            target.latest_jpeg = encoded.tobytes()
                            target.latest_frame_version += 1
                            target.preview_ready = True
                            target.preview_error = None
                            target.preview_event.set()
                except Exception as exc:
                    LOGGER.exception("Preview frame failed for %s", target.id)
                    message = f"{type(exc).__name__}: {exc}"
                    with self.manager.lock:
                        if isinstance(target, SimulationDraft):
                            if (
                                self.manager.draft is target
                                and target.preview_revision == revision
                            ):
                                target.preview_status = "ERROR"
                                target.error = message
                        else:
                            target.preview_error = message
                            target.preview_event.set()
                    if env is not None:
                        self.manager.simulator.close(env)
                        env = None
                        env_task_id = None
                last_render = time.monotonic()
        except Exception as exc:
            LOGGER.exception("Preview service failed")
        finally:
            if env is not None:
                self.manager.simulator.close(env)


class SimulationManager:
    def __init__(self, ui_config: UIConfig, eval_config: direct.EvalConfig):
        self.ui_config = ui_config
        self.eval_config = replace(eval_config, headless=True, env_only=False)
        direct.apply_runtime_environment(self.eval_config)
        _, liberox_root = direct.add_repo_paths(
            self.eval_config.vla_root,
            self.eval_config.liberox_root,
        )
        self.runtime = direct.load_runtime()
        self.simulator = LiberoXSimulator(self.runtime)
        self.recorder_factory = EpisodeRecorderFactory()
        self.evaluator = LiberoEvaluator()
        self.catalog = ConfiguredTaskCatalog(
            self.runtime,
            liberox_root,
            self.eval_config,
            self.ui_config.additional_tasks,
        )
        self.policy_catalog = PolicyCatalog(
            self.ui_config.policy_registry,
            str(self.eval_config.checkpoint),
            self.eval_config.stats_key,
        )
        self.provider = VLAAdapterPolicyProvider(
            self.runtime, self.eval_config, self.policy_catalog
        )
        try:
            self.spacemouse_config = load_spacemouse_config()
            self.spacemouse_config_error: str | None = None
        except Exception as exc:
            self.spacemouse_config = None
            self.spacemouse_config_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("SpaceMouse configuration unavailable: %s", self.spacemouse_config_error)
        self.lock = threading.RLock()
        self.controller = (
            None
            if self.spacemouse_config is None
            else SpaceMouseControllerService(self.spacemouse_config)
        )
        self.sessions: dict[str, SimulationSession] = {}
        self.legacy_sessions: dict[str, dict[str, Any]] = {}
        self.draft: SimulationDraft | None = None
        self.active_session_id: str | None = None
        self._trajectory_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
        self._frame_env = None
        self._frame_task_id: str | None = None
        self._frame_lock = threading.Lock()
        self.ui_config.output_root.mkdir(parents=True, exist_ok=True)
        self.repository = RunRepository(
            self.ui_config.catalog_path, self.ui_config.project_id
        )
        self.preview = PreviewService(self)
        self.refresh_history()

    def refresh_history(self) -> None:
        roots = tuple(dict.fromkeys((*self.ui_config.scan_roots, self.ui_config.output_root)))
        indexed = scan_trajectories(roots)
        indexed_ids = {item["id"] for item in indexed}
        # UI setup failures may intentionally have only a manifest/summary and
        # no trajectory. Keep those records visible and deletable after restart.
        manifest_paths = list(self.ui_config.output_root.rglob("run.json"))
        manifest_paths.extend(self.ui_config.output_root.glob("*/session.json"))
        for manifest in manifest_paths:
            directory = manifest.parent
            if directory.is_symlink() or manifest.is_symlink():
                continue
            if not manifest.is_file():
                continue
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            session_id = manifest_data.get("id")
            if not isinstance(session_id, str) or not session_id or session_id in indexed_ids:
                continue
            summary_data: dict[str, Any] = {}
            summary_path = directory / "summary.json"
            if summary_path.is_file() and not summary_path.is_symlink():
                try:
                    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_summary, dict):
                        summary_data = loaded_summary
                except Exception:
                    pass
            item = self._public_from_persisted(directory, manifest_data, summary_data)
            indexed.append(item)
            indexed_ids.add(session_id)
        with self.lock:
            self.legacy_sessions = {
                item["id"]: item
                for item in indexed
                if item["id"] not in self.sessions
            }
            for item in self.legacy_sessions.values():
                directory = Path(item["output_dir"])
                manifest = directory / "run.json"
                if not manifest.is_file():
                    manifest = directory / "session.json"
                managed = False
                try:
                    directory.resolve().relative_to(self.ui_config.output_root.resolve())
                    managed = (
                        not directory.is_symlink()
                        and manifest.is_file()
                        and json.loads(manifest.read_text(encoding="utf-8")).get("id") == item["id"]
                    )
                except (Exception, ValueError):
                    managed = False
                item["managed"] = managed
                item["legacy"] = not managed
                task_id = item.get("task_id")
                if task_id is not None:
                    try:
                        self.catalog.entry(str(task_id))
                    except ValueError:
                        task_id = None
                if task_id is None:
                    task_id = self.catalog.resolve_id(
                        item.get("level"), item.get("task_name")
                    )
                item["task_id"] = task_id
                item["branchable"] = bool(
                    item["kind"] == "original"
                    and task_id is not None
                    and item.get("trajectory")
                )

    def bootstrap(self) -> dict[str, Any]:
        return {
            "config": {
                "max_steps": self.eval_config.max_steps,
                "open_loop_steps": self.eval_config.open_loop_steps,
                "control_hz": self.eval_config.control_hz,
                "video_fps": self.eval_config.video_fps,
                "preview": {
                    "width": self.ui_config.preview_width,
                    "height": self.ui_config.preview_height,
                    "fps": self.ui_config.preview_fps,
                    "layout": "2x2",
                    "stream_width": self.ui_config.preview_width * 2,
                    "stream_height": self.ui_config.preview_height * 2,
                    "cameras": [
                        {"id": "agentview", "label": "主视角", "policy_input": True},
                        {"id": "robot0_eye_in_hand", "label": "腕部视角", "policy_input": True},
                        {"id": "oblique_minus_45", "label": "−45° 斜视角", "policy_input": False},
                        {"id": "oblique_plus_45", "label": "+45° 斜视角", "policy_input": False},
                    ],
                    "recorded_cameras": ["agentview", "robot0_eye_in_hand"],
                },
                "manual": {
                    "translation_gain": self.ui_config.manual_translation_gain,
                    "rotation_gain": self.ui_config.manual_rotation_gain,
                },
                "spacemouse": {
                    "configured": self.spacemouse_config is not None,
                    "dependency_version": SpaceMouseInput.dependency_version(),
                    "config_error": self.spacemouse_config_error,
                    "device_name": None
                    if self.spacemouse_config is None
                    else self.spacemouse_config.device_name,
                    "vendor_id": None
                    if self.spacemouse_config is None
                    else self.spacemouse_config.expected_vendor_id,
                    "product_id": None
                    if self.spacemouse_config is None
                    else self.spacemouse_config.expected_product_id,
                    "stale_timeout_ms": None
                    if self.spacemouse_config is None
                    else self.spacemouse_config.stale_timeout_ms,
                    "state": None if self.controller is None else self.controller.status()["state"],
                },
            },
            "model": self.provider.metadata(),
            "policy_catalog": self.policy_catalog.list_policies(),
            "task": self.catalog.metadata(self.catalog.default_task_id),
            "task_catalog": self.catalog.list_tasks(),
            "capabilities": {
                "model_switching": True,
                "task_switching": True,
                "pause": False,
                "step": False,
                "branch_depth": 1,
                "manual_control": True,
                "manual_sources": ["spacemouse"],
            },
        }

    def controller_status(self) -> dict[str, Any]:
        if self.controller is None:
            return {
                "state": "ERROR",
                "connected": False,
                "calibrated": False,
                "calibration_progress": 0.0,
                "message": "SpaceMouse 配置不可用",
                "error": self.spacemouse_config_error,
                "armed_session_id": None,
                "latency_ms": None,
                "latency_level": None,
                "stale": True,
            }
        return self.controller.status()

    def calibrate_controller(self) -> dict[str, Any]:
        with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("Cannot calibrate while a simulation is active")
        if self.controller is None:
            raise RuntimeError(f"SpaceMouse unavailable: {self.spacemouse_config_error}")
        return self.controller.start_calibration()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.lock:
            items = [record.public(safe_artifacts(record.output_dir)) for record in self.sessions.values()]
            items.extend(self.legacy_sessions.values())
        return sorted(
            items,
            key=lambda item: (item.get("created_at") or "", item["id"]),
            reverse=True,
        )

    def dataset_summary(self) -> dict[str, Any]:
        summary = self.repository.summary()
        summary["dataset_root"] = str(self.ui_config.dataset_root)
        summary["catalog"] = str(self.ui_config.catalog_path)
        summary["legacy_indexed"] = sum(
            1 for item in self.legacy_sessions.values() if item.get("legacy")
        )
        return summary

    def get_public(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id in self.sessions:
                record = self.sessions[session_id]
                return record.public(safe_artifacts(record.output_dir))
            if session_id in self.legacy_sessions:
                return dict(self.legacy_sessions[session_id])
        raise KeyError(session_id)

    def _new_record(
        self,
        *,
        kind: str,
        max_steps: int,
        open_loop_steps: int,
        policy_id: str = "base",
        task_id: str | None = None,
        parent: dict[str, Any] | None = None,
        resume_step: int | None = None,
        control_mode: str = "policy",
        manual_translation_gain: float | None = None,
        manual_rotation_gain: float | None = None,
    ) -> SimulationSession:
        if parent is not None:
            task_id = parent.get("task_id")
            if not task_id:
                raise ValueError("Source trajectory task is not available in the UI catalog")
            policy_id = str(parent.get("policy_id") or "base")
        task_id = task_id or self.catalog.default_task_id
        task = self.catalog.metadata(task_id)
        policy = self._policy_entry(policy_id)
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d_%H%M%S")
        task_group = str(task["task_name"]).lower()
        output_dir = (
            self.ui_config.output_root
            / task_group
            / now.strftime("%Y-%m-%d")
            / f"{stamp}__{session_id}"
        )
        episode_dir = output_dir / "episodes" / "episode_000"
        episode_dir.mkdir(parents=True, exist_ok=False)
        record = SimulationSession(
            id=session_id,
            kind=kind,
            output_dir=output_dir,
            max_steps=max_steps,
            open_loop_steps=open_loop_steps,
            policy_id=policy.policy_id,
            policy_label=policy.label,
            policy_base_checkpoint=policy.base_checkpoint,
            policy_overlay=None if policy.manifest is None else str(policy.manifest),
            policy_compatibility_sha256=policy.compatibility_sha256,
            task_id=task_id,
            task_level=task["level"],
            task_name=task["task_name"],
            task_prompt=task["prompt"],
            parent_session_id=None if parent is None else parent["id"],
            root_session_id=session_id if parent is None else (parent.get("root_session_id") or parent["id"]),
            source_trajectory=None if parent is None else parent["trajectory"],
            resume_step=resume_step,
            control_mode=control_mode,
            manual_source="spacemouse" if control_mode == "manual" else None,
            manual_translation_gain=manual_translation_gain,
            manual_rotation_gain=manual_rotation_gain,
            spacemouse_deadman_ms=(
                self.spacemouse_config.stale_timeout_ms
                if control_mode == "manual" and self.spacemouse_config is not None
                else None
            ),
            status="LOADING",
            branchable=parent is None,
            current_step=0 if resume_step is None else resume_step,
            state_count=0 if resume_step is None else resume_step + 1,
            action_count=0 if resume_step is None else resume_step,
        )
        self._persist_effective_config(record)
        return record

    def _claim(self, record: SimulationSession) -> None:
        with self.lock:
            if self.draft is not None:
                raise RuntimeError("Cancel or start the current draft before creating a session")
            if self.active_session_id is not None:
                active = self.sessions.get(self.active_session_id)
                if active is not None and active.status in ACTIVE_STATES:
                    raise RuntimeError(f"Session {active.id} is still active")
            self.sessions[record.id] = record
            self.active_session_id = record.id
            self._persist_manifest(record)

    @staticmethod
    def _validate_session_values(max_steps: int, open_loop_steps: int) -> None:
        if not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be in [1, 10000]")
        if not 1 <= open_loop_steps <= 8:
            raise ValueError("open_loop_steps must be in [1, 8]")

    def _policy_entry(self, policy_id: str):
        catalog = getattr(self, "policy_catalog", None)
        if catalog is not None:
            return catalog.entry(policy_id)
        if policy_id != "base":
            raise ValueError(f"Unknown policy_id: {policy_id}")
        checkpoint = str(getattr(getattr(self, "eval_config", None), "checkpoint", "unknown"))
        return SimpleNamespace(
            policy_id="base", label="VLA-Adapter · Object-Pro（基础模型）",
            base_checkpoint=checkpoint, manifest=None,
            compatibility_sha256=None,
        )

    def _start_record(self, record: SimulationSession) -> dict[str, Any]:
        self._claim(record)
        record.thread = threading.Thread(
            target=self._run_session,
            args=(record,),
            daemon=True,
            name=f"simulation-{record.id}",
        )
        record.thread.start()
        return record.public(safe_artifacts(record.output_dir))

    def create_original(
        self,
        max_steps: int,
        open_loop_steps: int,
        task_id: str | None = None,
        policy_id: str = "base",
        initial_jpeg: bytes | None = None,
    ) -> dict[str, Any]:
        self._validate_session_values(max_steps, open_loop_steps)
        record = self._new_record(
            kind="original",
            max_steps=max_steps,
            open_loop_steps=open_loop_steps,
            task_id=task_id,
            policy_id=policy_id,
        )
        if initial_jpeg is not None:
            record.latest_jpeg = initial_jpeg
            record.latest_frame_version = 1
        try:
            return self._start_record(record)
        except Exception:
            shutil.rmtree(record.output_dir, ignore_errors=True)
            raise

    def _draft_public(self, draft: SimulationDraft) -> dict[str, Any]:
        return draft.public(self.catalog.metadata(draft.task_id))

    def get_draft(self) -> dict[str, Any] | None:
        with self.lock:
            return None if self.draft is None else self._draft_public(self.draft)

    def _prepare_draft(self, draft: SimulationDraft, revision: int) -> None:
        try:
            state = self.catalog.initial_state(draft.task_id)
            with self.lock:
                if self.draft is not draft or draft.preview_revision != revision:
                    return
                draft.preview_status = "RENDERING"
                draft.error = None
            self.preview.submit(draft, state, revision)
        except Exception as exc:
            LOGGER.exception("Draft preview preparation failed")
            with self.lock:
                if self.draft is draft and draft.preview_revision == revision:
                    draft.preview_status = "ERROR"
                    draft.error = f"{type(exc).__name__}: {exc}"

    def _launch_draft_preview(self, draft: SimulationDraft) -> None:
        threading.Thread(
            target=self._prepare_draft,
            args=(draft, draft.preview_revision),
            daemon=True,
            name=f"draft-preview-{draft.id}",
        ).start()

    def create_draft(
        self, task_id: str, max_steps: int, open_loop_steps: int, policy_id: str = "base"
    ) -> dict[str, Any]:
        self._validate_session_values(max_steps, open_loop_steps)
        self.catalog.entry(task_id)
        policy = self._policy_entry(policy_id)
        with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("Cannot create a draft while a simulation is active")
            draft = SimulationDraft(
                id=uuid.uuid4().hex[:12],
                task_id=task_id,
                max_steps=max_steps,
                open_loop_steps=open_loop_steps,
                policy_id=policy.policy_id,
                policy_label=policy.label,
            )
            self.draft = draft
            public = self._draft_public(draft)
        self._launch_draft_preview(draft)
        return public

    def update_draft(
        self,
        *,
        task_id: str | None = None,
        max_steps: int | None = None,
        open_loop_steps: int | None = None,
        policy_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("Cannot modify a draft while a simulation is active")
            if self.draft is None:
                raise KeyError("draft")
            draft = self.draft
            next_task_id = task_id if task_id is not None else draft.task_id
            next_max_steps = max_steps if max_steps is not None else draft.max_steps
            next_open_loop = (
                open_loop_steps if open_loop_steps is not None else draft.open_loop_steps
            )
            next_policy_id = policy_id if policy_id is not None else draft.policy_id
            self._validate_session_values(next_max_steps, next_open_loop)
            self.catalog.entry(next_task_id)
            policy = self._policy_entry(next_policy_id)
            task_changed = next_task_id != draft.task_id
            draft.task_id = next_task_id
            draft.max_steps = next_max_steps
            draft.open_loop_steps = next_open_loop
            draft.policy_id = policy.policy_id
            draft.policy_label = policy.label
            if task_changed:
                draft.preview_revision += 1
                draft.preview_status = "PREPARING"
                draft.error = None
            public = self._draft_public(draft)
        if task_changed:
            self._launch_draft_preview(draft)
        return public

    def draft_preview(self) -> bytes:
        with self.lock:
            if self.draft is None:
                raise KeyError("draft")
            if self.draft.latest_jpeg is None:
                raise FileNotFoundError("Draft preview is not ready")
            return self.draft.latest_jpeg

    def discard_draft(self) -> bool:
        with self.lock:
            existed = self.draft is not None
            self.draft = None
            return existed

    def start_draft(self) -> dict[str, Any]:
        with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("Cannot start a draft while a simulation is active")
            if self.draft is None:
                raise KeyError("draft")
            draft = self.draft
            if draft.preview_status != "READY" or draft.latest_jpeg is None:
                raise RuntimeError("Draft preview must be ready before starting")
            self.draft = None
        try:
            return self.create_original(
                draft.max_steps,
                draft.open_loop_steps,
                task_id=draft.task_id,
                policy_id=draft.policy_id,
                initial_jpeg=draft.latest_jpeg,
            )
        except Exception:
            with self.lock:
                if self.draft is None:
                    self.draft = draft
            raise

    def create_branch(
        self,
        parent_id: str,
        resume_step: int,
        control_mode: str,
        open_loop_steps: int,
        translation_gain: float | None = None,
        rotation_gain: float | None = None,
    ) -> dict[str, Any]:
        parent = self.get_public(parent_id)
        if parent["kind"] != "original" or not parent.get("branchable"):
            raise ValueError("Only a completed original trajectory may create one-level branches")
        if parent["status"] not in TERMINAL_STATES:
            raise ValueError("The source session must finish before branching")
        if control_mode not in {"policy", "manual"}:
            raise ValueError("control_mode must be 'policy' or 'manual'")
        if control_mode == "policy":
            if translation_gain is not None or rotation_gain is not None:
                raise ValueError("manual gains must be omitted for policy branches")
        else:
            translation_gain = (
                self.ui_config.manual_translation_gain
                if translation_gain is None
                else float(translation_gain)
            )
            rotation_gain = (
                self.ui_config.manual_rotation_gain
                if rotation_gain is None
                else float(rotation_gain)
            )
            if not 0.05 <= translation_gain <= 1.0:
                raise ValueError("translation_gain must be in [0.05, 1.0]")
            if not 0.05 <= rotation_gain <= 1.0:
                raise ValueError("rotation_gain must be in [0.05, 1.0]")
            if self.controller is None:
                raise RuntimeError(f"SpaceMouse unavailable: {self.spacemouse_config_error}")
            controller_state = self.controller.status()["state"]
            if controller_state != "READY":
                raise RuntimeError(
                    "SpaceMouse must be connected and calibrated before takeover "
                    f"(current state: {controller_state})"
                )
        if not 1 <= open_loop_steps <= 8:
            raise ValueError("open_loop_steps must be in [1, 8]")
        action_count = int(parent["action_count"])
        if not 0 <= resume_step < action_count:
            raise ValueError(f"resume_step must be in [0, {max(0, action_count - 1)}]")
        record = self._new_record(
            kind="branch",
            max_steps=action_count,
            open_loop_steps=open_loop_steps,
            parent=parent,
            resume_step=resume_step,
            control_mode=control_mode,
            manual_translation_gain=translation_gain,
            manual_rotation_gain=rotation_gain,
        )
        try:
            self._copy_branch_source(record, parent)
            self._install_resume_preview(record, parent)
            self._claim(record)
        except Exception:
            shutil.rmtree(record.output_dir, ignore_errors=True)
            raise
        record.thread = threading.Thread(
            target=self._run_session,
            args=(record,),
            daemon=True,
            name=f"simulation-{record.id}",
        )
        record.thread.start()
        return record.public(safe_artifacts(record.output_dir))

    def stop(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            record = self.sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            if record.status in TERMINAL_STATES:
                return record.public(safe_artifacts(record.output_dir))
            record.stop_event.set()
            record.status = "STOPPING"
            self._persist_manifest(record)
            return record.public(safe_artifacts(record.output_dir))

    def manual_connect(self, session_id: str) -> None:
        record = self._manual_record(session_id)
        record.manual_connected = True

    def manual_settings(
        self,
        session_id: str,
        translation_gain: float,
        rotation_gain: float,
    ) -> None:
        record = self._manual_record(session_id)
        translation_gain = float(translation_gain)
        rotation_gain = float(rotation_gain)
        if not 0.05 <= translation_gain <= 1.0:
            raise ValueError("translation_gain must be in [0.05, 1.0]")
        if not 0.05 <= rotation_gain <= 1.0:
            raise ValueError("rotation_gain must be in [0.05, 1.0]")
        with self.lock:
            record.manual_translation_gain = translation_gain
            record.manual_rotation_gain = rotation_gain
        if self.controller is not None:
            self.controller.set_gains(record.id, translation_gain, rotation_gain)

    def manual_disconnect(self, session_id: str) -> None:
        record = self._manual_record(session_id)
        record.manual_connected = False
        if record.status in ACTIVE_STATES:
            record.stop_event.set()

    def _manual_record(self, session_id: str) -> SimulationSession:
        with self.lock:
            record = self.sessions.get(session_id)
            if record is None or record.control_mode != "manual":
                raise ValueError("Session is not a manual branch")
            return record

    def _copy_branch_source(self, record: SimulationSession, parent: dict[str, Any]) -> None:
        source = Path(parent["trajectory"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source trajectory not found: {source}")
        destination = record.episode_dir / "source_trajectory.npz"
        temporary = record.episode_dir / ".source_trajectory.npz.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        record.source_trajectory = str(destination.resolve())

    def _install_resume_preview(
        self, record: SimulationSession, parent: dict[str, Any]
    ) -> None:
        assert record.resume_step is not None
        artifacts = parent.get("artifacts", {})
        video_value = next(
            (
                value
                for name, value in artifacts.items()
                if name == "agentview.mp4" or name.endswith("/agentview.mp4")
            ),
            None,
        )
        if video_value is None:
            return
        capture = cv2.VideoCapture(str(video_value))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, record.resume_step)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.ui_config.jpeg_quality]
        )
        if ok:
            content = encoded.tobytes()
            atomic_write_bytes(record.episode_dir / "resume_preview.jpg", content)
            record.latest_jpeg = content
            record.latest_frame_version += 1

    def _persist_effective_config(self, record: SimulationSession) -> None:
        """Write the immutable run inputs once, before the worker starts."""
        eval_config = getattr(self, "eval_config", None)
        atomic_write_yaml(
            record.output_dir / "config.yaml",
            {
                "schema_version": 1,
                "project_id": getattr(self.ui_config, "project_id", "libero_x_vla"),
                "task": {
                    "task_id": record.task_id,
                    "level": record.task_level,
                    "task_name": record.task_name,
                    "prompt": record.task_prompt,
                    "init_state_index": 0,
                },
                "policy": {
                    "provider": "vla_adapter",
                    "policy_id": record.policy_id,
                    "label": record.policy_label,
                    "base_checkpoint": record.policy_base_checkpoint,
                    "overlay": record.policy_overlay,
                    "compatibility_sha256": record.policy_compatibility_sha256,
                    "stats_key": getattr(eval_config, "stats_key", None),
                },
                "simulation": {
                    "max_steps": record.max_steps,
                    "open_loop_steps": record.open_loop_steps,
                    "control_hz": getattr(eval_config, "control_hz", 20),
                    "seed": getattr(eval_config, "seed", 0),
                    "control_mode": record.control_mode,
                    "resume_step": record.resume_step,
                },
            },
        )

    def _persist_manifest(self, record: SimulationSession) -> None:
        trajectory = None
        if record.trajectory is not None:
            try:
                trajectory = Path(record.trajectory).resolve().relative_to(
                    record.output_dir.resolve()
                ).as_posix()
            except ValueError:
                trajectory = str(Path(record.trajectory).resolve())
        manifest = {
                "schema_version": SESSION_MANIFEST_SCHEMA_VERSION,
                "managed_by": "liberox_data_studio",
                "project_id": getattr(getattr(self, "ui_config", None), "project_id", "libero_x_vla"),
                "id": record.id,
                "kind": record.kind,
                "task_id": record.task_id,
                "level": record.task_level,
                "task_name": record.task_name,
                "task": record.task_prompt,
                "status": record.status,
                "created_at": record.created_at,
                "completed_at": record.completed_at,
                "parent_session_id": record.parent_session_id,
                "root_session_id": record.root_session_id,
                "resume_step": record.resume_step,
                "control_mode": record.control_mode,
                "policy_id": record.policy_id,
                "policy_label": record.policy_label,
                "policy_base_checkpoint": record.policy_base_checkpoint,
                "policy_overlay": record.policy_overlay,
                "policy_compatibility_sha256": record.policy_compatibility_sha256,
                "current_step": record.current_step,
                "max_steps": record.max_steps,
                "open_loop_steps": record.open_loop_steps,
                "state_count": record.state_count,
                "action_count": record.action_count,
                "policy_queries": record.policy_queries,
                "trajectory": trajectory,
                "success": record.success,
                "error": record.error,
            }
        atomic_write_json(record.output_dir / "run.json", manifest)
        repository = getattr(self, "repository", None)
        if repository is not None:
            repository.upsert(manifest, record.output_dir)

    @staticmethod
    def _public_from_persisted(
        directory: Path,
        manifest: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Rebuild the API view without duplicating it in ``session.json``."""
        result = (
            summary["result"]
            if isinstance(summary.get("result"), dict)
            else summary
        )
        context = (
            summary["context"]
            if isinstance(summary.get("context"), dict)
            else summary
        )
        timing = (
            summary["timing"]
            if isinstance(summary.get("timing"), dict)
            else summary
        )
        branch = (
            summary["branch"]
            if isinstance(summary.get("branch"), dict)
            else summary
        )
        controller = (
            summary.get("controller")
            if isinstance(summary.get("controller"), dict)
            else {}
        )
        trajectory_name = manifest.get("trajectory")
        trajectory = None
        if isinstance(trajectory_name, str) and trajectory_name:
            candidate = (directory / trajectory_name).resolve()
            try:
                candidate.relative_to(directory.resolve())
            except ValueError:
                candidate = Path("/__invalid_trajectory__")
            if candidate.is_file():
                trajectory = str(candidate)
        action_count = int(result.get("steps", manifest.get("action_count", 0)) or 0)
        state_count = int(manifest.get("state_count", action_count + (1 if action_count else 0)) or 0)
        return {
            "id": manifest["id"],
            "kind": manifest.get("kind", "original"),
            "task_id": context.get("task_id", manifest.get("task_id")),
            "parent_session_id": branch.get(
                "parent_session_id", manifest.get("parent_session_id")
            ),
            "root_session_id": manifest.get("root_session_id"),
            "source_trajectory": None,
            "resume_step": branch.get("resume_step", manifest.get("resume_step")),
            "control_mode": summary.get(
                "control_mode", manifest.get("control_mode", "policy")
            ),
            "policy_id": manifest.get("policy_id", "base"),
            "policy_label": manifest.get("policy_label"),
            "policy_base_checkpoint": manifest.get(
                "policy_base_checkpoint", manifest.get("checkpoint")
            ),
            "policy_overlay": manifest.get("policy_overlay"),
            "policy_compatibility_sha256": manifest.get(
                "policy_compatibility_sha256"
            ),
            "manual_source": (
                "spacemouse" if controller else manifest.get("manual_source")
            ),
            "manual_translation_gain": controller.get(
                "translation_gain", manifest.get("manual_translation_gain")
            ),
            "manual_rotation_gain": controller.get(
                "rotation_gain", manifest.get("manual_rotation_gain")
            ),
            "spacemouse_status": None,
            "spacemouse_connected": None,
            "spacemouse_stale": None,
            "spacemouse_latency_ms": None,
            "spacemouse_deadman_ms": None,
            "status": manifest.get("status", summary.get("status", "ERROR")),
            "created_at": manifest.get("created_at", summary.get("created_at")),
            "completed_at": manifest.get("completed_at", summary.get("completed_at")),
            "task": context.get("task", manifest.get("task")),
            "task_name": context.get("task_name", manifest.get("task_name")),
            "level": context.get("level", manifest.get("level")),
            "checkpoint": context.get("checkpoint", manifest.get("checkpoint")),
            "max_steps": int(result.get("target_steps", manifest.get("max_steps", action_count)) or 0),
            "open_loop_steps": summary.get(
                "open_loop_steps", manifest.get("open_loop_steps")
            ),
            "current_step": int(manifest.get("current_step", action_count) or 0),
            "state_count": state_count,
            "action_count": action_count,
            "policy_queries": int(result.get("policy_queries", manifest.get("policy_queries", 0)) or 0),
            "success": bool(result.get("success", False)),
            "error": result.get("error", manifest.get("error")),
            "stopped_reason": result.get("stopped_reason"),
            "measured_control_hz": timing.get(
                "measured_control_hz", manifest.get("measured_control_hz")
            ),
            "simulated_duration_seconds": float(
                timing.get(
                    "simulated_duration_seconds",
                    manifest.get("simulated_duration_seconds", 0.0),
                )
                or 0.0
            ),
            "output_dir": str(directory.resolve()),
            "trajectory": trajectory,
            "artifacts": safe_artifacts(directory),
            "branchable": False,
            "legacy": False,
            "managed": True,
            "preparation_phase": "completed" if not result.get("error") else "failed",
            "preparation_message": result.get("error"),
            "countdown_remaining": None,
            "preview_ready": False,
            "preparation_timing": {
                "total_seconds": timing.get("preparation_seconds"),
                "model_load_seconds": timing.get("model_load_seconds"),
                "model_cache_hit": float(bool(timing.get("model_cache_hit", False))),
            },
        }

    def _set_status(self, record: SimulationSession, status: str) -> None:
        with self.lock:
            record.status = status
            self._persist_manifest(record)

    def _set_spacemouse_status(
        self,
        record: SimulationSession,
        status: str,
        *,
        connected: bool | None = None,
        stale: bool | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self.lock:
            record.spacemouse_status = status
            record.spacemouse_connected = connected
            record.spacemouse_stale = stale
            record.spacemouse_latency_ms = latency_ms

    def _spacemouse_action(self, record: SimulationSession, step: int) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("SpaceMouse controller service is unavailable")
        snapshot: SpaceMouseSnapshot = self.controller.snapshot(record.id)
        latency_ms = (
            None
            if snapshot.sample_age_seconds is None
            else snapshot.sample_age_seconds * 1000.0
        )
        if snapshot.error is not None:
            status = "error"
        elif not snapshot.connected:
            status = "disconnected"
        elif snapshot.stale:
            status = "stale"
        else:
            status = "ready"
        with self.lock:
            record.spacemouse_status = status
            record.spacemouse_connected = snapshot.connected
            record.spacemouse_stale = snapshot.stale
            record.spacemouse_latency_ms = latency_ms
            row: dict[str, Any] = {
                "step": step,
                "sequence": snapshot.sequence,
                "captured_monotonic": snapshot.captured_monotonic,
                "device_timestamp": snapshot.device_timestamp,
                "sample_age_ms": "" if latency_ms is None else latency_ms,
                "connected": snapshot.connected,
                "stale": snapshot.stale,
                "button_left": snapshot.buttons[0] if len(snapshot.buttons) > 0 else 0,
                "button_right": snapshot.buttons[1] if len(snapshot.buttons) > 1 else 0,
                "translation_gain": record.manual_translation_gain,
                "rotation_gain": record.manual_rotation_gain,
                "error": snapshot.error or "",
            }
            for prefix, values in (
                ("raw", snapshot.raw_axes),
                ("corrected", snapshot.corrected_axes),
                ("command", snapshot.command_axes),
            ):
                for axis, value in zip(("x", "y", "z", "rx", "ry", "rz"), values):
                    row[f"{prefix}_{axis}"] = value
            row["gripper_action"] = snapshot.action[6]
            record.spacemouse_samples.append(row)
        if snapshot.error is not None:
            raise RuntimeError(f"SpaceMouse reader failed: {snapshot.error}")
        return np.asarray(snapshot.action, dtype=np.float32)

    @staticmethod
    def _serializable_spacemouse_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
        diagnostics = dict(diagnostics or {})
        event_times = np.asarray(diagnostics.pop("event_times", []), dtype=np.float64)
        intervals_ms = np.diff(event_times) * 1000.0
        if len(intervals_ms):
            diagnostics["hid_interval_ms"] = {
                "p50": float(np.percentile(intervals_ms, 50)),
                "p95": float(np.percentile(intervals_ms, 95)),
                "p99": float(np.percentile(intervals_ms, 99)),
                "max": float(np.max(intervals_ms)),
            }
        else:
            diagnostics["hid_interval_ms"] = None
        return diagnostics

    def _persist_spacemouse_samples(self, record: SimulationSession) -> dict[str, Any]:
        """Save detailed samples as CSV and return a compact summary block."""
        rows = record.spacemouse_samples
        if rows:
            atomic_write_csv(
                record.episode_dir / "spacemouse_samples.csv",
                list(rows[0]),
                rows,
            )
        ages = np.asarray(
            [row["sample_age_ms"] for row in rows if row["sample_age_ms"] != ""],
            dtype=np.float64,
        )
        latency = None
        if len(ages):
            latency = {
                "p50_ms": float(np.percentile(ages, 50)),
                "p95_ms": float(np.percentile(ages, 95)),
                "max_ms": float(np.max(ages)),
            }
        assert self.spacemouse_config is not None
        calibration = record.spacemouse_calibration or {}
        calibration_result = calibration.get("result", {})
        diagnostics = record.spacemouse_diagnostics or {}
        hid_intervals = diagnostics.get("hid_interval_ms")
        if isinstance(hid_intervals, dict):
            hid_intervals = {
                "p95_ms": hid_intervals.get("p95"),
                "max_ms": hid_intervals.get("max"),
            }
        return {
            "type": "spacemouse",
            "device": self.spacemouse_config.device_name,
            "translation_gain": record.manual_translation_gain,
            "rotation_gain": record.manual_rotation_gain,
            "sample_count": len(rows),
            "sample_age_ms": latency,
            "calibration": {
                "id": calibration.get("id"),
                "bias": calibration_result.get("bias"),
                "movement_resets": calibration_result.get("movement_resets"),
            },
            "hid_interval_ms": hid_intervals,
            "reader_error": diagnostics.get("reader_error"),
        }

    def _run_session(self, record: SimulationSession) -> None:
        recorder: TrajectoryRecorder | None = None
        env = None
        source_trajectory: dict[str, np.ndarray] | None = None
        restore_error: float | None = None
        timing: dict[str, float | None] = {}
        controller_summary: dict[str, Any] | None = None
        prep_started = time.monotonic()
        task = self.catalog.entry(record.task_id)
        bddl, _ = self.catalog.paths(record.task_id)
        seed = self.eval_config.seed

        def phase(name: str, message: str) -> None:
            with self.lock:
                record.preparation_phase = name
                record.preparation_message = message
                record.preparation_timing[f"{name}_at_seconds"] = (
                    time.monotonic() - prep_started
                )
                self._persist_manifest(record)

        try:
            phase("loading_source", "读取轨迹与初始状态")
            if record.control_mode == "policy":
                phase("loading_model", "加载策略模型")
                model_was_loaded = self.provider.loaded
                model_load_started = time.monotonic()
                try:
                    self.provider.load(record.open_loop_steps, record.policy_id)
                finally:
                    with self.lock:
                        record.preparation_timing["model_load_seconds"] = (
                            time.monotonic() - model_load_started
                        )
                        record.preparation_timing["model_cache_hit"] = float(
                            model_was_loaded
                        )
                        self._persist_manifest(record)
            if record.kind == "original":
                initial_state = self.catalog.initial_state(record.task_id)
                recorder = self.recorder_factory.original(float(self.eval_config.control_hz))
            else:
                assert record.source_trajectory is not None and record.resume_step is not None
                source_trajectory, source_metadata = load_trajectory(
                    Path(record.source_trajectory)
                )
                seed = int(source_metadata.get("seed", seed))
                initial_state = source_trajectory["sim_state"][record.resume_step]
                recorder = self.recorder_factory.branch(
                    source_trajectory,
                    record.resume_step,
                    float(source_metadata.get("control_hz", self.eval_config.control_hz)),
                )

            session_config = replace(
                self.eval_config,
                max_steps=record.max_steps,
                open_loop_steps=record.open_loop_steps,
                trials=1,
                headless=True,
            )
            phase("warming_controller", "预热 MuJoCo 控制器")
            self.simulator.prewarm(bddl, initial_state, session_config)
            phase("creating_environment", "创建控制环境")
            env = self.simulator.create(
                bddl,
                session_config,
                max_steps=record.max_steps,
                seed=seed,
            )
            phase("restoring_state", "恢复所选帧状态")
            observation = self.simulator.restore(env, initial_state)
            observation = self.simulator.observations(env, observation)
            if record.kind == "original":
                recorder.record_initial(env, observation)
            else:
                restored = np.asarray(env.get_sim_state())
                expected = np.asarray(initial_state)
                restore_error = float(np.max(np.abs(restored - expected)))
                if restore_error > 1e-9:
                    raise RuntimeError(
                        "MuJoCo state restoration mismatch: "
                        f"max_abs_error={restore_error}"
                    )

            record.current_step = recorder.action_count
            record.state_count = recorder.state_count
            record.action_count = recorder.action_count
            phase("preparing_preview", "准备实时四视角")
            record.preview_event.clear()
            record.preview_error = None
            self.preview.submit(record, env.get_sim_state())
            if not record.preview_event.wait(20.0):
                raise TimeoutError("Preview service did not produce the first frame within 20 seconds")
            if record.preview_error:
                raise RuntimeError(f"Preview service failed: {record.preview_error}")
            record.preparation_timing["ready_seconds"] = time.monotonic() - prep_started

            if record.control_mode == "manual":
                if self.controller is None:
                    raise RuntimeError("SpaceMouse controller service is unavailable")
                assert record.manual_translation_gain is not None
                assert record.manual_rotation_gain is not None
                record.spacemouse_calibration = self.controller.calibration_snapshot()
                self._set_status(record, "READY")
                for remaining in (3, 2, 1):
                    with self.lock:
                        record.countdown_remaining = remaining
                        record.preparation_phase = "countdown"
                        record.preparation_message = f"{remaining} 秒后开始 SpaceMouse 接管"
                        self._persist_manifest(record)
                    if record.stop_event.wait(1.0):
                        break
                record.countdown_remaining = None
                if not record.stop_event.is_set():
                    self.controller.arm(
                        record.id,
                        record.manual_translation_gain,
                        record.manual_rotation_gain,
                    )
                    record.spacemouse_status = "armed"
                    record.spacemouse_connected = True
                    record.spacemouse_stale = False
            with self.lock:
                record.preparation_phase = "complete"
                record.preparation_message = None
                record.preparation_timing["total_seconds"] = time.monotonic() - prep_started
            self._set_status(record, "STOPPING" if record.stop_event.is_set() else "RUNNING")

            limiter = direct.RealTimeControlLimiter(
                float(getattr(env.env, "control_freq", session_config.control_hz)),
                True,
            )

            def query_policy(current_observation: dict[str, Any], _step: int):
                current_observation = self.simulator.observations(env, current_observation)
                return current_observation, self.provider.predict(
                    current_observation, task.prompt
                )

            def on_transition(_observation: dict[str, Any], step: int, success: bool) -> None:
                with self.lock:
                    record.current_step = step
                    record.action_count = recorder.action_count
                    record.state_count = recorder.state_count
                    record.success = self.evaluator.success(success)
                    record.policy_queries = len(recorder.inference_query_steps)
                self.preview.submit(record, env.get_sim_state())

            common = dict(
                env=env,
                recorder=recorder,
                initial_observation=observation,
                target_action_count=record.max_steps,
                rate_limiter=limiter,
                open_loop_steps=record.open_loop_steps,
                stop_requested=record.stop_event.is_set,
                on_transition=on_transition,
                stop_on_success=record.kind == "original",
                horizon_reason="source_horizon" if record.kind == "branch" else "max_steps",
            )
            if record.control_mode == "manual":
                result = run_control_loop(
                    **common,
                    action_source="human",
                    manual_query=lambda step: self._spacemouse_action(record, step),
                )
            else:
                result = run_control_loop(
                    **common,
                    action_source="policy_requery" if record.kind == "branch" else "policy",
                    policy_query=query_policy,
                    policy_action_transform=self.provider.process_action,
                )

            record.success = result.success
            record.policy_queries = len(recorder.inference_query_steps)
            record.stopped_reason = result.stopped_reason
            record.current_step = recorder.action_count
            record.action_count = recorder.action_count
            record.state_count = recorder.state_count
            timing = direct.summarize_control_timing(
                result.control_step_times, recorder.control_hz
            )
            record.measured_control_hz = timing["measured_control_hz"]
            record.simulated_duration_seconds = float(
                timing["simulated_duration_seconds"] or 0.0
            )
            record.preparation_phase = "postprocessing"
            record.preparation_message = "保存轨迹与结果"
            self._set_status(record, "POSTPROCESSING")
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.stopped_reason = "error"
            record.preparation_message = "准备或运行失败"
            LOGGER.exception("Session %s failed", record.id)
        finally:
            if record.control_mode == "manual" and self.controller is not None:
                record.spacemouse_diagnostics = self._serializable_spacemouse_diagnostics(
                    self.controller.diagnostics()
                )
                self.controller.disarm(record.id)
                if record.spacemouse_status != "error":
                    record.spacemouse_status = "stopped"

            suffix_start = record.resume_step or 0
            has_new_actions = bool(
                recorder is not None
                and recorder.action_count > (suffix_start if record.kind == "branch" else 0)
            )
            if env is not None and recorder is not None and has_new_actions:
                combined_pending = record.episode_dir / ".vla_views.pending.mp4"
                main_pending = record.episode_dir / ".agentview.pending.mp4"
                try:
                    direct.postprocess_recorded_trajectory(
                        runtime=self.runtime,
                        env=env,
                        recorder=recorder,
                        save_observations=True,
                        combined_path=combined_pending,
                        main_view_path=main_pending,
                        fps=self.eval_config.video_fps,
                        video_camera=self.eval_config.video_camera,
                        video_width=self.eval_config.video_width,
                        video_height=self.eval_config.video_height,
                        main_view_width=self.eval_config.main_view_video_width,
                        main_view_height=self.eval_config.main_view_video_height,
                    )
                    combined_pending.replace(record.episode_dir / "vla_views.mp4")
                    main_pending.replace(record.episode_dir / "agentview.mp4")
                except Exception as exc:
                    combined_pending.unlink(missing_ok=True)
                    main_pending.unlink(missing_ok=True)
                    record.error = record.error or (
                        f"{type(exc).__name__}: post-processing failed: {exc}"
                    )
                    record.stopped_reason = "error"
                    LOGGER.exception("Session %s video post-processing failed", record.id)
            if env is not None:
                self.simulator.close(env)

            if recorder is not None and recorder.state_count:
                try:
                    metadata = self._metadata(record, timing, restore_error)
                    staging = record.episode_dir / f".trajectory-staging-{uuid.uuid4().hex}"
                    staging.mkdir()
                    paths = self.recorder_factory.save(
                        recorder,
                        staging / "trajectory",
                        metadata,
                        save_observations=has_new_actions,
                        create_plot=record.kind == "original" and has_new_actions,
                        intervention_step=record.resume_step,
                        save_metadata_json=False,
                    )
                    if (
                        has_new_actions
                        and source_trajectory is not None
                        and record.resume_step is not None
                    ):
                        paths.update(self.recorder_factory.compare(
                            source_trajectory,
                            recorder.arrays(),
                            record.resume_step,
                            recorder.control_hz,
                            staging / "trajectory_action_comparison",
                            branch_label="re-inference"
                            if record.control_mode == "policy"
                            else "human takeover",
                        ))
                    published: dict[str, str] = {}
                    for key, value in paths.items():
                        source_path = Path(value)
                        destination = record.episode_dir / source_path.name
                        source_path.replace(destination)
                        published[key] = str(destination)
                    staging.rmdir()
                    record.trajectory = published["trajectory"]
                except Exception as exc:
                    record.error = record.error or (
                        f"{type(exc).__name__}: persistence failed: {exc}"
                    )
                    record.stopped_reason = "error"
                    LOGGER.exception("Session %s trajectory persistence failed", record.id)

            if record.manual_source == "spacemouse":
                try:
                    controller_summary = self._persist_spacemouse_samples(record)
                except Exception as exc:
                    record.error = record.error or (
                        f"{type(exc).__name__}: SpaceMouse persistence failed: {exc}"
                    )
                    record.stopped_reason = "error"

            record.completed_at = utc_now()
            record.status = "ERROR" if record.error else "COMPLETED"
            record.preparation_phase = "failed" if record.error else "completed"
            record.preparation_message = record.error if record.error else None
            record.countdown_remaining = None
            record.branchable = bool(
                record.kind == "original" and recorder is not None and recorder.action_count > 0
            )
            summary = self._summary(record, timing, restore_error, controller_summary)
            try:
                atomic_write_json(record.output_dir / "summary.json", summary)
            except Exception as exc:
                record.error = record.error or (
                    f"{type(exc).__name__}: final summary failed: {exc}"
                )
                record.status = "ERROR"
                LOGGER.exception("Session %s final summary persistence failed", record.id)
            try:
                self._persist_manifest(record)
            except Exception as exc:
                record.error = record.error or (
                    f"{type(exc).__name__}: final manifest failed: {exc}"
                )
                record.status = "ERROR"
                LOGGER.exception("Session %s final manifest persistence failed", record.id)
            finally:
                with self.lock:
                    if self.active_session_id == record.id:
                        self.active_session_id = None

    def _metadata(
        self,
        record: SimulationSession,
        timing: dict[str, float | None],
        restore_error: float | None,
    ) -> dict[str, Any]:
        bddl, _ = self.catalog.paths(record.task_id)
        provider_metadata = self.provider.metadata()
        return {
            "session_id": record.id,
            "parent_session_id": record.parent_session_id,
            "root_session_id": record.root_session_id,
            "source_trajectory": record.source_trajectory,
            "source_resume_step": record.resume_step,
            "target_total_steps": record.max_steps,
            "control_mode": record.control_mode,
            "manual_source": record.manual_source,
            "manual_translation_gain": record.manual_translation_gain,
            "manual_rotation_gain": record.manual_rotation_gain,
            "spacemouse_deadman_ms": record.spacemouse_deadman_ms,
            "spacemouse_latency_ms": record.spacemouse_latency_ms,
            "spacemouse_status": record.spacemouse_status,
            "spacemouse_calibration": record.spacemouse_calibration,
            "preparation_timing": dict(record.preparation_timing),
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "task_id": record.task_id,
            "task": record.task_prompt,
            "task_name": record.task_name,
            "level": record.task_level,
            "bddl": str(bddl),
            "checkpoint": str(self.eval_config.checkpoint)
            if record.control_mode == "policy"
            else None,
            "policy_id": record.policy_id,
            "policy_label": record.policy_label,
            "policy_base_checkpoint": record.policy_base_checkpoint,
            "policy_overlay": record.policy_overlay,
            "policy_compatibility_sha256": record.policy_compatibility_sha256,
            "policy_device": provider_metadata["model_device"]
            if record.control_mode == "policy"
            else None,
            "seed": self.eval_config.seed,
            "open_loop_steps": record.open_loop_steps,
            "control_hz": self.eval_config.control_hz,
            "realtime_control": True,
            "policy_queries": record.policy_queries,
            "success": record.success,
            "stopped_reason": record.stopped_reason,
            "error": record.error,
            "restore_max_abs_error": restore_error,
            **timing,
        }

    def _summary(
        self,
        record: SimulationSession,
        timing: dict[str, float | None],
        restore_error: float | None,
        controller: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the small human-facing outcome summary.

        Replay/configuration details live in ``trajectory.npz``'s embedded
        metadata.  This file intentionally carries only identity, context,
        outcome, and the performance values normally inspected by a user.
        """
        summary: dict[str, Any] = {
            "schema_version": SESSION_SUMMARY_SCHEMA_VERSION,
            "run_id": record.id,
            "kind": record.kind,
            "status": record.status,
            "control_mode": record.control_mode,
            "policy": {
                "policy_id": record.policy_id,
                "label": record.policy_label,
                "base_checkpoint": record.policy_base_checkpoint,
                "overlay": record.policy_overlay,
                "compatibility_sha256": record.policy_compatibility_sha256,
            },
            "task": record.task_prompt,
            "result": {
                "success": record.success,
                "steps": record.action_count,
                "target_steps": record.max_steps,
                "policy_queries": record.policy_queries,
                "stopped_reason": record.stopped_reason,
                "error": record.error,
            },
            "timing": {
                "simulated_duration_seconds": timing.get(
                    "simulated_duration_seconds", record.simulated_duration_seconds
                ),
                "measured_control_hz": timing.get(
                    "measured_control_hz", record.measured_control_hz
                ),
                "mean_control_interval_ms": self._seconds_to_ms(
                    timing.get("control_interval_mean_seconds")
                ),
                "max_control_interval_ms": self._seconds_to_ms(
                    timing.get("control_interval_max_seconds")
                ),
                "preparation_seconds": record.preparation_timing.get("total_seconds"),
                "model_load_seconds": record.preparation_timing.get("model_load_seconds"),
                "model_cache_hit": bool(
                    record.preparation_timing.get("model_cache_hit", 0.0)
                ),
            },
        }
        if record.kind == "branch":
            summary["branch"] = {
                "parent_session_id": record.parent_session_id,
                "root_session_id": record.root_session_id,
                "resume_step": record.resume_step,
                "restore_max_abs_error": restore_error,
            }
        if controller is not None:
            summary["controller"] = controller
        return summary

    @staticmethod
    def _seconds_to_ms(value: float | None) -> float | None:
        return None if value is None else float(value) * 1000.0

    def _trajectory_for(self, session_id: str):
        public = self.get_public(session_id)
        path_value = public.get("trajectory")
        if not path_value:
            raise ValueError("Session has no saved trajectory")
        path = str(Path(path_value).resolve())
        with self.lock:
            cached = self._trajectory_cache.get(path)
        if cached is None:
            cached = load_trajectory(Path(path))
            with self.lock:
                if len(self._trajectory_cache) >= 4:
                    self._trajectory_cache.pop(next(iter(self._trajectory_cache)))
                self._trajectory_cache[path] = cached
        return cached

    def frame_state(self, session_id: str, step: int) -> dict[str, Any]:
        trajectory, metadata = self._trajectory_for(session_id)
        state_count = len(trajectory["sim_state"])
        if not 0 <= step < state_count:
            raise IndexError(step)
        action_count = len(trajectory["raw_action"])
        action_index = min(step, max(0, action_count - 1))
        has_action = action_count > 0 and step < action_count
        return {
            "step": step,
            "time_seconds": float(trajectory["time_seconds"][step]),
            "eef_position_m": trajectory["eef_position"][step].tolist(),
            "eef_axis_angle_rad": trajectory["eef_axis_angle"][step].tolist(),
            "gripper_qpos": trajectory["gripper_qpos"][step].tolist(),
            "raw_action": trajectory["raw_action"][action_index].tolist() if has_action else None,
            "env_action": trajectory["env_action"][action_index].tolist() if has_action else None,
            "success": bool(trajectory["done"][:step].any()) if step else False,
            "control_hz": float(metadata.get("control_hz", self.eval_config.control_hz)),
        }

    def render_frame(self, session_id: str, step: int) -> bytes:
        trajectory, metadata = self._trajectory_for(session_id)
        task_id = metadata.get("task_id") or self.catalog.resolve_id(
            metadata.get("level"), metadata.get("task_name")
        )
        if not task_id:
            raise ValueError("Trajectory task is not available in the UI catalog")
        if not 0 <= step < len(trajectory["sim_state"]):
            raise IndexError(step)
        with self._frame_lock:
            if self._frame_env is None or self._frame_task_id != task_id:
                if self._frame_env is not None:
                    self.simulator.close(self._frame_env)
                    self._frame_env = None
                bddl, _ = self.catalog.paths(task_id)
                self._frame_env = self.simulator.create(
                    bddl,
                    self.eval_config,
                    max_steps=max(1, len(trajectory["sim_state"])),
                    seed=int(metadata.get("seed", self.eval_config.seed)),
                )
                self._frame_task_id = task_id
            self.simulator.restore(self._frame_env, trajectory["sim_state"][step])
            frame = self.simulator.render(
                self._frame_env,
                direct.MAIN_VIEW_CAMERA,
                self.ui_config.preview_width,
                self.ui_config.preview_height,
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, self.ui_config.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return encoded.tobytes()

    def artifact_path(self, session_id: str, name: str) -> Path:
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid artifact name")
        public = self.get_public(session_id)
        path_value = public.get("artifacts", {}).get(name)
        if not path_value:
            raise FileNotFoundError(name)
        path = Path(path_value).resolve()
        output_dir = Path(public["output_dir"]).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError:
            raise FileNotFoundError(name) from None
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def delete_session(self, session_id: str, confirm_session_id: str) -> None:
        if confirm_session_id != session_id:
            raise ValueError("confirm_session_id must exactly match the session id")
        with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("Cannot delete while a simulation is active")
            controller_state = self.controller_status()["state"]
            if controller_state in {"CALIBRATING", "ARMED"}:
                raise RuntimeError("Cannot delete while the controller is calibrating or armed")
            public = self.get_public(session_id)
            if not public.get("managed"):
                raise ValueError("Only UI-owned sessions can be deleted")
            if public.get("status") not in TERMINAL_STATES:
                raise RuntimeError("Only completed sessions can be deleted")
            directory = Path(public["output_dir"])
            if directory.is_symlink():
                raise ValueError("Refusing to delete a symbolic-link session directory")
            resolved = directory.resolve()
            try:
                resolved.relative_to(self.ui_config.output_root.resolve())
            except ValueError:
                raise ValueError("Session directory is outside the UI output root")
            manifest_path = resolved / "run.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise ValueError("Managed session marker is missing or unsafe")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("id") != session_id:
                raise ValueError("Session marker id does not match the requested id")
            shutil.rmtree(resolved)
            repository = getattr(self, "repository", None)
            if repository is not None:
                repository.delete(session_id)
            self.sessions.pop(session_id, None)
            self.legacy_sessions.pop(session_id, None)
            for cache_key in list(self._trajectory_cache):
                try:
                    Path(cache_key).resolve().relative_to(resolved)
                except ValueError:
                    continue
                self._trajectory_cache.pop(cache_key, None)

    def close(self) -> None:
        with self.lock:
            active_id = self.active_session_id
            if active_id and active_id in self.sessions:
                self.sessions[active_id].stop_event.set()
                thread = self.sessions[active_id].thread
            else:
                thread = None
        if thread is not None:
            # A synchronous CUDA call cannot be interrupted safely. Shutdown
            # waits for the worker to reach its next control boundary.
            thread.join()
        self.preview.close()
        if self.controller is not None:
            self.controller.close()
        with self._frame_lock:
            if self._frame_env is not None:
                self.simulator.close(self._frame_env)
                self._frame_env = None
                self._frame_task_id = None
        self.provider.unload()
