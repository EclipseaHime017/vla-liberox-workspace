"""Small, versioned human keyframes; no observation/video decoding or model calls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json


SIDECAR_NAME = "stage_annotation.json"
ACTIVE = {"LOADING", "READY", "RUNNING", "STOPPING", "POSTPROCESSING"}


class StageAnnotationService:
    def __init__(self, run_service: Any, offline_root: Path):
        self.run_service = run_service
        self.offline_root = Path(offline_root)
        self._lock = threading.RLock()
        self._sources: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
        self._math = None

    def _module(self):
        if self._math is None:
            path = self.offline_root / "src/vla_rynn_iql/stage_rewards.py"
            spec = importlib.util.spec_from_file_location("_human_stage_rewards", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Stage reward definitions are unavailable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._math = module
        return self._math

    def _defaults(self) -> tuple[int, float]:
        raw = yaml.safe_load((self.offline_root / "configs/liberox_iql.yaml").read_text())
        return int(raw["data"]["success_consecutive_steps"]), float(raw["reward"].get("stage_exponent", 2))

    @staticmethod
    def _fingerprint(path: Path) -> tuple:
        value = path.stat()
        return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    @staticmethod
    def _paths(run: dict) -> tuple[Path, Path]:
        trajectory = Path(str(run.get("trajectory") or ""))
        if trajectory.is_symlink() or trajectory.parent.is_symlink():
            raise ValueError("Stage annotation does not accept symlink trajectories")
        if trajectory.name != "trajectory.npz" or not trajectory.is_file():
            raise FileNotFoundError(f"Trajectory unavailable: {run.get('id')}")
        sidecar = trajectory.parent / SIDECAR_NAME
        if sidecar.is_symlink():
            raise ValueError("Stage annotation sidecar must not be a symlink")
        return trajectory, sidecar

    def _source(self, path: Path) -> dict:
        """Hash/load small control data once per file version; never open observations."""
        signature = self._fingerprint(path)
        key = str(path.resolve())
        cached = self._sources.get(key)
        if cached and cached[0] == signature:
            self._sources.move_to_end(key)
            return cached[1]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        with np.load(path, allow_pickle=False) as arrays:
            done = arrays["done"]
            times = arrays["time_seconds"]
            count = len(arrays["env_action"])
        if (count < 1 or done.shape != (count,) or done.dtype.kind != "b"
                or times.shape != (count + 1,) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Stage annotation requires N boolean done flags and N+1 ordered observation times")
        if signature != self._fingerprint(path):
            raise RuntimeError("Trajectory changed during read; reload the editor")
        source = dict(trajectory_sha256=digest.hexdigest(), action_count=count,
                      done=done.tolist(), time_seconds=times.astype(float).tolist())
        self._sources[key] = (signature, source)
        self._sources.move_to_end(key)
        while len(self._sources) > 32:
            self._sources.popitem(last=False)
        return source

    @staticmethod
    def _read_sidecar(path: Path) -> tuple[dict | None, bytes]:
        if not path.exists():
            return None, b""
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Stage annotation exceeds the metadata size limit")
        data = path.read_bytes()
        try:
            return json.loads(data), data
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, data

    @staticmethod
    def _revision(source: dict, data: bytes, threshold: int) -> str:
        return hashlib.sha256(source["trajectory_sha256"].encode() + data + str(threshold).encode()).hexdigest()

    def _view(self, run: dict, threshold: int, exponent: float) -> dict:
        trajectory, sidecar = self._paths(run)
        source = self._source(trajectory)
        payload, data = self._read_sidecar(sidecar)
        math = self._module()
        response = {"run_id": run["id"], "status": "missing", "error": None,
                    "action_count": source["action_count"], "time_seconds": source["time_seconds"],
                    "success_step": math.confirmed_success_step(source["done"], threshold),
                    "success_consecutive_steps": threshold, "exponent": exponent,
                    "keyframes": [], "scores": [], "anchors": [],
                    "revision": self._revision(source, data, threshold)}
        if not data:
            return response
        try:
            valid = math.validate_stage_annotation(
                payload, run_id=run["id"], trajectory_sha256=source["trajectory_sha256"],
                done=source["done"], success_consecutive_steps=threshold,
            )
            response.update(status="ready", keyframes=valid["keyframes"],
                            annotation_sha256=valid["annotation_sha256"])
            # Labels remain valid when a reward recipe cannot be evaluated.
            # The preview is disposable; only manual marks are persisted.
            try:
                context = math.stage_annotation_context(
                    valid, done=source["done"], success_consecutive_steps=threshold,
                    exponent=exponent,
                )
                response.update(scores=math.stage_scores(context).tolist(),
                                anchors=math.stage_anchors(context), derivation_error=None)
            except (ValueError, TypeError, KeyError) as exc:
                response.update(derivation_error=str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            response.update(status="stale", error=str(exc))
            # Preserve valid-looking user marks for explicit review/resave, never train from them.
            if isinstance(payload, dict) and isinstance(payload.get("keyframes"), list):
                response["keyframes"] = [frame for frame in payload["keyframes"] if isinstance(frame, dict)
                    and type(frame.get("step")) is int and frame.get("kind") in {"positive", "negative"}]
        return response

    def detail(self, run_id: str) -> dict:
        with self._lock:
            return self._view(self.run_service.get_run(run_id), *self._defaults())

    def save(self, run_id: str, keyframes: list[dict], exponent: float, revision: str | None) -> dict:
        with self._lock:
            run = self.run_service.get_run(run_id)
            if run.get("status") in ACTIVE:
                raise RuntimeError("Cannot mark keyframes while this simulation is active")
            threshold, _ = self._defaults()
            trajectory, sidecar = self._paths(run)
            source = self._source(trajectory)
            _, data = self._read_sidecar(sidecar)
            current = self._revision(source, data, threshold)
            if revision != current:
                raise ConflictError("轨迹或标注已改变，请重新加载后保存", code="STAGE_REVISION_CONFLICT")
            payload = self._module().build_stage_annotation(
                run_id=run_id, trajectory_sha256=source["trajectory_sha256"], done=source["done"],
                keyframes=keyframes, success_consecutive_steps=threshold, exponent=exponent,
            )
            if self._fingerprint(trajectory) != self._sources[str(trajectory.resolve())][0]:
                raise RuntimeError("Trajectory changed during save; reload the editor")
            atomic_write_json(sidecar, payload)
            return self._view(run, threshold, exponent)

    def validate_members(self, members: list[dict], success_consecutive_steps: int) -> dict[str, dict]:
        """Preflight all selected members, even ones whose prefixes replay deduplicates."""
        result, errors = {}, []
        with self._lock:
            for member in members:
                run_id = member["run_id"]
                try:
                    trajectory, sidecar = self._paths(self.run_service.get_run(run_id))
                    source = self._source(trajectory)
                    expected = member.get("artifacts", {}).get("trajectory", {}).get("sha256")
                    if expected and expected != source["trajectory_sha256"]:
                        raise ValueError("源轨迹与冻结数据集不匹配")
                    payload, _ = self._read_sidecar(sidecar)
                    if payload is None:
                        raise ValueError("缺少已保存的 Stage 标注")
                    result[run_id] = self._module().validate_stage_annotation(
                        payload, run_id=run_id, trajectory_sha256=source["trajectory_sha256"],
                        done=source["done"], success_consecutive_steps=success_consecutive_steps,
                    )
                except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                    errors.append(f"{run_id}: {exc}")
        if errors:
            raise ValueError("Stage-based 训练已停止；缺失或无效的标注：\n" + "\n".join(errors))
        return result
