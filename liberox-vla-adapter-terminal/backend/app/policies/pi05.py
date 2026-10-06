"""Managed π₀.₅ subprocess; no OpenPI/Transformers imports in the simulator."""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import signal
import subprocess
import threading

import numpy as np

from .pi05_catalog import PROJECT


class Pi05PolicyProvider:
    def __init__(self, runtime, eval_config, catalog):
        self.runtime, self.eval_config, self.catalog = runtime, eval_config, catalog
        self.process = None
        self.current_policy_entry = None
        self.current_policy_id = None
        self._lock = threading.RLock()

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None and self.current_policy_entry is not None

    def _receive(self, timeout):
        try:
            value = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.unload()
            raise TimeoutError("π₀.₅ worker timed out; check its environment, checkpoint and device memory") from exc
        if value is None or "error" in value:
            self.unload()
            raise RuntimeError(value.get("error") if value else "π₀.₅ worker exited; inspect the task log")
        return value

    def _send(self, value, timeout=120):
        try:
            self.process.stdin.write(json.dumps(value) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.unload()
            raise RuntimeError("π₀.₅ inference worker disconnected") from exc
        return self._receive(timeout)

    def load(self, open_loop_steps, policy_id="pi05-libero-base", *, expected_content_sha256=None):
        with self._lock:
            entry = self.catalog.select(policy_id)
            if entry.family != "pi05" or (expected_content_sha256 is not None and entry.content_sha256 != expected_content_sha256):
                raise ValueError("Selected π₀.₅ identity changed; select the model again")
            if self.loaded and self.current_policy_entry.content_sha256 == entry.content_sha256:
                self.current_policy_entry, self.current_policy_id = entry, policy_id
                return
            self.unload()
            self.responses = queue.Queue()
            self.process = subprocess.Popen([
                "conda", "run", "--no-capture-output", "-n", entry.model_config["environment"],
                "python", "-u", str(PROJECT / "scripts/pi05_worker.py"),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
            process, responses = self.process, self.responses

            def receive():
                try:
                    for line in process.stdout:
                        if line.startswith("PI05_RPC "):
                            responses.put(json.loads(line[len("PI05_RPC "):]))
                finally:
                    responses.put(None)

            threading.Thread(target=receive, daemon=True).start()
            try:
                ready = self._send({"model": entry.model_config, "owner_pid": os.getpid(),
                    "base_revision": entry.base_revision, "content_sha256": entry.content_sha256,
                    "actor": str(entry.actor) if entry.actor else None,
                    "actor_sha256": entry.component_sha256.get("actor")}, timeout=600)
                if ready.get("identity") != entry.content_sha256:
                    raise ValueError("π₀.₅ worker loaded an unexpected model")
                if self.catalog.select(policy_id).content_sha256 != entry.content_sha256:
                    raise ValueError("π₀.₅ artifacts changed during loading")
                self.current_policy_entry, self.current_policy_id = entry, policy_id
            except BaseException:
                self.unload()
                raise

    def seed(self, seed):
        with self._lock:
            if not self.loaded:
                raise RuntimeError("π₀.₅ is not loaded")
            self._send({"op": "seed", "seed": int(seed)})

    def predict(self, observation, prompt, disabled_policy_cameras=()):
        with self._lock:
            if not self.loaded:
                raise RuntimeError("π₀.₅ is not loaded")
            # Same Panda state and camera orientation as the official LIBERO example.
            from trajectory_utils import quaternion_to_axis_angle
            images = {}
            for key, name, toggle in (("agentview_image", "observation/image", "agentview"),
                                      ("robot0_eye_in_hand_image", "observation/wrist_image", "robot0_eye_in_hand")):
                image = np.asarray(observation[key])[::-1, ::-1].copy()
                images[name] = np.zeros_like(image) if toggle in disabled_policy_cameras else image
            images["observation/state"] = np.concatenate([
                observation["robot0_eef_pos"], quaternion_to_axis_angle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            ]).astype(np.float32)
            buffer = io.BytesIO()
            np.savez(buffer, **images)
            result = self._send({"op": "infer", "arrays": base64.b64encode(buffer.getvalue()).decode(), "prompt": prompt})
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                self.unload()
                raise ValueError("Invalid π₀.₅ action chunk")
            return actions[:8]

    def process_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Invalid π₀.₅ environment action")
        return np.clip(action, -1.0, 1.0)

    def unload(self):
        with self._lock:
            process, self.process = self.process, None
            self.current_policy_entry = self.current_policy_id = None
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    process.wait()
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                finally:
                    process.stdin.close()
                    process.stdout.close()

    def metadata(self):
        entry = self.current_policy_entry
        return {"provider": "pi05", "loaded": self.loaded, "gpu": "cuda:0 (π₀.₅ worker)",
                "model_device": "cuda:0" if self.loaded else None,
                "checkpoint": entry.base_checkpoint if entry else None,
                "policy_id": self.current_policy_id, "policy_label": entry.label if entry else "π₀.₅ · LIBERO",
                "model_switching": True, "action_codec": "libero_env_v1",
                "action_schema": {"size": 7, "range": [-1, 1], "units": "normalized OSC_POSE command",
                                  "predicted_chunk_size": 8, "native_action_horizon": 10}}
