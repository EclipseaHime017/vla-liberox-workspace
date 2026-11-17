"""Loopback-only inference over an authenticated SSH tunnel; no simulator on the server."""
from __future__ import annotations

import base64
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import socket
import threading
import time
import uuid
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .io import sha256_file

PROTOCOL = 1
MAX_BODY = 2 * 1024 * 1024
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_SHAPES = {"robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,)}


def read_token(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("Token file must be private (chmod 600)")
    value = path.read_text().strip()
    if len(value) < 32:
        raise ValueError("Use a random token with at least 32 characters")
    return value


def pack_observation(observation):
    result = {}
    for name in CAMERAS:
        frame = np.asarray(observation[name])
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError("Expected raw RGB uint8 256x256 camera frames")
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG", compress_level=1)
        result[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    for name, shape in STATE_SHAPES.items():
        value = np.asarray(observation[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation state: {name}")
        result[name] = value.tolist()
    return result


def unpack_observation(payload):
    if not isinstance(payload, dict) or set(payload) != set(CAMERAS) | set(STATE_SHAPES):
        raise ValueError("Invalid observation fields")
    result = {}
    for name in CAMERAS:
        data = base64.b64decode(payload[name], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (256, 256) or image.mode != "RGB":
                raise ValueError("Expected lossless 256x256 RGB PNG")
            result[name] = np.array(image)
    for name, shape in STATE_SHAPES.items():
        value = np.asarray(payload[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation state: {name}")
        result[name] = value
    return result


def load_policy(path, vla_root):
    import importlib
    import torch
    import yaml
    from .base_models import family_adapter, model_contract, execution_config
    from .config import LoadedConfig
    from .config_sources import UniqueKeyLoader
    from .model_artifacts import decode_child, validate_policy

    digest = sha256_file(path)
    payload = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    validate_policy(path, payload)
    native, parent = decode_child(payload)
    settings = native["model_config"]
    adapter = family_adapter(settings["family"])
    if adapter.resolve_revision(settings, hash_file=sha256_file, parent=parent) != parent["revision"]:
        raise ValueError("Actual base weights differ from the checkpoint's frozen parent")
    if not torch.cuda.is_available():
        raise RuntimeError("Remote inference requires CUDA")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = LoadedConfig(path, {"model": settings,
        "training": {"device": "cuda:0", "dtype": "bfloat16", "seed": 0},
        "paths": {"vla_adapter_root": str(vla_root)}})
    backend = importlib.import_module(adapter.BACKEND)
    predict, process_action = backend.evaluation_policy(config, path, parent)
    validate_policy(path, payload)
    if sha256_file(path) != digest or adapter.resolve_revision(settings, hash_file=sha256_file, parent=parent) != parent["revision"]:
        raise ValueError("Model changed while loading")
    metadata = {"protocol": PROTOCOL, "instance_id": uuid.uuid4().hex,
                "policy_sha256": digest, "policy": payload, "contract": model_contract(settings),
                "execution": execution_config(settings, trained=True), "gpu": torch.cuda.get_device_name(0),
                "transport": "lossless_png_raw_observation", "actions": "environment_ready"}
    return predict, process_action, metadata


class Sessions:
    """One GPU model, serialized calls, and independent RNG streams per rollout."""

    def __init__(self, predict, process_action, metadata):
        self.predict, self.process_action, self.metadata = predict, process_action, metadata
        self.lock = threading.Lock()
        self.sessions = {}
        self.last_activity = time.monotonic()

    @staticmethod
    def rng_state():
        import torch
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())

    @staticmethod
    def restore_rng(state):
        import torch
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.cuda.set_rng_state(state[3])

    def call(self, route, payload):
        import torch
        if payload.get("instance_id") != self.metadata["instance_id"] or payload.get("policy_sha256") != self.metadata["policy_sha256"]:
            raise ValueError("Inference instance/checkpoint identity changed")
        queued = time.monotonic()
        with self.lock:
            wait = time.monotonic() - queued
            self.last_activity = time.monotonic()
            expired = [key for key, item in self.sessions.items() if self.last_activity - item["last_activity"] > 600]
            for key in expired:
                del self.sessions[key]
            if route == "/session":
                seed = payload["seed"]
                if type(seed) is not int or not 0 <= seed <= 2147483647 or len(self.sessions) >= 64:
                    raise ValueError("Invalid seed or session capacity reached")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                key = uuid.uuid4().hex
                self.sessions[key] = {"rng": self.rng_state(), "sequence": 0, "last_activity": self.last_activity}
                return {"session_id": key}
            key = payload["session_id"]
            if route == "/close":
                self.sessions.pop(key, None)
                return {"closed": True}
            if route != "/infer":
                raise ValueError("Unknown operation")
            item = self.sessions[key]
            sequence = payload["sequence"]
            if type(sequence) is not int or sequence != item["sequence"]:
                raise ValueError("Out-of-order request; inference requests must not be retried")
            prompt = payload["prompt"]
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
                raise ValueError("Invalid task prompt")
            observation = unpack_observation(payload["observation"])
            self.restore_rng(item["rng"])
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    raw = np.asarray(self.predict(observation, prompt), dtype=np.float32)
                    horizon = self.metadata["contract"]["io"]["native_action_horizon"]
                    if raw.shape != (horizon, 7) or not np.isfinite(raw).all():
                        raise ValueError("Invalid policy output")
                    actions = np.asarray([self.process_action(action.copy()) for action in raw], dtype=np.float32)
                    if actions.shape != raw.shape or not np.isfinite(actions).all():
                        raise ValueError("Invalid environment action")
                torch.cuda.synchronize()
                item.update(rng=self.rng_state(), sequence=sequence + 1, last_activity=time.monotonic())
                return {"actions": actions.tolist(), "sequence": sequence,
                        "inference_ms": 1000 * (time.monotonic() - started), "queue_ms": 1000 * wait,
                        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                        "reserved_mib": torch.cuda.memory_reserved() / 2**20}
            except BaseException:
                self.sessions.pop(key, None)
                raise


def serve(policy, vla_root, token, port, *, idle_seconds=900):
    predict, action, metadata = load_policy(policy, vla_root)
    sessions = Sessions(predict, action, metadata)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            self.connection.settimeout(180)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            authorized = hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)
            if not authorized:
                self.send_error(401)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/meta":
                    result = metadata
                else:
                    result = sessions.call(self.path, payload)
                response = {"instance_id": metadata["instance_id"], "policy_sha256": metadata["policy_sha256"], **result}
                status = 200
            except Exception as exc:
                response, status = {"error": f"{type(exc).__name__}: {exc}"}, 400
            data = json.dumps(response, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    with Server(("127.0.0.1", port), Handler) as server:
        server.timeout = 1
        print(json.dumps({"status": "READY", "port": server.server_port, **metadata}), flush=True)
        while time.monotonic() - sessions.last_activity < idle_seconds:
            server.handle_request()


class Client:
    def __init__(self, url, token, *, expected_sha256=None):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path not in ("", "/"):
            raise ValueError("Use an HTTP loopback URL forwarded through SSH")
        self.url, self.token = url.rstrip("/"), token
        self.connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=180)
        try:
            self.metadata = self._request("/meta", {})
            if self.metadata.get("protocol") != PROTOCOL or self.metadata.get("actions") != "environment_ready":
                raise ValueError("Unsupported inference protocol")
            if expected_sha256 is not None and self.metadata["policy_sha256"] != expected_sha256:
                raise ValueError("Remote checkpoint differs from requested model")
        except BaseException:
            self.connection.close()
            raise
        self.session_id = None
        self.sequence = 0
        self.metrics = []

    def _request(self, route, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        self.connection.request("POST", route, body=data, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        with self.connection.getresponse() as response:
            body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise ValueError("Remote response exceeded its size limit")
            value = json.loads(body)
            if response.status != 200:
                raise RuntimeError(value.get("error", f"Remote HTTP {response.status}"))
        if route != "/meta" and any(value.get(key) != self.metadata[key] for key in ("instance_id", "policy_sha256")):
            raise ValueError("Remote inference identity changed")
        return value

    def _payload(self):
        return {key: self.metadata[key] for key in ("instance_id", "policy_sha256")} | {"session_id": self.session_id}

    def reset(self, seed):
        if self.session_id is not None:
            self.close()
        self.session_id = self._request("/session", {**self._payload(), "seed": seed})["session_id"]
        self.sequence = 0

    def predict(self, observation, prompt):
        if self.session_id is None:
            raise ValueError("Create a rollout session before inference")
        started = time.monotonic()
        packed = pack_observation(observation)
        payload = {**self._payload(), "sequence": self.sequence, "prompt": prompt, "observation": packed}
        result = self._request("/infer", payload)
        if result["sequence"] != self.sequence:
            raise ValueError("Wrong inference response sequence")
        self.sequence += 1
        actions = np.asarray(result["actions"], dtype=np.float32)
        horizon = self.metadata["contract"]["io"]["native_action_horizon"]
        if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid remote action chunk")
        self.metrics.append({key: result[key] for key in ("inference_ms", "queue_ms", "allocated_mib", "reserved_mib")} |
                            {"roundtrip_ms": 1000 * (time.monotonic() - started),
                             "request_bytes": len(json.dumps(payload).encode())})
        return actions

    def close(self):
        try:
            if self.session_id is not None:
                self._request("/close", self._payload())
                self.session_id = None
        finally:
            self.connection.close()
