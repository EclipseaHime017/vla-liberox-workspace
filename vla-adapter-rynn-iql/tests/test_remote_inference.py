from pathlib import Path

import numpy as np
import pytest

from vla_rynn_iql.remote_inference import Client, Sessions, pack_observation, unpack_observation, read_token


def obs():
    return {"agentview_image": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
            "robot0_eye_in_hand_image": np.full((256, 256, 3), 123, dtype=np.uint8),
            "robot0_eef_pos": np.array([.1, .2, .3]), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
            "robot0_gripper_qpos": np.array([.01, -.01])}


def test_observation_transport_is_lossless_and_does_not_flip():
    original = obs()
    decoded = unpack_observation(pack_observation(original))
    for name in original:
        np.testing.assert_array_equal(decoded[name], original[name])


def test_observation_rejects_nonfinite_state_and_wrong_images():
    source = obs()
    source["robot0_eef_pos"][0] = np.nan
    with pytest.raises(ValueError):
        pack_observation(source)
    source = obs()
    source["agentview_image"] = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pack_observation(source)


@pytest.mark.parametrize("endpoint", ["http://example.com:80", "https://127.0.0.1:80", "file:///tmp/x", "http://127.0.0.1/a"])
def test_client_requires_loopback_ssh_tunnel(endpoint):
    with pytest.raises(ValueError):
        Client(endpoint, "x" * 32)


def test_token_requires_private_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 64)
    path.chmod(0o644)
    with pytest.raises(ValueError):
        read_token(path)
    path.chmod(0o600)
    assert read_token(path) == "x" * 64


def test_interleaved_rollouts_preserve_random_streams(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state().clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    identity = {"instance_id": "one", "policy_sha256": "a" * 64}
    metadata = {**identity, "contract": {"io": {"native_action_horizon": 2}}}
    sessions = Sessions(lambda *_: torch.rand(2, 7).numpy(), lambda a: a, metadata)
    def new():
        return sessions.call("/session", {**identity, "seed": 5})["session_id"]
    def infer(key, sequence):
        return sessions.call("/infer", {**identity, "session_id": key, "sequence": sequence,
                                        "prompt": "task", "observation": pack_observation(obs())})["actions"]
    a, b = new(), new()
    a0, b0, a1, b1 = infer(a, 0), infer(b, 0), infer(a, 1), infer(b, 1)
    np.testing.assert_array_equal(a0, b0)
    np.testing.assert_array_equal(a1, b1)
    assert a0 != a1
    with pytest.raises(ValueError, match="Out-of-order"):
        infer(a, 1)
    with pytest.raises(ValueError, match="identity"):
        sessions.call("/session", {**identity, "instance_id": "replacement", "seed": 0})


def test_client_rejects_wrong_checkpoint_and_closes_connection(monkeypatch):
    import http.client
    from types import SimpleNamespace
    closed = []
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs:
                        SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(Client, "_request", lambda *_: {
        "protocol": 1, "actions": "environment_ready", "policy_sha256": "b" * 64})
    with pytest.raises(ValueError, match="requested model"):
        Client("http://127.0.0.1:18210", "x" * 64, expected_sha256="a" * 64)
    assert closed == [True]


def test_simulation_worker_does_not_reuse_environment_across_levels(tmp_path, monkeypatch):
    import json
    import os
    from types import SimpleNamespace
    from vla_rynn_iql import remote_evaluation as remote, evaluation_runtime as runtime
    from vla_rynn_iql.io import sha256_file
    from vla_rynn_iql.server_resources import process_identity

    created, closed = [], []
    class FakeEnv:
        language_instruction = "task"
        env = SimpleNamespace(sim=SimpleNamespace(get_state=lambda: np.zeros(2)))
        def seed(self, _): pass
        def reset(self): pass
        def close(self): closed.append(True)
    class FakeClient:
        metadata = {"contract": {"io": {"native_action_horizon": 2}}}
        metrics = []
        def __init__(self, *_args, **_kwargs): pass
        def reset(self, _): pass
        def close(self): pass
    def create(_factory, task, *_args):
        created.append(task["level"])
        return FakeEnv()
    monkeypatch.setattr(remote, "register_worker", lambda _: None)
    monkeypatch.setattr(remote, "read_token", lambda _: "x" * 64)
    monkeypatch.setattr(remote, "Client", FakeClient)
    monkeypatch.setattr(runtime, "initialize_simulation", lambda *_: (object(), {}))
    monkeypatch.setattr(runtime, "render_preflight", lambda *_: {})
    monkeypatch.setattr(runtime, "load_states", lambda _: [np.zeros(2)])
    monkeypatch.setattr(runtime, "create_env", create)
    monkeypatch.setattr(runtime, "episode", lambda *_args, **_kwargs: {"error": None, "steps": 2})
    asset = tmp_path / "asset"
    asset.write_text("fixture")
    task = {"task_name": "same_name", "max_steps": 2}
    for kind in ("bddl", "init"):
        task.update({f"{kind}_path": str(asset), f"{kind}_sha256": sha256_file(asset)})
    request = {"directory": str(tmp_path / "worker"), "launcher_identity": process_identity(os.getpid()),
               "lease_token": "test", "endpoint": "unused", "token_file": "unused", "policy_sha256": "a" * 64,
               "metadata": FakeClient.metadata, "config": {"render_backend": "egl", "success_streak": 5},
               "open_loop_steps": 2, "assignments": [
                   {"task": {**task, "level": level}, "schedule": {"seed": 0, "init_state_index": 0}}
                   for level in ("LEVEL1", "LEVEL2")]}
    result = remote.simulation_worker(request)
    assert result["status"] == "COMPLETED", result["error"]
    assert created == ["LEVEL1", "LEVEL2"]
    assert len(closed) == 2
    assert json.loads((tmp_path / "worker/result.json").read_text())["status"] == "COMPLETED"


def test_cancel_retires_lease_and_only_stops_owned_workers(monkeypatch):
    from types import SimpleNamespace
    from vla_rynn_iql import remote_evaluation as remote
    retired, signaled = [], []
    lease = SimpleNamespace(retire=lambda: retired.append(True), workers=lambda: [{"pid": 123, "start": "own"}])
    monkeypatch.setattr(remote, "signal_worker", lambda identity, sig: signaled.append((identity, sig)))
    record = {"canceled": False}
    previous = remote.signal.getsignal(remote.signal.SIGTERM)
    with remote.cancel_on_signal(lease, record):
        remote.signal.getsignal(remote.signal.SIGTERM)(remote.signal.SIGTERM, None)
        assert record["canceled"]
    assert retired == [True]
    assert signaled == [({"pid": 123, "start": "own"}, remote.signal.SIGTERM)]
    assert remote.signal.getsignal(remote.signal.SIGTERM) == previous
