#!/usr/bin/env python3
"""Private stdio worker owned by the UI's model provider (no listening socket)."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    output = sys.stdout

    def reply(value):
        output.write("PI05_RPC " + json.dumps(value) + "\n")
        output.flush()

    request = json.loads(sys.stdin.readline())
    owner = request["owner_pid"]

    def watch_owner():
        while True:
            time.sleep(2)
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                os._exit(1)

    threading.Thread(target=watch_owner, daemon=True).start()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from types import SimpleNamespace
            from vla_rynn_iql import pi05
            from vla_rynn_iql.io import sha256_file
            from vla_rynn_iql.pi05_assets import identity_digest
            settings = request["model"]
            config = SimpleNamespace(raw={"model": settings}, section=lambda key: {
                "training": {"device": "cuda:0", "dtype": "bfloat16"}, "model": settings}[key])
            components = pi05.load_components(config, training=False)
            if identity_digest(components.identity) != request["base_revision"]:
                raise ValueError("Selected π₀.₅ base identity changed before loading")
            if request.get("actor"):
                actor = Path(request["actor"])
                if sha256_file(actor) != request["actor_sha256"]:
                    raise ValueError("Selected π₀.₅ actor changed before loading")
                pi05.restore_actor(components, settings, actor.parent)
            from openpi.policies.policy import Policy
            policy = Policy(components.model, transforms=[components.input_transform],
                            output_transforms=[components.output_transform], is_pytorch=True,
                            pytorch_device="cuda:0", sample_kwargs={"num_steps": settings["num_inference_steps"]})
        reply({"ready": True, "identity": request["content_sha256"]})
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "seed":
                torch.manual_seed(request["seed"])
                torch.cuda.manual_seed_all(request["seed"])
                reply({"ok": True})
                continue
            with np.load(io.BytesIO(base64.b64decode(request["arrays"])), allow_pickle=False) as arrays:
                observation = {name: arrays[name] for name in arrays.files}
            observation["prompt"] = request["prompt"]
            with contextlib.redirect_stdout(sys.stderr):
                actions = policy.infer(observation)["actions"]
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError(f"π₀.₅ returned invalid actions: {actions.shape}")
            reply({"actions": actions.tolist()})
    except Exception as exc:
        reply({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
