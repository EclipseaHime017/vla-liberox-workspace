"""Opt-in end-to-end torchrun smoke tests; all experiment products are temporary."""
import json
import math
import os
from pathlib import Path
import shutil
import time

import pytest
import torch
import yaml

from vla_rynn_iql.config import PROJECT_ROOT
from vla_rynn_iql.model_artifacts import decode_child
from vla_rynn_iql.server_config import load_server_config
from vla_rynn_iql.server_inputs import discover_tasks
from vla_rynn_iql.server_pipeline import ServerRun, training_settings
from test_server_inputs import portable_export


@pytest.mark.skipif(os.environ.get("RUN_SERVER_GPU_SMOKE") != "1", reason="opt-in model GPU training")
@pytest.mark.parametrize("family,base", [("pi05", "pi05-libero-base"), ("vla_adapter", "base")])
@pytest.mark.parametrize("method", ["bc", "iql"])
def test_real_model_server_training_and_export(configured, tmp_path, monkeypatch, family, base, method):
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    source, config, _ = portable_export(configured, tmp_path)
    config.raw["paths"]["vla_adapter_root"] = str(PROJECT_ROOT.parent / "third_party/VLA-Adapter")
    config.path.write_text(yaml.safe_dump(config.raw))
    settings = yaml.safe_load((PROJECT_ROOT / "configs/server_pipeline.yaml").read_text())
    settings.update(training_config=str(config.path), runs_root=str(source),
                    output_root=str(tmp_path / "training"), cache_root=str(tmp_path / "cache"),
                    task_id=None, reward_source="sparse")
    settings["distributed"].update(gpu_ids=[0], data_workers_per_rank=0, persistent_workers=False,
                                    timeout_seconds=60)
    settings["overrides"] = {
        "model": {"family": family, "base_id": base, "backbone": "frozen"},
        "training": {"method": method, "train_steps": 2, "micro_batch_size": 1,
                     "gradient_accumulation_steps": 1, "checkpoint_interval": 2,
                     "actor_lr_warmup_steps": 0},
        "logging": {"wandb": {"enabled": False}, "tensorboard": False, "console_interval_steps": 1},
    }
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(settings))
    server = load_server_config(path)
    task, = discover_tasks(server.runs_root, config.raw["data"]["project_id"])
    run = ServerRun(server, task, training_settings(server, task), server.distributed, "sparse")
    try:
        run.start()
        deadline = time.monotonic() + 240
        while run.poll() is None and time.monotonic() < deadline:
            time.sleep(.2)
        errors = "\n".join(line for line in (run.directory / "training.log").read_text().splitlines()
                           if "NCCL INFO" not in line)
        assert run.poll() == 0, errors
        assert run.state["status"] == "COMPLETED", run.state
        metrics = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]
        assert len(metrics) == 2
        assert all(math.isfinite(row["actor_loss"]) and row["actor_grad_norm"] > 0 for row in metrics)
        assert all(row["algorithm"] == method and row["world_size"] == 1 for row in metrics)
        summary = json.loads((run.directory / "summary.json").read_text())
        policy_path = Path(summary["policy_overlay"])
        payload = yaml.safe_load(policy_path.read_text())
        native, parent = decode_child(payload)
        assert parent["family"] == family and payload["algorithm"] == method
        assert native["model_config"]["base_id"] == base
        if family == "pi05":
            from safetensors import safe_open
            state = torch.load(policy_path.parent / native["actor"], map_location="cpu", weights_only=True)
            assert not any(name.startswith("paligemma_with_expert.paligemma.") for name in state)
            expert = [name for name in state if ".gemma_expert." in name and name.endswith("weight")]
            with safe_open(Path(parent["checkpoint"]) / "model.safetensors", framework="pt", device="cpu") as initial:
                changed = [name for name in expert[:20]
                           if not torch.equal(state[name], initial.get_tensor(name).to(state[name].dtype))]
            assert changed, "No sampled action-expert weights changed"
            assert all("actor_flow_mse" in row for row in metrics)
        else:
            assert "action_head" in native and "actor_l1_dx" in metrics[-1]
        print(f"GPU {family}/{method}: loss={metrics[-1]['actor_loss']:.6f}, "
              f"peak={metrics[-1]['cuda_peak_memory_gib']:.2f} GiB, schema={payload['schema_version']}")
    finally:
        if run.process is not None and run.process.poll() is None:
            run.cancel()
            while run.poll() is None:
                time.sleep(.2)
        run.close()
        shutil.rmtree(tmp_path)
