"""Headless server planning/launch; shared by the full-screen UI and batch CLI."""
from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_train_config, validate_train_config
from .io import atomic_json
from .methods import training_method
from .server_config import validate_global_batch
from .server_inputs import GlobalTask, prepare_global_inputs, saved_reward


def visible_devices(gpu_ids, environ=None) -> str:
    """Select only within a scheduler's allocation, if one was supplied."""
    environ = os.environ if environ is None else environ
    inherited = environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return ",".join(map(str, gpu_ids))
    allocation = [item.strip() for item in inherited.split(",")]
    if any(not item or item == "-1" for item in allocation) or max(gpu_ids) >= len(allocation):
        raise ValueError("gpu_ids must index the existing CUDA_VISIBLE_DEVICES allocation; "
                         f"requested {list(gpu_ids)}, allocation={inherited!r}")
    return ",".join(allocation[index] for index in gpu_ids)


def training_settings(server, task: GlobalTask, *, overrides=None, source=None) -> dict:
    changes = copy.deepcopy(server.overrides if overrides is None else overrides)
    raw = load_train_config(server.training_config, overrides=changes, overrides_path=server.path).raw
    raw["data"].update(task_ids=[task.task_id], selection_manifest=None, stage_annotations_manifest=None)
    raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None)
    if training_method(raw).requires_rewards:
        name = source or server.reward_source
        stage_exponent = raw["reward"]["stage_exponent"]
        records = [saved_reward(run, name)[0] for run in task.selected(name)
                   if run.rewards[name].name != "stage_annotation.json"] if name != "sparse" else []
        # Each entry retains its own semantic recipe. The first is only the
        # default discount/reduction; explicit training overrides take priority.
        if records:
            saved = records[0]
            raw["reward"].update(saved.get("annotation_config", {}))
            raw["reward"].update(saved["reward_config"])
            raw["reward"]["final_normalization"] = saved["reward_config"].get("final_normalization", "none")
        raw["reward"].update(changes.get("reward", {}))
        if name == "stage":
            # Bare labels use the configured global default, never a neighbour's
            # saved Stage recipe. Explicit snapshots keep their per-run p.
            raw["reward"]["stage_exponent"] = stage_exponent
        raw["reward"].update(source=name, rynnvalue=name == "rynnvalue" or (
            name == "final" and raw["reward"]["shaping_weight"] > 0))
        macro_only = any(record["reward_config"].get("fusion_mode") == "multiplicative" for record in records)
        if name == "final" and macro_only:
            if changes.get("reward", {}).get("accumulate_primitive_steps") is True:
                raise ValueError("Global Final Reward includes multiplication; cumulative reward must be Off")
            raw["reward"]["accumulate_primitive_steps"] = False
    return validate_train_config(raw, server.path).raw


def execution_plan(server, task, raw, distributed, source):
    if any(path.resolve().is_relative_to(server.runs_root.resolve()) for path in
           (server.output_root, server.cache_root)):
        raise ValueError("Server output and cache must be outside the copied runs root")
    selected = task.selected(source if training_method(raw).requires_rewards else None)
    if not selected:
        raise ValueError(f"No eligible marked trajectories for {raw['training']['method']}/{source}: {task.task_id}")
    global_batch, local = validate_global_batch(raw, distributed)
    return {"runs_root": str(server.runs_root), "task": task.task_id, **task.summary(),
        "members": len(selected), "selected_run_ids": [run.run_id for run in selected],
        "skipped_run_ids": [run.run_id for run in task.runs if run not in selected],
        "training": raw["training"], "model": raw["model"],
        "reward_source": source if training_method(raw).requires_rewards else None,
        "reward": raw["reward"] if training_method(raw).requires_rewards else None,
        "include_post_success": raw["data"]["include_post_success"],
        "distributed": asdict(distributed), "global_micro_batch_size": global_batch,
        "cuda_visible_devices": visible_devices(distributed.gpu_ids),
        "per_gpu_micro_batch_size": local,
        "actor_effective_batch_size": global_batch * raw["training"]["gradient_accumulation_steps"],
        "stages": ["verify copied runs / prepare", "reuse global rewards (no model forward)", "mmap cache", "DDP + ZeRO-1 training"]}


class ServerRun:
    """A subprocess lifecycle with durable status and cooperative safe cancellation."""
    def __init__(self, server, task, raw, distributed, source):
        self.server, self.task, self.raw = server, task, raw
        self.distributed, self.source = distributed, source
        self.process = None
        self.log = None
        self.directory = None
        self.cancel_at = None
        self.terminated_at = None
        self.state = execution_plan(server, task, raw, distributed, source)

    def start(self):
        identifier = f"server_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        self.directory = self.server.output_root / identifier
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state.update(status="PREPARING", run_id=identifier, created_at=datetime.now(timezone.utc).isoformat())
        self._save()
        try:
            raw = copy.deepcopy(self.raw)
            raw["paths"]["output_dir"] = str(self.directory)
            raw = prepare_global_inputs(self.task, self.directory / "inputs", raw, self.source)
            raw["training"]["device"] = "cuda:0"
            config_path = self.directory / "effective_config.yaml"
            config = validate_train_config(raw, config_path)
            config_path.write_text(yaml.safe_dump(config.raw, sort_keys=False))
            execution = self.directory / "execution.json"
            atomic_json(execution, {"distributed": asdict(self.distributed),
                "cache_root": str(self.server.cache_root), "run_dir": str(self.directory)})
            worker = Path(__file__).resolve().parents[2] / "scripts/train_iql_distributed.py"
            argv = ["conda", "run", "--no-capture-output", "-n", self.server.environment,
                    "python", "-u", "-m", "torch.distributed.run", "--standalone",
                    f"--nproc-per-node={self.distributed.world_size}", str(worker),
                    "--config", str(config_path), "--execution", str(execution)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible_devices(self.distributed.gpu_ids, env)
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            self.log = (self.directory / "training.log").open("w")
            self.process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            self.state.update(status="RUNNING", pid=self.process.pid, argv=argv)
            self._save()
        except BaseException as exc:
            self.state.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                              error=f"{type(exc).__name__}: {exc}")
            self._save()
            self.close()
            raise

    def _save(self):
        atomic_json(self.directory / "pipeline.json", self.state)

    def cancel(self):
        if self.cancel_at is None and self.process is not None and self.process.poll() is None:
            self.cancel_at = time.monotonic()
            # Shared file reaches every rank without killing torchrun's supervisor
            # before workers have consolidated a safe ZeRO checkpoint.
            (self.directory / "cancel.request").touch()
            self.state["status"] = "STOPPING"
            self._save()

    def poll(self):
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None and self.cancel_at is not None:
            now = time.monotonic()
            command = None
            if self.terminated_at is None and now - self.cancel_at > 120:
                command, self.terminated_at = signal.SIGTERM, now
                self.state["error"] = "Safe stop timed out; process group terminated; use last complete checkpoint"
                self._save()
            elif self.terminated_at is not None and now - self.terminated_at > 10:
                command = signal.SIGKILL
            if command is not None:
                try:
                    os.killpg(self.process.pid, command)
                except ProcessLookupError:
                    pass
        if code is not None and self.state.get("exit_code") is None:
            summary = self.directory / "summary.json"
            result = json.loads(summary.read_text()) if summary.is_file() else {}
            self.state.update(status=result.get("status", "INTERRUPTED" if self.cancel_at else
                                               "COMPLETED" if code == 0 else "FAILED"),
                              exit_code=code, result=result)
            if not result and code == 0:
                self.state.update(status="FAILED", error="Worker exited without a completed summary")
            self._save()
            self.close()
        return code

    def progress(self):
        if self.directory is None:
            return {}
        path = self.directory / "progress.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def tail(self, limit=8):
        if self.directory is None:
            return []
        path = self.directory / "training.log"
        if not path.is_file():
            return []
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-limit:]

    def close(self):
        if self.log is not None:
            self.log.close()
            self.log = None
