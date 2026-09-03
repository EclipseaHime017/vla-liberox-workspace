#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_rynn_iql.evaluation_store import bind_reward_manifest
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.server_config import load_server_config, validate_global_batch
from vla_rynn_iql.terminal_pipeline import (
    annotation_cache_valid,
    bound_evaluation_count,
    build_selection_manifest,
    dataset_roots,
    discover_candidates,
    mark_prepare_cache,
    merged_training_config,
    prepare_cache_valid,
    prepare_fingerprint,
    resolve_task_id,
    reward_cache_valid,
    select_candidates,
    validate_effective_config,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _counts(items: list[Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in items:
        key = f"{item.source_type}/{item.outcome}"
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def _verify_conda_environments(names: set[str]) -> None:
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("conda is not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Cannot list Conda environments: {exc.stderr.strip()}") from exc
    available = {Path(path).name for path in json.loads(result.stdout).get("envs", [])}
    missing = sorted(names - available)
    if missing:
        raise RuntimeError(f"Missing Conda environments: {missing}; available: {sorted(available)}")


class ServerStageRunner:
    def __init__(self, state_path: Path, state: dict[str, Any]):
        self.state_path = state_path
        self.state = state
        self.process: subprocess.Popen[Any] | None = None
        self.interrupted = False

    def install_signal_handlers(self) -> None:
        def stop(signum: int, _frame: Any) -> None:
            self.interrupted = True
            self.state.update(status="STOPPING", updated_at=_utc_now())
            atomic_json(self.state_path, self.state)
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(
                        self.process.pid,
                        signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM,
                    )
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

    def run(
        self,
        stage_id: str,
        label: str,
        argv: list[str],
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        stage = self.state["stages"][stage_id]
        stage.update(status="RUNNING", started_at=_utc_now(), argv=argv)
        self.state.update(status="RUNNING", current_stage=stage_id, updated_at=_utc_now())
        atomic_json(self.state_path, self.state)
        print(f"\n[{stage_id}] {label}", flush=True)
        started = time.monotonic()
        self.process = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            env=environment,
            start_new_session=True,
        )
        return_code = self.process.wait()
        self.process = None
        stage.update(
            status=(
                "COMPLETED" if return_code == 0
                else "INTERRUPTED" if self.interrupted
                else "FAILED"
            ),
            completed_at=_utc_now(),
            elapsed_seconds=time.monotonic() - started,
            return_code=return_code,
        )
        self.state["updated_at"] = _utc_now()
        atomic_json(self.state_path, self.state)
        if return_code:
            if self.interrupted:
                raise KeyboardInterrupt(f"{label} interrupted")
            raise RuntimeError(f"{label} failed with exit code {return_code}")

    def python_stage(
        self,
        stage_id: str,
        label: str,
        conda_environment: str,
        script: str,
        config_path: Path,
        extra: list[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.run(
            stage_id,
            label,
            [
                "conda", "run", "--no-capture-output", "-n", conda_environment,
                "python", str(ROOT / "scripts" / script), "--config", str(config_path),
                *(extra or []),
            ],
            environment=environment,
        )

    def skip(self, stage_id: str, reason: str) -> None:
        self.state["stages"][stage_id].update(
            status="SKIPPED", reason=reason, completed_at=_utc_now(), elapsed_seconds=0.0
        )
        self.state["updated_at"] = _utc_now()
        atomic_json(self.state_path, self.state)


def _confirm(assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("stdin is not a TTY; pass --yes for unattended execution")
    if input("Start this server pipeline? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise KeyboardInterrupt("Server pipeline canceled before execution")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated single-node DDP + ZeRO-1 server pipeline"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "server_pipeline.yaml")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print only")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-annotate", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    server = load_server_config(args.config)
    raw = merged_training_config(server.terminal)
    global_batch, local_batch = validate_global_batch(raw, server.distributed)
    roots = dataset_roots(raw, server.terminal.pipeline_root / "imports")
    candidates, rejected = discover_candidates(
        roots,
        raw["data"]["project_id"],
        int(raw["data"]["success_consecutive_steps"]),
    )
    canonical_task = resolve_task_id(server.terminal.selection["task_id"], candidates)
    raw["data"]["task_ids"] = [canonical_task]
    selected = select_candidates(candidates, server.terminal.selection, canonical_task)
    selection_manifest = build_selection_manifest(
        selected, raw, server.terminal.selection, canonical_task
    )
    fingerprint = prepare_fingerprint(selection_manifest, raw)
    work_dir = server.terminal.pipeline_root / "cache" / "prepared" / fingerprint / "work"
    selection_path = (
        server.terminal.pipeline_root / "datasets" / selection_manifest["id"] / "dataset.json"
    )
    raw["paths"]["work_dir"] = str(work_dir.resolve())
    raw["data"]["selection_manifest"] = str(selection_path.resolve())
    validate_effective_config(raw)
    prepare_skip = prepare_cache_valid(work_dir, fingerprint)
    annotation_skip = (
        prepare_skip
        and annotation_cache_valid(work_dir, raw["reward"])
        and not args.force_annotate
    )
    reward_skip = annotation_skip and reward_cache_valid(work_dir, raw["reward"])
    evaluated = bound_evaluation_count(selection_manifest)
    print("\n=== VLA-Adapter server DDP + ZeRO-1 pipeline ===")
    print(f"Task               : {canonical_task}")
    print(f"Eligible / selected: {len(candidates)} / {len(selected)} {_counts(selected)}")
    print(f"Rejected manifests : {len(rejected)}")
    print(f"Bound evaluations  : {evaluated}/{len(selected)}")
    print(f"GPU IDs            : {list(server.distributed.gpu_ids)}")
    print(f"World size         : {server.distributed.world_size}")
    print(f"Global/local batch : {global_batch}/{local_batch}")
    print(
        "Actor global batch  : "
        f"{global_batch * int(raw['iql']['gradient_accumulation_steps'])}"
    )
    print(f"ZeRO/backend       : stage {server.distributed.zero_stage} / {server.distributed.backend}")
    print(f"Replay cache       : {server.replay_cache.root}")
    print("Stages             : prepare → annotate → rewards → bind → mmap cache → DDP train")
    print(
        "Cache plan         : "
        f"annotation={'hit' if annotation_skip else 'run'} / "
        f"reward={'hit' if reward_skip else 'rebuild'}"
    )
    if args.dry_run:
        print("Dry run complete; no server pipeline output was created.")
        return 0
    _confirm(args.yes)
    _verify_conda_environments(set(server.terminal.environments.values()))

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    if selection_path.is_file():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection_manifest:
            raise RuntimeError(f"Frozen selection hash collision at {selection_path}")
    else:
        atomic_json(selection_path, selection_manifest)
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}__"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_dir = server.terminal.pipeline_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_path = run_dir / "effective_config.yaml"
    _atomic_yaml(effective_path, raw)
    state_path = run_dir / "pipeline.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "id": run_id,
        "status": "READY",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "current_stage": None,
        "task_id": canonical_task,
        "selection_id": selection_manifest["id"],
        "selected_count": len(selected),
        "prepare_fingerprint": fingerprint,
        "effective_config": str(effective_path),
        "server_config": str(server.path),
        "training_result": None,
        "error": None,
        "distributed": {
            "gpu_ids": list(server.distributed.gpu_ids),
            "world_size": server.distributed.world_size,
            "global_micro_batch_size": global_batch,
            "local_micro_batch_size": local_batch,
            "zero_stage": server.distributed.zero_stage,
        },
        "cache": {
            "prepare_hit": bool(prepare_skip and not args.force_prepare),
            "annotation_manifest_hit": bool(annotation_skip),
            "reward_manifest_hit": bool(reward_skip),
        },
        "stages": {
            name: {"status": "PENDING", "started_at": None, "completed_at": None}
            for name in ("prepare", "annotate", "rewards", "bind", "cache", "train")
        },
    }
    atomic_json(state_path, state)
    runner = ServerStageRunner(state_path, state)
    runner.install_signal_handlers()
    single_gpu_environment = os.environ.copy()
    single_gpu_environment["CUDA_VISIBLE_DEVICES"] = str(
        server.distributed.gpu_ids[0]
    )
    try:
        if prepare_skip and not args.force_prepare:
            runner.skip("prepare", "prepare fingerprint and manifest hash match")
        else:
            runner.python_stage(
                "prepare", "Prepare selected trajectories",
                server.terminal.environments["prepare"], "prepare_dataset.py", effective_path,
                environment=single_gpu_environment,
            )
            mark_prepare_cache(work_dir, fingerprint, selection_manifest["dataset_sha256"])

        if annotation_cache_valid(work_dir, raw["reward"]) and not args.force_annotate:
            runner.skip("annotate", "complete official-output annotation cache matches")
        else:
            runner.python_stage(
                "annotate", "RynnValue trajectory evaluation",
                server.terminal.environments["annotate"], "annotate_rewards.py", effective_path,
                ["--overwrite"] if args.force_annotate else None,
                environment=single_gpu_environment,
            )

        if reward_cache_valid(work_dir, raw["reward"]) and not args.force_annotate:
            runner.skip("rewards", "deterministic reward derivation cache matches")
        else:
            runner.python_stage(
                "rewards", "Derive IQL rewards from cached RynnValue outputs",
                server.terminal.environments["prepare"],
                "materialize_rewards.py", effective_path,
                ["--force"] if args.force_annotate else None,
                environment=single_gpu_environment,
            )

        bind_started = time.monotonic()
        state["stages"]["bind"].update(status="RUNNING", started_at=_utc_now())
        state.update(current_stage="bind", updated_at=_utc_now())
        atomic_json(state_path, state)
        binding = bind_reward_manifest(
            work_dir / "dataset_manifest.json",
            work_dir / "rewards" / "reward_manifest.json",
        )
        state["stages"]["bind"].update(
            status="COMPLETED",
            completed_at=_utc_now(),
            elapsed_seconds=time.monotonic() - bind_started,
            result=binding,
        )
        atomic_json(state_path, state)

        cache_result_path = run_dir / "cache_result.json"
        cache_extra = [
            "--server-config", str(server.path),
            "--result-file", str(cache_result_path),
        ]
        if args.force_cache or server.replay_cache.rebuild:
            cache_extra.append("--rebuild")
        runner.python_stage(
            "cache", "Build or validate mmap replay cache",
            server.terminal.environments["prepare"], "build_server_cache.py", effective_path,
            cache_extra,
            environment=single_gpu_environment,
        )
        cache_result = json.loads(cache_result_path.read_text(encoding="utf-8"))
        state["cache"]["replay_path"] = cache_result["path"]
        state["cache"]["replay_item_count"] = cache_result["item_count"]
        state["cache"]["replay_hit"] = cache_result["reused"]
        atomic_json(state_path, state)

        train_result = run_dir / "train_result.json"
        worker = ROOT / "scripts" / "train_iql_distributed.py"
        train_argv = [
            "conda", "run", "--no-capture-output", "-n",
            server.terminal.environments["train"],
            "torchrun", "--standalone",
            f"--nproc-per-node={server.distributed.world_size}",
            str(worker),
            "--config", str(effective_path),
            "--server-config", str(server.path),
            "--result-file", str(train_result),
        ]
        child_environment = os.environ.copy()
        child_environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(value) for value in server.distributed.gpu_ids
        )
        child_environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        runner.run(
            "train", "VLA-Adapter distributed IQL post-training", train_argv,
            environment=child_environment,
        )
        result = json.loads(train_result.read_text(encoding="utf-8"))
        state.update(
            status="COMPLETED",
            current_stage=None,
            completed_at=_utc_now(),
            updated_at=_utc_now(),
            training_result=result,
        )
        atomic_json(state_path, state)
        print(f"\nServer pipeline completed: {run_dir}")
        print(f"Policy overlay           : {result['policy_overlay']}")
        return 0
    except KeyboardInterrupt as exc:
        state.update(
            status="INTERRUPTED", current_stage=None, completed_at=_utc_now(),
            updated_at=_utc_now(), error=str(exc),
        )
        atomic_json(state_path, state)
        print(f"\nServer pipeline interrupted; state saved to {state_path}", file=sys.stderr)
        return 130
    except Exception as exc:
        active = state.get("current_stage")
        if active in state["stages"] and state["stages"][active]["status"] == "RUNNING":
            state["stages"][active].update(status="FAILED", completed_at=_utc_now())
        state.update(
            status="FAILED", current_stage=None, completed_at=_utc_now(),
            updated_at=_utc_now(), error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(state_path, state)
        print(f"\nServer pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
