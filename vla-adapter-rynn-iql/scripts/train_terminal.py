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
from vla_rynn_iql.config import LoadedConfig, reward_source
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.rewards import load_stage_annotations
from vla_rynn_iql.terminal_pipeline import (
    annotation_cache_valid,
    bound_evaluation_count,
    build_selection_manifest,
    dataset_roots,
    discover_candidates,
    load_terminal_config,
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


def _category_counts(items: list[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = f"{item.source_type}/{item.outcome}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _print_preflight(
    *,
    task_id: str,
    candidates: list[Any],
    selected: list[Any],
    rejected_count: int,
    selection_manifest: dict[str, Any],
    raw: dict[str, Any],
    prepare_skip: bool,
    annotation_skip: bool,
    reward_skip: bool,
    evaluated_count: int,
    force_prepare: bool,
    force_annotate: bool,
) -> None:
    data, reward, iql = raw["data"], raw["reward"], raw["iql"]
    wandb = raw["logging"]["wandb"]
    task_candidates = [item for item in candidates if item.task_id == task_id]
    members = selection_manifest["members"]
    roots = {member["root_run_id"] for member in members}
    validation_roots = {
        member["root_run_id"] for member in members if member["split"] == "validation"
    }
    print("\n=== VLA-Adapter RynnValue + IQL terminal pipeline ===")
    print(f"Task              : {task_id}")
    print(f"Eligible          : {len(task_candidates)} ({_category_counts(task_candidates)})")
    print(f"Selected          : {len(selected)} ({_category_counts(selected)})")
    print(f"Rejected manifests: {rejected_count}")
    print(f"Actions / chunks  : {sum(item.action_count for item in selected)} / "
          f"{sum(member['chunk_count'] for member in members)}")
    print(f"Root split        : {len(roots) - len(validation_roots)} train / "
          f"{len(validation_roots)} validation")
    print(f"Bound evaluations : {evaluated_count}/{len(selected)}")
    source = reward_source(reward)
    print(f"Reward source     : {source}")
    if source == "rynnvalue":
        print(f"RynnValue         : {reward['model']} @ {reward['revision']}")
    elif source == "stage":
        print(f"Stage exponent    : {reward['stage_exponent']}")
    print(
        f"IQL               : steps={iql['train_steps']}, "
        f"warmup={iql['critic_warmup_steps']}, beta={iql['beta']}, "
        f"max_weight={iql['max_advantage_weight']}, "
        f"micro_batch={iql['micro_batch_size']}, "
        f"actor_batch={iql['micro_batch_size'] * iql['gradient_accumulation_steps']}, "
        f"sample_budget={iql['train_steps'] * iql['micro_batch_size']}"
    )
    print(
        "W&B               : "
        + (
            f"{wandb['mode']} · {wandb['project']}"
            + (f" · group={wandb['group']}" if wandb["group"] else "")
            if wandb["enabled"] else "disabled"
        )
    )
    print(f"Success threshold : {data['success_consecutive_steps']} consecutive steps")
    print("Stages            :")
    print(f"  [1/5] prepare   : {'RUN' if force_prepare or not prepare_skip else 'SKIP (hash match)'}")
    print(f"  [2/5] annotate  : " + ("SKIP (independent reward source)" if source != "rynnvalue" else
          "RUN (forced)" if force_annotate else "SKIP (official-output cache)" if annotation_skip
          else "RUN missing/incompatible VLM output"))
    print(f"  [3/5] rewards   : {'SKIP (derivation cache)' if reward_skip else 'RUN fast deterministic reduction'}")
    print("  [4/5] bind      : " + ("RUN (atomic sidecars)" if source == "rynnvalue" else "SKIP"))
    print("  [5/5] train     : RUN (new output; resume only when configured)")
    print()


def _confirm(assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("stdin is not a TTY; pass --yes for unattended execution")
    answer = input("Start this pipeline? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise KeyboardInterrupt("Pipeline canceled before execution")


def _verify_conda_environments(names: set[str]) -> None:
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("conda is not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Cannot list Conda environments: {exc.stderr.strip()}") from exc
    payload = json.loads(result.stdout)
    available = {Path(path).name for path in payload.get("envs", [])}
    missing = sorted(names - available)
    if missing:
        raise RuntimeError(f"Missing Conda environments: {missing}; available: {sorted(available)}")


class StageRunner:
    def __init__(self, state_path: Path, state: dict[str, Any]):
        self.state_path = state_path
        self.state = state
        self.process: subprocess.Popen[Any] | None = None
        self.interrupted = False

    def install_signal_handlers(self) -> None:
        def stop(signum: int, _frame: Any) -> None:
            self.interrupted = True
            self.state["status"] = "STOPPING"
            self.state["updated_at"] = _utc_now()
            atomic_json(self.state_path, self.state)
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

    def stage(
        self, stage_id: str, label: str, environment: str, script: str,
        config_path: Path, extra: list[str] | None = None,
    ) -> None:
        argv = [
            "conda", "run", "--no-capture-output", "-n", environment,
            "python", str(ROOT / "scripts" / script), "--config", str(config_path),
            *(extra or []),
        ]
        stage = self.state["stages"][stage_id]
        stage.update(status="RUNNING", started_at=_utc_now(), argv=argv)
        self.state.update(status="RUNNING", current_stage=stage_id, updated_at=_utc_now())
        atomic_json(self.state_path, self.state)
        print(f"\n[{stage_id}] {label}", flush=True)
        started = time.monotonic()
        self.process = subprocess.Popen(argv, cwd=str(ROOT), start_new_session=True)
        return_code = self.process.wait()
        self.process = None
        stage.update(
            status="COMPLETED" if return_code == 0 else "INTERRUPTED" if self.interrupted else "FAILED",
            completed_at=_utc_now(), elapsed_seconds=time.monotonic() - started,
            return_code=return_code,
        )
        self.state["updated_at"] = _utc_now()
        atomic_json(self.state_path, self.state)
        if return_code != 0:
            if self.interrupted:
                raise KeyboardInterrupt(f"{label} interrupted")
            raise RuntimeError(f"{label} failed with exit code {return_code}")

    def skip(self, stage_id: str, reason: str) -> None:
        self.state["stages"][stage_id].update(
            status="SKIPPED", reason=reason, completed_at=_utc_now(), elapsed_seconds=0.0,
        )
        self.state["updated_at"] = _utc_now()
        atomic_json(self.state_path, self.state)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a resumable terminal-only RynnValue + IQL training pipeline"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "terminal_pipeline.yaml")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild the prepared manifest")
    parser.add_argument("--force-annotate", action="store_true", help="Recompute all RynnValue outputs")
    args = parser.parse_args()

    config = load_terminal_config(args.config)
    raw = merged_training_config(config)
    roots = dataset_roots(raw, config.pipeline_root / "imports")
    candidates, rejected = discover_candidates(
        roots, raw["data"]["project_id"],
        int(raw["data"]["success_consecutive_steps"]),
    )
    canonical_task = resolve_task_id(config.selection["task_id"], candidates)
    raw["data"]["task_ids"] = [canonical_task]
    selected = select_candidates(candidates, config.selection, canonical_task)
    selection_manifest = build_selection_manifest(
        selected, raw, config.selection, canonical_task,
    )
    fingerprint = prepare_fingerprint(selection_manifest, raw)
    work_dir = config.pipeline_root / "cache" / "prepared" / fingerprint / "work"
    selection_path = (
        config.pipeline_root / "datasets" / selection_manifest["id"] / "dataset.json"
    )
    raw["paths"]["work_dir"] = str(work_dir.resolve())
    raw["data"]["selection_manifest"] = str(selection_path.resolve())
    validate_effective_config(raw)
    source = reward_source(raw["reward"])
    stage_snapshot = None
    if source == "stage":
        stage_snapshot = load_stage_annotations(LoadedConfig(config.path, raw), {
            "episodes": [{"run_id": member["run_id"],
                          "trajectory_path": member["artifacts"]["trajectory"]["path"],
                          "trajectory_sha256": member["artifacts"]["trajectory"]["sha256"]}
                         for member in selection_manifest["members"]],
        })

    prepare_skip = prepare_cache_valid(work_dir, fingerprint)
    annotation_skip = (
        source == "rynnvalue" and prepare_skip and annotation_cache_valid(work_dir, raw["reward"])
        and not args.force_annotate
    )
    reward_skip = (
        annotation_skip and reward_cache_valid(work_dir, raw["reward"])
    )
    evaluated_count = bound_evaluation_count(selection_manifest) if source == "rynnvalue" else 0
    _print_preflight(
        task_id=canonical_task, candidates=candidates, selected=selected,
        rejected_count=len(rejected), selection_manifest=selection_manifest, raw=raw,
        prepare_skip=prepare_skip, annotation_skip=annotation_skip,
        reward_skip=reward_skip,
        evaluated_count=evaluated_count, force_prepare=args.force_prepare,
        force_annotate=args.force_annotate,
    )
    if args.dry_run:
        print("Dry run complete; no pipeline, dataset, or training result was created.")
        return 0
    _confirm(args.yes)
    needed_environments = {config.environments["prepare"], config.environments["train"]}
    if source == "rynnvalue":
        needed_environments.add(config.environments["annotate"])
    _verify_conda_environments(needed_environments)

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    if selection_path.is_file():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection_manifest:
            raise RuntimeError(f"Frozen selection hash collision at {selection_path}")
    else:
        atomic_json(selection_path, selection_manifest)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}__{uuid.uuid4().hex[:8]}"
    run_dir = config.pipeline_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_path = run_dir / "effective_config.yaml"
    if stage_snapshot is not None:
        snapshot_path = run_dir / "stage_annotations.json"
        atomic_json(snapshot_path, stage_snapshot)
        raw["data"]["stage_annotations_manifest"] = str(snapshot_path.resolve())
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
        "selection_sha256": selection_manifest["dataset_sha256"],
        "selected_count": len(selected),
        "prepare_fingerprint": fingerprint,
        "effective_config": str(effective_path),
        "work_dir": str(work_dir),
        "training_result": None,
        "cache": {
            "prepare_hit": bool(prepare_skip and not args.force_prepare),
            "annotation_manifest_hit": bool(annotation_skip),
            "reward_manifest_hit": bool(reward_skip),
            "bound_evaluations_before": evaluated_count,
        },
        "error": None,
        "stages": {
            name: {"status": "PENDING", "started_at": None, "completed_at": None}
            for name in ("prepare", "annotate", "rewards", "bind", "train")
        },
    }
    atomic_json(state_path, state)
    runner = StageRunner(state_path, state)
    runner.install_signal_handlers()
    try:
        if prepare_skip and not args.force_prepare:
            runner.skip("prepare", "prepare fingerprint and manifest hash match")
        else:
            runner.stage(
                "prepare", "Prepare selected trajectories", config.environments["prepare"],
                "prepare_dataset.py", effective_path,
            )
            mark_prepare_cache(
                work_dir, fingerprint, selection_manifest["dataset_sha256"],
            )

        prepared = json.loads(
            (work_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        state["prepared_dataset"] = {
            "dataset_sha256": prepared["dataset_sha256"],
            "episode_count": prepared["episode_count"],
            "success_count": prepared["success_count"],
            "chunk_count": prepared["chunk_count"],
        }
        atomic_json(state_path, state)

        if source != "rynnvalue":
            runner.skip("annotate", f"{source} rewards do not use RynnValue")
        elif annotation_cache_valid(work_dir, raw["reward"]) and not args.force_annotate:
            runner.skip("annotate", "complete official-output annotation cache matches")
        else:
            runner.stage(
                "annotate", "RynnValue trajectory evaluation", config.environments["annotate"],
                "annotate_rewards.py", effective_path,
                ["--overwrite"] if args.force_annotate else None,
            )

        if reward_cache_valid(work_dir, raw["reward"]) and not args.force_annotate:
            runner.skip("rewards", "deterministic reward derivation cache matches")
        else:
            runner.stage(
                "rewards", f"Derive {source} IQL rewards",
                config.environments["prepare"], "materialize_rewards.py", effective_path,
                ["--force"] if args.force_annotate else None,
            )

        if source == "rynnvalue":
            bind_started = time.monotonic()
            state["stages"]["bind"].update(status="RUNNING", started_at=_utc_now())
            state.update(current_stage="bind", updated_at=_utc_now())
            atomic_json(state_path, state)
            binding = bind_reward_manifest(
                work_dir / "dataset_manifest.json",
                work_dir / "rewards" / "reward_manifest.json",
            )
            state["stages"]["bind"].update(
                status="COMPLETED", completed_at=_utc_now(),
                elapsed_seconds=time.monotonic() - bind_started, result=binding,
            )
            state["cache"]["bound_count"] = binding["bound_count"]
            state["cache"]["bound_reused_count"] = binding["skipped_count"]
            atomic_json(state_path, state)
            print(f"\n[bind] bound={binding['bound_count']} reused={binding['skipped_count']}")
        else:
            runner.skip("bind", "Independent rewards never modify RynnValue sidecars")
        if runner.interrupted:
            raise KeyboardInterrupt("Pipeline stop requested during evaluation binding")

        train_result = run_dir / "train_result.json"
        runner.stage(
            "train", "VLA-Adapter IQL post-training", config.environments["train"],
            "train_iql.py", effective_path,
            ["--result-file", str(train_result)],
        )
        result = json.loads(train_result.read_text(encoding="utf-8"))
        state.update(
            status="COMPLETED", current_stage=None, completed_at=_utc_now(),
            updated_at=_utc_now(), training_result=result,
        )
        atomic_json(state_path, state)
        print(f"\nPipeline completed: {run_dir}")
        print(f"Policy overlay    : {result['policy_overlay']}")
        return 0
    except KeyboardInterrupt as exc:
        state.update(
            status="INTERRUPTED", current_stage=None, completed_at=_utc_now(),
            updated_at=_utc_now(), error=str(exc),
        )
        atomic_json(state_path, state)
        print(f"\nPipeline interrupted; state saved to {state_path}", file=sys.stderr)
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
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
