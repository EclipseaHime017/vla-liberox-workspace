#!/usr/bin/env python3
"""YAML-driven server training for SSH, nohup and Slurm jobs."""
import argparse
import json
import signal
import sys
import time
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_rynn_iql.server_config import configure_server
from vla_rynn_iql.server_pipeline import ServerRun, execution_plan, training_settings
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.server_inputs import choose_task, discover_tasks, prepare_global_inputs, open_run_source


def run_batch(run):
    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        # Signals received while verifying/copying inputs are deferred until a
        # worker exists, so neither the supervisor nor its children are orphaned.
        run.start()
        print(f"Log: {run.directory / 'training.log'}", flush=True)
        last_step = None
        while run.poll() is None:
            if stop_requested and run.cancel_at is None:
                run.cancel()
                print("Safe stop requested; waiting for distributed checkpoint...", flush=True)
            progress = run.progress()
            if progress and progress.get("step") != last_step:
                last_step = progress.get("step")
                print(json.dumps(progress), flush=True)
            time.sleep(1)
    finally:
        try:
            run.shutdown()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(run.state, ensure_ascii=False), flush=True)
    return 0 if run.state["status"] == "COMPLETED" else 130 if run.state["status"] == "INTERRUPTED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/server_pipeline.yaml")
    parser.add_argument("--task", help="Canonical task ID from the export or run collection")
    parser.add_argument("--runs-root", type=Path, help="Copied runs, exported dataset directory or ZIP")
    parser.add_argument("--model", help="Registered model variant ID")
    parser.add_argument("--method", choices=("bc", "iql"), help="Training method; overrides YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show plan without writing a training run")
    args = parser.parse_args(argv)
    server = configure_server(args.config, task=args.task, runs_root=args.runs_root,
                              model=args.model, method=args.method)
    with open_run_source(server.runs_root) as root:
        return execute(replace(server, runs_root=root), dry_run=args.dry_run)


def execute(server, *, dry_run=False):
    project = load_train_config(server.training_config).raw["data"]["project_id"]
    task = choose_task(discover_tasks(server.runs_root, project), server.task_id)
    raw = training_settings(server, task)
    plan = execution_plan(server, task, raw, server.distributed, server.reward_source)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="server-input-check-") as temp:
            raw["paths"]["output_dir"] = str(Path(temp) / "output")
            prepare_global_inputs(task, Path(temp) / "inputs", raw, server.reward_source,
                                  data_overrides=server.overrides["data"])
        return 0
    run = ServerRun(server, task, raw, server.distributed, server.reward_source)
    return run_batch(run)


if __name__ == "__main__":
    raise SystemExit(main())
