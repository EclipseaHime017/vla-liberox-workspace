#!/usr/bin/env python3
"""Validate one model end to end with fixed smoke-test settings via the platform queue."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


TRAIN_STEPS = 2
CONTROL_STEPS = 40
SEED = 0
METHOD = "bc"
PLATFORM_URL = "http://127.0.0.1:8000"
TASK_ID = "LEVEL1::EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove"


class JobFailed(RuntimeError):
    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


class Client:
    def __init__(self, url: str):
        if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).hostname:
            raise ValueError("Platform address must use HTTP(S)")
        self.url = url.rstrip("/")

    def request(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=180) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Platform request failed: {exc}; requests are not automatically retried") from exc


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--model", required=True, help="Exact registered base variant ID")
    return root


def simulation_request(entry):
    return {
        "policy_id": entry["policy_id"], "task_id": TASK_ID, "trials": 1,
        "max_steps": CONTROL_STEPS, "open_loop_steps": entry["io"]["replay_horizon"],
        "realtime": False, "init_state_indices": [0],
        "base_seed": SEED, "seed_count": 1, "schedule_seed": SEED,
    }


def prepare(client, args):
    entry = client.request("GET", f"/api/models/{quote(args.model, safe='')}")
    if entry["policy_id"] != args.model:
        raise ValueError("Platform returned a different model ID")
    if entry["kind"] != "base":
        raise ValueError("Training tests require a registered base model, not an exported child")
    selected = {"algorithm": METHOD, "model_family": entry["family"], "model_base_id": args.model}
    defaults = client.request("GET", "/api/training/defaults?" + urlencode(selected))
    if (defaults["algorithm"] != METHOD or defaults["model"]["model_base_id"] != args.model
            or defaults["model"]["model_family"] != entry["family"]):
        raise ValueError("Training defaults do not match the selected model")
    parameters = {
        **selected, "train_steps": TRAIN_STEPS, "micro_batch_size": 1,
        "gradient_accumulation_steps": 1, "checkpoint_interval": TRAIN_STEPS,
        "actor_lr_warmup_steps": 0, "seed": SEED, "resume_checkpoint": None,
        "console_interval_steps": 1, "tensorboard": False, "wandb_enabled": False,
    }
    return entry, parameters


def wait_for_job(client, job):
    path = f"/api/jobs/{quote(job['id'], safe='')}"
    offset, previous = 0, None
    try:
        if "status" not in job:
            job = client.request("GET", path)
        while True:
            if job["status"] != previous:
                print(f"{job['id']}: {job['status']}", flush=True)
                previous = job["status"]
            logs = client.request("GET", f"{path}/logs?offset={offset}")
            print(logs["text"], end="", flush=True)
            offset = logs["next_offset"]
            if job["status"] in {"COMPLETED", "FAILED", "CANCELED"}:
                print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
                if job["status"] != "COMPLETED":
                    location = job.get("config_path") or job.get("output_path") or path
                    raise JobFailed(
                        f"{job['id']} / {job.get('stage') or job['status']}: "
                        f"{job.get('error') or job['status']}\nLocation: {location}\nLogs: {path}/logs",
                        130 if job["status"] == "CANCELED" else 1,
                    )
                return job
            time.sleep(2)
            job = client.request("GET", path)
    except KeyboardInterrupt:
        client.request("POST", path + "/stop")
        raise JobFailed(f"Stop requested for {job['id']}; status remains available in the platform.", 130)


def record_fixture(client, entry):
    record = client.request("POST", "/api/sessions", {
        "policy_id": entry["policy_id"], "task_id": TASK_ID,
        "max_steps": CONTROL_STEPS, "open_loop_steps": entry["io"]["replay_horizon"],
        "seed": SEED, "init_state_index": 0, "disabled_policy_cameras": [],
    })
    wait_for_job(client, {"id": record["work_job_id"]})
    record = client.request("GET", f"/api/sessions/{quote(record['id'], safe='')}")
    if (record["status"] != "COMPLETED" or record["error"]
            or record["policy_id"] != entry["policy_id"] or record["task_id"] != TASK_ID
            or record["action_count"] != CONTROL_STEPS or record["state_count"] != CONTROL_STEPS + 1
            or record["policy_queries"] < 1 or not record["trajectory"]
            or "episodes/episode_000/trajectory_observations.npz" not in record["artifacts"]):
        raise ValueError("Base simulation did not save the complete test trajectory")
    pinned = client.request("GET", f"/api/models/{quote(entry['policy_id'], safe='')}")
    if pinned["content_sha256"] != record["policy_content_sha256"]:
        raise ValueError("Base model changed after the recorded simulation")
    return record, pinned


def freeze_fixture(client, record):
    job = wait_for_job(client, client.request("POST", "/api/training-datasets", {
        "name": f"Model smoke · {record['id']}", "task_id": TASK_ID,
        "selection": {"mode": "manual", "run_ids": [record["id"]]},
        "validation_fraction": 0.0, "split_seed": SEED,
        "success_consecutive_steps": 5, "include_post_success": True,
    }))
    identifier = job["result"]["dataset_id"]
    dataset = client.request("GET", f"/api/training-datasets/{quote(identifier, safe='')}")
    if (dataset["id"] != identifier or dataset["task_id"] != TASK_ID
            or [member["run_id"] for member in dataset["members"]] != [record["id"]]
            or dataset["integrity_status"] == "BROKEN"):
        raise ValueError("Generated test dataset does not match the recorded trajectory")
    # Frozen membership is unchanged; exclude the recording from future research packages.
    client.request("PATCH", f"/api/datasets/runs/{quote(record['id'], safe='')}/labels", {"is_test": True})
    return identifier


def simulate(client, entry):
    job = wait_for_job(client, client.request("POST", "/api/evaluations", simulation_request(entry)))
    result = client.request("GET", f"/api/evaluations/{quote(job['id'], safe='')}")
    aggregate, policy = result["aggregate"], result["policy_snapshot"]
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    # Failure to finish the task is not a runtime error in a short smoke test.
    if (aggregate["errors"] or aggregate["completed_trials"] != 1
            or len(result["trials"]) != 1 or result["trials"][0]["steps"] != CONTROL_STEPS):
        raise JobFailed(f"{job['id']}: simulation did not complete a valid rollout")
    if policy["policy_id"] != entry["policy_id"] or policy["family"] != entry["family"]:
        raise ValueError("Simulation used a different model")
    if entry["kind"] != "base" and policy["content_sha256"] != entry["content_sha256"]:
        raise ValueError("Exported model changed before simulation")
    return policy


def exported_model(client, job, base, base_snapshot):
    summary, metrics = job["training_summary"], job["metrics"]
    if (summary["status"] != "completed" or summary["steps"] != TRAIN_STEPS
            or summary["algorithm"] != METHOD or summary["resumed_from_step"] != 0
            or summary["micro_batch_size"] != 1 or summary["transitions_processed"] != TRAIN_STEPS):
        raise ValueError("Training did not complete the fixed smoke-test budget")
    if metrics["step"] != TRAIN_STEPS or any(
        not math.isfinite(float(metrics[key])) for key in ("actor_loss", "actor_grad_norm", "actor_learning_rate")
    ) or metrics["actor_learning_rate"] <= 0:
        raise ValueError("Training metrics are missing, non-finite or indicate no learning")
    pinned = job["parameters"]["model"]
    if any(pinned[key] != base_snapshot[key] for key in ("base_checkpoint", "base_revision", "stats_key")):
        raise ValueError("Training and base simulation used different parent weights")
    path = PurePosixPath(summary["policy_overlay"])
    if not path.is_absolute() or path.name != "policy.yaml":
        raise ValueError("Training did not return an exported policy manifest")
    child = client.request("GET", f"/api/models/{quote(path.parent.name, safe='')}")
    if (child["policy_id"] != path.parent.name or child["kind"] == "base"
            or child["parent_model_id"] != base["policy_id"] or child["family"] != base["family"]
            or child["training_step"] != TRAIN_STEPS or child["algorithm"] != METHOD
            or child["io"] != base["io"] or not child["components"]
            or child["manifest"]["dataset_sha256"] != summary["dataset_sha256"]
            or any(child[key] != pinned[key] for key in ("base_checkpoint", "base_revision"))):
        raise ValueError("Exported child does not match this training run and its selected parent")
    return child


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        client = Client(PLATFORM_URL)
        base, parameters = prepare(client, args)
        print(json.dumps({
            "model": base["policy_id"], "model_config": base["model_config"],
            "stages": ["validate", "base simulation", "freeze test data", "training", "export validation", "child simulation"],
            "simulation": simulation_request(base), "training": parameters,
        }, ensure_ascii=False, indent=2), flush=True)
        print("[1/5] Base simulation and test recording", flush=True)
        record, base_snapshot = record_fixture(client, base)
        print("[2/5] Freeze dedicated test dataset", flush=True)
        dataset_id = freeze_fixture(client, record)
        print("[3/5] Short training", flush=True)
        job = wait_for_job(client, client.request("POST", "/api/training-runs", {
            "dataset_id": dataset_id, "parameters": parameters,
        }))
        print("[4/5] Export and parent validation", flush=True)
        child = exported_model(client, job, base, base_snapshot)
        print("[5/5] Exported child simulation", flush=True)
        simulate(client, child)
        print(f"PASS: {base['policy_id']} -> {child['policy_id']}", flush=True)
        return 0
    except JobFailed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("Interrupted before a job ID was received; check the platform queue before resubmitting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
