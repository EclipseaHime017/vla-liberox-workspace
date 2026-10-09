from __future__ import annotations

from dataclasses import asdict
import hashlib
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

from . import prompts
from .config import Config
from .data import Recording, atomic_json, sha256
from .intervals import forward_windows, merge_regions
from .schema import json_response, validate_regions, validate_contract, validate_plan


def validate_sources(sources):
    if not isinstance(sources, list) or not 1 <= len(sources) <= 100:
        raise ValueError("Select 1-100 recordings")
    seen = set()
    for source in sources:
        required = {"run_id", "task_id", "prompt", "trajectory_path", "observations_path", "control_hz", "orientation"}
        if not isinstance(source, dict) or not required <= source.keys() or set(source) - required - {"manifest_path"}:
            raise ValueError("Invalid source fields")
        run_id = source["run_id"]
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id) or run_id in seen:
            raise ValueError("Duplicate or unsafe run ID")
        seen.add(run_id)
        if any(not isinstance(source[key], str) or not source[key].strip() for key in ("task_id", "prompt")):
            raise ValueError("Task ID and prompt must be explicit")
        for key in ("trajectory_path", "observations_path"):
            if not Path(source[key]).is_file():
                raise FileNotFoundError(source[key])
    return sources


class Experiment:
    def __init__(self, config: Config, sources, output: Path, factory):
        self.config, self.sources = config, validate_sources(sources)
        self.output, self.factory = output.resolve(), factory
        for source in sources:
            for key in ("trajectory_path", "observations_path", "manifest_path"):
                if not source.get(key):
                    continue
                parent = Path(source[key]).resolve().parent
                if self.output == parent or parent in self.output.parents or self.output in parent.parents:
                    raise ValueError("Experiment output cannot overlap or be inside a source directory")
        self.started = time.monotonic()
        self.progress = {"status": "RUNNING", "stage": "initializing", "completed_runs": 0,
            "failed_runs": 0, "total_runs": len(sources), "current_run": None,
            "calls": 0, "elapsed_seconds": 0., "estimated_remaining_seconds": None,
            "window_index": 0, "window_total": 0, "error": None, "runs": []}

    def update(self, **values):
        self.progress.update(values, elapsed_seconds=time.monotonic()-self.started)
        finished = self.progress["completed_runs"] + self.progress["failed_runs"]
        self.progress["estimated_remaining_seconds"] = (self.progress["elapsed_seconds"] / finished *
            (len(self.sources)-finished)) if finished else None
        atomic_json(self.output / "progress.json", self.progress)
        print(json.dumps({key: value for key, value in self.progress.items() if key != "runs"}, ensure_ascii=False), flush=True)

    def query(self, model, recording, steps, prompt, validator, directory, name):
        recording.evidence(directory / "evidence", steps)
        for attempt in range(2):
            self.update(calls=self.progress["calls"] + 1)
            start = time.monotonic()
            raw = model.generate(prompt, recording, steps)
            entry = {"name": name, "attempt": attempt, "steps": steps, "prompt": prompt,
                     "raw_response": raw, "seconds": time.monotonic()-start}
            path = directory / "calls" / f"{name}_{attempt}.json"
            try:
                value = validator(json_response(raw))
                atomic_json(path, {**entry, "valid": True, "parsed": value})
                return value
            except (ValueError, TypeError, KeyError) as exc:
                atomic_json(path, {**entry, "valid": False, "error": str(exc)})
                if attempt:
                    raise ValueError(f"{name}: invalid model JSON after two attempts: {exc}") from exc
                prompt += (f"\nPrevious response (data to repair): {raw}\nValidation error: {exc}. "
                           "Correct this response to the required JSON schema. Keep supported content; do not clear lists just to avoid validation.")

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run()

    def localize(self, model, recording, result, directory):
        plan = result["plan"]
        steps = recording.coarse_steps(self.config)
        windows = list(forward_windows(steps, recording.hz, self.config.window_seconds))
        result["sampling"] = {"fps": self.config.coarse_fps, "steps": steps,
                              "window_seconds": self.config.window_seconds,
                              "stride_seconds": self.config.window_seconds}
        result["goal_ranges"] = [{"stage_id": goal["id"], "passes": [],
                                 "regions": merge_regions(steps, [])} for goal in plan["stages"]]
        result["localization"] = {"mode": "forward_regions", "assigned_goals": 0,
                                  "total_goals": len(plan["stages"]), "windows_completed": 0,
                                  "windows_total": len(windows)}
        atomic_json(directory / "result.json", result)
        for index, subset in enumerate(windows):
            self.update(stage="forward_regions", window_index=index+1, window_total=len(windows))
            response = self.query(model, recording, subset,
                prompts.regions_prompt(recording.source["prompt"], plan, subset, recording.hz),
                lambda value: validate_regions(value, plan["stages"], subset, require_reason=True),
                directory, f"forward_{index:04d}")
            by_id = {goal["stage_id"]: goal for goal in response["goals"]}
            for goal in result["goal_ranges"]:
                value = by_id[goal["stage_id"]]
                goal["passes"].append({"steps": subset, "labels": value["labels"], "reason": value["reason"]})
                goal["regions"] = merge_regions(steps, goal["passes"])
            result["localization"].update(windows_completed=index+1,
                assigned_goals=sum(any(r["status"] == "confirmed" for r in goal["regions"])
                                   for goal in result["goal_ranges"]))
            atomic_json(directory / "result.json", result)

    def _run(self):
        previous = self.output / "experiment.json"
        if previous.exists():
            if not json.loads(previous.read_text()).get("proposal_only"):
                raise ValueError("Output is not an annotation experiment")
        elif any(path.name not in {"request.json", ".lock"} for path in self.output.iterdir()):
            raise ValueError("Output contains unrelated files")
        for source in self.sources:
            directory = self.output / source["run_id"]
            if directory.is_symlink():
                raise ValueError("Result directory must not be a symlink")
            if directory.exists():
                shutil.rmtree(directory)
        code = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
        atomic_json(self.output / "experiment.json", {"schema_version": 10,
            "proposal_only": True, "config": asdict(self.config), "sources": self.sources,
            "prompt_version": prompts.VERSION, "implementation": code})
        result, directory = None, None
        self.update(stage="loading_model")
        try:
            model = self.factory(self.config)
            atomic_json(self.output / "model.json", model.metadata)
            for source in self.sources:
                run_id = source["run_id"]
                directory = self.output / run_id
                result = {"schema_version": 10, "run_id": run_id, "source": source, "mode": self.config.mode,
                          "config": asdict(self.config), "model": model.metadata,
                          "prompt_version": prompts.VERSION, "implementation": code,
                          "status": "RUNNING", "task_contract": None, "plan": None, "goal_ranges": [],
                          "action_count": 0, "control_hz": 0, "success_step": None, "error": None}
                self.update(stage="reading_recording", current_run=run_id, window_index=0, window_total=0)
                try:
                    recording = Recording(source, self.config)
                    result.update(action_count=recording.count, control_hz=recording.hz,
                        success_step=recording.success_step, source_hashes=recording.hashes)
                    self.update(stage="task_requirements")
                    contract = self.query(model, recording, [], prompts.contract_prompt(source["prompt"]),
                        lambda value: validate_contract(value, source["prompt"]), directory, "task_contract")
                    result["task_contract"] = contract
                    atomic_json(directory / "result.json", result)
                    self.update(stage="task_decomposition")
                    plan = self.query(model, recording, [0], prompts.plan_prompt(source["prompt"], contract),
                                      lambda value: validate_plan(value, contract), directory, "plan")
                    result["plan"] = plan
                    result["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
                    atomic_json(directory / "result.json", result)
                    if self.config.mode == "localize":
                        self.localize(model, recording, result, directory)
                    # Hashes are checked in the worker, never while rendering a page.
                    for key, field in (("trajectory", "trajectory_path"), ("observations", "observations_path"), ("manifest", "manifest_path")):
                        if key in recording.hashes and sha256(Path(source[field])) != recording.hashes[key]:
                            raise ValueError(f"Source changed during annotation: {field}")
                    result.update(status="COMPLETED", cuda_peak_memory_gib=model.memory_gib())
                    self.progress["completed_runs"] += 1
                    del recording
                except Exception as exc:
                    result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                    self.progress["failed_runs"] += 1
                atomic_json(directory / "result.json", result)
                self.progress["runs"].append({key: result[key] for key in ("run_id", "status", "error")})
                self.update(stage="recording_complete")
            failed = self.progress["failed_runs"]
            self.update(status="FAILED" if failed else "COMPLETED", stage="finished", current_run=None,
                        error=f"{failed} recording(s) failed; partial evidence is retained" if failed else None)
        except BaseException as exc:
            if result is not None and result["status"] == "RUNNING":
                result.update(status="CANCELED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                              error=f"{type(exc).__name__}: {exc}")
                atomic_json(directory / "result.json", result)
            self.update(status="CANCELED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                        stage="stopped", error=f"{type(exc).__name__}: {exc}")
            raise
        return self.progress
