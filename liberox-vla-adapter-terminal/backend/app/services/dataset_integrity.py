"""Explicit, CPU-only verification of source files and scoped reward inputs."""
from __future__ import annotations

from .dataset_export_rewards import ExportRewards
from .global_reward_binding import global_members
from .trajectory_reward_snapshot import SOURCES


def _check(operation) -> dict:
    try:
        value = operation()
        if value is None:
            return {"status": "MISSING", "error": "缺少评价结果"}
        return {"status": "READY"}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"status": "ERROR", "error": str(exc)}


def _counts(states: list[dict]) -> dict:
    return {"total": len(states), **{name: sum(s["status"] == status for s in states)
            for name, status in (("ready", "READY"), ("missing", "MISSING"), ("invalid", "ERROR"))}}


def _label_state(stages, run_id: str, dataset_id: str | None = None) -> dict:
    try:
        view = stages.detail(run_id, dataset_id)
        return {"status": {"ready": "READY", "missing": "MISSING"}.get(view["status"], "ERROR"),
                "error": view.get("error"), "origin": view.get("origin")}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"status": "ERROR", "error": str(exc)}


def _global_states(jobs, dataset: dict, source: str, *, use_dataset_labels: bool = False) -> dict:
    records, errors = global_members(jobs, dataset, source, use_dataset_labels=use_dataset_labels)
    return {**{member["run_id"]: {"status": "READY"} for member, *_ in records},
            **{item["run_id"]: {"status": item.get("status", "ERROR"), "error": item["error"]}
               for item in errors}}


def verify_dataset(jobs, dataset_id: str) -> dict:
    """Run in the work queue, not list requests; no model forward or reward writes."""
    dataset = jobs.datasets.verify(dataset_id)
    _, frozen = jobs.datasets._load(dataset_id)
    members = frozen["members"]
    local = ExportRewards(jobs, dataset)
    global_only = ExportRewards(jobs, {**dataset, "evaluation_version_ids": {}})
    report = {"dataset_id": dataset_id, "dataset_name": dataset["name"],
              "checked_at": dataset["last_verified_at"], "member_count": len(members),
              "integrity_status": dataset["integrity_status"], "integrity_error": dataset["integrity_error"],
              "sources": []}
    for source in ("keyframes", *SOURCES, "robometer"):
        global_states = (_global_states(jobs, dataset, source) if source in SOURCES else {})
        inherited_stage = (_global_states(jobs, dataset, source, use_dataset_labels=True)
                           if source == "stage" and not dataset["evaluation_version_ids"].get(source) else {})
        scopes = {"global": [], "dataset": [], "effective": []}
        issues, inherited = [], 0
        for member in members:
            run_id = member["run_id"]
            owned = bool(dataset["evaluation_version_ids"].get(source))
            if source == "keyframes":
                global_state = _label_state(jobs.stage_annotations, run_id)
                local_state = _label_state(jobs.stage_annotations, run_id, dataset_id)
                effective = local_state
                is_inherited = local_state.get("origin") == "global_copy"
                owned = local_state.get("origin") not in {"global_copy", "missing"}
            else:
                global_state = (global_states[run_id] if source in SOURCES
                                else _check(lambda: global_only.robometer(member)))
                local_state = (_check(lambda: local.robometer(member) if source == "robometer"
                                      else local.reward(member, source)) if owned else {"status": "MISSING"})
                effective = local_state if owned else inherited_stage.get(run_id, global_state)
                is_inherited = not owned
            for scope, state in (("global", global_state), ("dataset", local_state), ("effective", effective)):
                scopes[scope].append(state)
                if state["status"] != "READY" and (scope != "dataset" or owned):
                    message = state.get("error") or "缺少评价结果"
                    if scope == "effective" and not owned:
                        message = f"数据集无专属结果，且全局评价不可用：{message}"
                    issues.append({"run_id": run_id, "scope": scope, "error": message})
            inherited += int(is_inherited and effective["status"] == "READY")
        report["sources"].append({"source": source, **{scope: _counts(values) for scope, values in scopes.items()},
                                  "inherited": inherited, "issues": issues})
    return report
