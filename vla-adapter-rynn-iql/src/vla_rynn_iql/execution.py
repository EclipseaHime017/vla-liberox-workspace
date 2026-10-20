"""Recorded execution layout, independent of the model trained from the data."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path

from .base_models import BASE_MODELS, execution_config

CHUNKING_RULE = "recorded_execution_v1"


def validate_task_prompt(source: dict, prompt: str | None, *, context: str) -> None:
    """Reject old custom-language data rather than silently relabelling its actions."""
    records = [source]
    for name in ("trajectory_segments", "chunks", "evaluation_chunks"):
        values = source.get(name)
        if isinstance(values, list):
            records.extend(value for value in values if isinstance(value, dict))
    for record in records:
        value = record.get("policy_instruction")
        if value is not None and (
            not isinstance(value, str) or not isinstance(prompt, str) or value.strip() != prompt.strip()
        ):
            raise ValueError(f"{context}: recorded custom instruction differs from the full task prompt; "
                             "this recording cannot be used for training or continuation")


def _steps(value) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Recorded open_loop_steps must be a positive integer")
    return value


def execution_steps(run: dict, metadata: dict, *, config_path: Path | None = None) -> tuple[int, str]:
    """Prefer saved execution settings; legacy records use their source model default."""
    for label, source in (("run", run), ("trajectory", metadata)):
        if source.get("open_loop_steps") is not None:
            return _steps(source["open_loop_steps"]), label
    if config_path is not None and config_path.is_file():
        import yaml
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not isinstance(saved.get("simulation", {}), dict):
            raise ValueError(f"Invalid saved simulation configuration: {config_path}")
        value = saved.get("simulation", {}).get("open_loop_steps")
        if value is not None:
            return _steps(value), "saved_simulation_config"
    for source in (run, metadata):
        snapshot = source.get("policy_snapshot") or {}
        value = snapshot.get("inference", {}).get("open_loop_steps")
        if value is not None:
            return _steps(value), "policy_snapshot"
        settings = snapshot.get("model_config") or source.get("model_config")
        if settings:
            trained = bool(snapshot.get("manifest") or source.get("policy_overlay"))
            if trained:
                from .model_artifacts import legacy_contract
                settings = {**settings, "contract": snapshot.get("model_contract") or legacy_contract(settings)}
            return execution_config(settings, trained=trained)["open_loop_steps"], "source_model_default"
        overlay = source.get("policy_overlay")
        if overlay:
            from .storage_paths import storage_path
            path = storage_path(overlay)
            if path.is_file():
                import yaml
                from .model_artifacts import decode_child, legacy_contract
                raw, parent = decode_child(yaml.safe_load(path.read_text(encoding="utf-8")))
                settings = {"family": raw.get("family", "vla_adapter"), **(raw.get("model_config") or {})}
                settings["contract"] = parent["contract"] if parent else legacy_contract(settings)
                return execution_config(settings, trained=True)["open_loop_steps"], "source_child_default"
        for name in ("policy_id", "base_id", "policy_base_id"):
            base = BASE_MODELS.get(source.get(name))
            if base is not None:
                return base.inference["open_loop_steps"], "source_model_default"
        checkpoint = source.get("policy_base_checkpoint") or source.get("base_checkpoint") or source.get("checkpoint")
        matches = [base for base in BASE_MODELS.values()
                   if checkpoint and checkpoint in (base.checkpoint, base.source)]
        if len(matches) == 1:
            return matches[0].inference["open_loop_steps"], "source_model_default"
    raise ValueError("Recording has no execution settings or identifiable source model; "
                     "restore its source model registration to use that model's open_loop_steps")


def query_steps(values, count: int) -> list[int]:
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
        raise ValueError("Recorded inference query steps must be integer action indices")
    result = [int(value) for value in values]
    if any(value < 0 or value > count for value in result) or result != sorted(set(result)):
        raise ValueError("Recorded inference query steps must be unique, increasing action indices")
    # A stop requested during the last query can leave an unexecuted proposal.
    return [value for value in result if value < count]


def nominal_chunk_count(run: dict) -> int:
    """Cheap list/package summary; Prepare resolves exact query/success boundaries."""
    count = int(run.get("action_count") or 0)
    if count == 0:
        return 0
    steps, _ = execution_steps(run, {})
    segments = run.get("trajectory_segments")
    if not segments:
        resume = int(run.get("resume_step") or 0) if run.get("kind") == "branch" else 0
        segments = [{"start": start, "end": end} for start, end in ((0, resume), (resume, count)) if end > start]
    return sum((segment["end"] - segment["start"] + int(segment.get("open_loop_steps", steps)) - 1)
               // int(segment.get("open_loop_steps", steps)) for segment in segments)
