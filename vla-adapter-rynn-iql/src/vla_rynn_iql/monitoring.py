from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")

TENSORBOARD_GROUPS = {
    "loss": ("q_loss", "value_loss", "actor_loss"),
    "value": ("q_mean", "value_mean", "advantage_mean"),
    "iql": ("advantage_weight_mean",),
    "optimization": (
        "actor_learning_rate",
        "actor_grad_norm",
        "action_head_parameter_norm",
        "proprio_projector_parameter_norm",
    ),
    "action_l1": tuple(f"actor_l1_{name}" for name in ACTION_NAMES),
    "gripper": (
        "actor_gripper_prediction_mean",
        "actor_gripper_target_mean",
        "actor_gripper_target_close_fraction",
    ),
    "system": ("steps_per_second", "cuda_peak_memory_gib"),
}


def log_tensorboard_metric(writer: Any, metric: dict[str, Any]) -> None:
    """Write one JSONL metric record using stable, grouped TensorBoard tags."""
    if writer is None:
        return
    step = int(metric["step"])
    for group, keys in TENSORBOARD_GROUPS.items():
        for key in keys:
            value = metric.get(key)
            if value is not None:
                writer.add_scalar(f"{group}/{key}", float(value), step)


def read_metrics_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield validated records from a trainer metrics.jsonl file."""
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                metric = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(metric, dict) or type(metric.get("step")) is not int:
                raise ValueError(
                    f"Metric at {path}:{line_number} must be an object with integer step"
                )
            if "steps_per_second" not in metric:
                elapsed = metric.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)) and elapsed > 0:
                    metric["steps_per_second"] = metric["step"] / float(elapsed)
            if "cuda_peak_memory_gib" not in metric:
                memory = metric.get("cuda_peak_memory_bytes")
                if isinstance(memory, (int, float)):
                    metric["cuda_peak_memory_gib"] = float(memory) / (1024.0 ** 3)
            yield metric


def write_metrics_to_tensorboard(
    metrics_path: Path,
    writer: Any,
) -> int:
    count = 0
    for metric in read_metrics_jsonl(metrics_path):
        log_tensorboard_metric(writer, metric)
        count += 1
    return count


def convert_run_metrics(run_dir: Path, log_dir: Path | None = None) -> tuple[Path, int]:
    """Convert a completed legacy run's JSONL metrics into TensorBoard events."""
    run_dir = run_dir.expanduser().resolve()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Training metrics do not exist: {metrics_path}")
    target = (log_dir or (run_dir / "tensorboard-imported")).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"TensorBoard import directory is not empty: {target}; choose another --log-dir"
        )
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "tensorboard is not installed; install requirements-train.txt"
        ) from exc
    writer = SummaryWriter(log_dir=str(target))
    try:
        count = write_metrics_to_tensorboard(metrics_path, writer)
        writer.flush()
    finally:
        writer.close()
    return target, count
