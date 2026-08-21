from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    "system": (
        "steps_per_second",
        "progress_percent",
        "estimated_remaining_seconds",
        "cuda_peak_memory_gib",
    ),
}


def format_duration(seconds: float | int | None) -> str:
    """Format a duration for stable, compact terminal progress output."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "--:--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _metric_number(metric: dict[str, Any], key: str, precision: int = 4) -> str:
    value = metric.get(key)
    if value is None or not isinstance(value, (int, float)):
        return "--"
    return f"{float(value):.{precision}g}"


@dataclass
class TrainingProgressReporter:
    """Compute a rolling ETA and format periodic, log-friendly progress lines."""

    total_steps: int
    start_step: int
    interval_steps: int
    warmup_steps: int
    window_steps: int = 100
    _points: deque[tuple[int, float]] = field(init=False)

    def __post_init__(self) -> None:
        if self.total_steps <= self.start_step:
            raise ValueError("total_steps must be larger than start_step")
        if self.interval_steps <= 0 or self.window_steps <= 0:
            raise ValueError("progress intervals must be positive")
        self._points = deque(maxlen=self.window_steps + 1)

    def update(self, metric: dict[str, Any]) -> bool:
        """Add derived metrics and return whether this step should be printed."""
        step = int(metric["step"])
        elapsed = float(metric["elapsed_seconds"])
        self._points.append((step, elapsed))
        if len(self._points) >= 2:
            old_step, old_elapsed = self._points[0]
            rate = (step - old_step) / max(elapsed - old_elapsed, 1e-9)
        else:
            rate = (step - self.start_step) / max(elapsed, 1e-9)
        remaining = max(self.total_steps - step, 0)
        eta_seconds = remaining / rate if rate > 0 else None
        metric["steps_per_second"] = rate
        metric["progress_percent"] = 100.0 * step / self.total_steps
        metric["estimated_remaining_seconds"] = eta_seconds
        metric["estimated_completion_time"] = (
            (datetime.now().astimezone() + timedelta(seconds=eta_seconds))
            .isoformat(timespec="seconds")
            if eta_seconds is not None else None
        )
        completed_this_run = step - self.start_step
        return (
            completed_this_run == 1
            or step == self.total_steps
            or completed_this_run % self.interval_steps == 0
        )

    def format(self, metric: dict[str, Any]) -> str:
        step = int(metric["step"])
        fraction = min(max(step / self.total_steps, 0.0), 1.0)
        width = 20
        complete = min(width, int(fraction * width))
        bar = "#" * complete + "-" * (width - complete)
        phase = "BC-warmup" if step <= self.warmup_steps else "IQL"
        finish = metric.get("estimated_completion_time") or "unknown"
        return (
            f"TRAIN [{bar}] {step}/{self.total_steps} "
            f"({metric['progress_percent']:6.2f}%) | phase={phase} | "
            f"elapsed={format_duration(metric.get('elapsed_seconds'))} | "
            f"ETA={format_duration(metric.get('estimated_remaining_seconds'))} "
            f"(finish {finish}) | {metric['steps_per_second']:.3f} step/s | "
            f"loss(q/v/actor)={_metric_number(metric, 'q_loss')}/"
            f"{_metric_number(metric, 'value_loss')}/"
            f"{_metric_number(metric, 'actor_loss')} | "
            f"Q/V/A={_metric_number(metric, 'q_mean')}/"
            f"{_metric_number(metric, 'value_mean')}/"
            f"{_metric_number(metric, 'advantage_mean')} | "
            f"weight={_metric_number(metric, 'advantage_weight_mean')} | "
            f"lr={_metric_number(metric, 'actor_learning_rate', 3)} | "
            f"VRAM={_metric_number(metric, 'cuda_peak_memory_gib', 3)} GiB"
        )


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
