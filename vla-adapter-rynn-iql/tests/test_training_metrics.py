from __future__ import annotations

import numpy as np
import torch

from vla_rynn_iql.monitoring import (
    convert_run_metrics,
    log_tensorboard_metric,
    write_metrics_to_tensorboard,
)
from vla_rynn_iql.training import _action_diagnostics, _module_parameter_norm


class FakeWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))


def test_action_diagnostics_expose_gripper_and_each_action_axis():
    target = torch.tensor(
        [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]]]
    )
    prediction = target + 0.25
    metrics = _action_diagnostics(prediction, target, torch.ones((1, 2)))
    for name in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"):
        assert np.isclose(metrics[f"actor_l1_{name}"], 0.25)
    assert np.isclose(metrics["actor_gripper_target_close_fraction"], 0.5)
    assert np.isclose(metrics["actor_gripper_target_mean"], 0.5)
    assert np.isclose(metrics["actor_gripper_prediction_mean"], 0.75)


def test_module_parameter_norm_reports_whole_module_l2_norm():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[3.0, 4.0]]))
    assert np.isclose(_module_parameter_norm(module), 5.0)


def test_tensorboard_groups_metrics_and_skips_missing_values():
    writer = FakeWriter()
    log_tensorboard_metric(
        writer,
        {
            "step": 12,
            "actor_loss": 0.4,
            "actor_grad_norm": None,
            "action_head_parameter_norm": 4.2,
            "actor_l1_gripper": 0.2,
            "actor_gripper_target_close_fraction": 0.3,
        },
    )
    assert ("loss/actor_loss", 0.4, 12) in writer.scalars
    assert ("action_l1/actor_l1_gripper", 0.2, 12) in writer.scalars
    assert (
        "gripper/actor_gripper_target_close_fraction", 0.3, 12
    ) in writer.scalars
    assert ("optimization/action_head_parameter_norm", 4.2, 12) in writer.scalars
    assert not any(name == "optimization/actor_grad_norm" for name, _, _ in writer.scalars)


def test_existing_jsonl_metrics_can_be_imported(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"step": 2, "q_loss": 1.5, "elapsed_seconds": 4.0, '
        '"cuda_peak_memory_bytes": 1073741824}\n',
        encoding="utf-8",
    )
    writer = FakeWriter()
    assert write_metrics_to_tensorboard(metrics, writer) == 1
    assert ("loss/q_loss", 1.5, 2) in writer.scalars
    assert ("system/steps_per_second", 0.5, 2) in writer.scalars
    assert ("system/cuda_peak_memory_gib", 1.0, 2) in writer.scalars


def test_legacy_conversion_creates_a_real_tensorboard_event(tmp_path):
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        '{"step": 1, "actor_loss": 0.25}\n', encoding="utf-8"
    )
    log_dir, count = convert_run_metrics(run_dir)
    assert count == 1
    assert any(path.name.startswith("events.out.tfevents") for path in log_dir.iterdir())
