from __future__ import annotations

import torch

from vla_rynn_iql.iql import (
    PixelIQL, advantage_weights, chunk_bellman_target, expectile_loss,
    weighted_masked_l1,
)


def _batch(batch_size=2):
    return {
        "pixels": torch.zeros(batch_size, 6, 32, 32, dtype=torch.uint8),
        "next_pixels": torch.zeros(batch_size, 6, 32, 32, dtype=torch.uint8),
        "proprio": torch.zeros(batch_size, 8), "next_proprio": torch.zeros(batch_size, 8),
        "actions": torch.zeros(batch_size, 8, 7), "action_mask": torch.ones(batch_size, 8),
        "reward": torch.tensor([1.0, -1.0]), "bootstrap_mask": torch.tensor([0.0, 1.0]),
        "chunk_length": torch.tensor([3, 8]),
    }


def test_iql_update_and_advantage_are_finite():
    model = PixelIQL()
    metrics = model.update(_batch())
    assert torch.isfinite(torch.tensor([metrics.q_loss, metrics.value_loss])).all()
    assert model.advantage(_batch()).shape == (2,)
    assert isinstance(model.q_optimizer, torch.optim.AdamW)
    assert isinstance(model.value_optimizer, torch.optim.AdamW)
    assert model.q_optimizer.param_groups[0]["weight_decay"] == 0.01
    assert model.value_optimizer.param_groups[0]["weight_decay"] == 0.01


def test_iql_optimizer_and_gradient_clip_are_configurable():
    model = PixelIQL(
        critic_optimizer="adamw", critic_weight_decay=1e-4,
        value_optimizer="adamw", value_weight_decay=2e-4,
        critic_max_grad_norm=7.0, value_max_grad_norm=8.0,
    )
    assert isinstance(model.q_optimizer, torch.optim.AdamW)
    assert isinstance(model.value_optimizer, torch.optim.AdamW)
    assert model.q_optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert model.value_optimizer.param_groups[0]["weight_decay"] == 2e-4
    assert model.critic_max_grad_norm == 7.0
    assert model.value_max_grad_norm == 8.0


def test_weighted_masked_l1_ignores_padding():
    prediction = torch.tensor([[[1.0], [100.0]]])
    target = torch.zeros_like(prediction)
    loss = weighted_masked_l1(prediction, target, torch.tensor([[1, 0]]), torch.tensor([2.0]))
    assert loss.item() == 2.0


def test_advantage_weight_is_capped():
    weights = advantage_weights(torch.tensor([-1.0, 10.0]), beta=10.0, maximum=100.0)
    assert weights[1].item() == 100.0


def test_bellman_target_discounts_once_per_macro_action():
    target = chunk_bellman_target(
        torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]),
        torch.tensor([3, 8]), torch.tensor([1.0, 0.0]), 0.9,
    )
    torch.testing.assert_close(target, torch.tensor([1.0 + 2.0 * 0.9, 1.0]))


def test_bellman_target_uses_actual_duration_for_accumulated_step_rewards():
    target = chunk_bellman_target(
        torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]),
        torch.tensor([3, 8]), torch.tensor([1.0, 0.0]), 0.9,
        accumulate_primitive_steps=True,
    )
    torch.testing.assert_close(target, torch.tensor([1.0 + 2.0 * 0.9**3, 1.0]))


def test_iql_checkpoint_restores_models_and_optimizers():
    torch.manual_seed(3)
    source = PixelIQL()
    source.update(_batch())
    restored = PixelIQL()
    restored.restore(source.checkpoint())
    for left, right in zip(source.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(left, right)


def test_policy_advantage_uses_online_q_by_default():
    model = PixelIQL()
    with torch.no_grad():
        for parameter in model.q1.parameters():
            parameter.zero_()
        for parameter in model.q2.parameters():
            parameter.zero_()
        for parameter in model.target_q1.parameters():
            parameter.zero_()
        for parameter in model.target_q2.parameters():
            parameter.zero_()
        model.q1.head[-1].bias.fill_(2.0)
        model.q2.head[-1].bias.fill_(3.0)
        model.target_q1.head[-1].bias.fill_(20.0)
        model.target_q2.head[-1].bias.fill_(30.0)
        for parameter in model.value.parameters():
            parameter.zero_()
    torch.testing.assert_close(model.advantage(_batch()), torch.full((2,), 2.0))
    torch.testing.assert_close(
        model.advantage(_batch(), target=True), torch.full((2,), 20.0)
    )
