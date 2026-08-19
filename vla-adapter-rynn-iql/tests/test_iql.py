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


def test_weighted_masked_l1_ignores_padding():
    prediction = torch.tensor([[[1.0], [100.0]]])
    target = torch.zeros_like(prediction)
    loss = weighted_masked_l1(prediction, target, torch.tensor([[1, 0]]), torch.tensor([2.0]))
    assert loss.item() == 2.0


def test_advantage_weight_is_capped():
    weights = advantage_weights(torch.tensor([-1.0, 10.0]), beta=10.0, maximum=100.0)
    assert weights[1].item() == 100.0


def test_bellman_target_discounts_by_actual_chunk_length():
    target = chunk_bellman_target(
        torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]),
        torch.tensor([3, 8]), torch.tensor([1.0, 0.0]), 0.9,
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
