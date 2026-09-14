from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vla_rynn_iql.data import REPLAY_POLICY
from vla_rynn_iql.rewards import reward_manifest_digest
from vla_rynn_iql.training import _restore_checkpoint, _save_checkpoint


def test_checkpoint_records_full_replay_policy(configured, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    components = SimpleNamespace(
        action_head=torch.nn.Linear(1, 1), proprio_projector=torch.nn.Linear(1, 1),
        stats_key="test",
    )
    optimizer = torch.optim.Adam(components.action_head.parameters())
    checkpoint = _save_checkpoint(
        tmp_path, 1, components, SimpleNamespace(checkpoint=lambda: {}), optimizer,
        torch.Generator(), configured, {"dataset_sha256": "dataset"}, {},
    )
    assert json.loads((checkpoint / "checkpoint.json").read_text())["replay_policy"] == REPLAY_POLICY


@pytest.mark.parametrize("split,terminal,recorded,policy,reject", [
    ("train", 17, 22, None, True),
    ("train", 17, 22, REPLAY_POLICY, False),
    ("train", None, 22, None, False),
    ("train", 21, 22, None, False),
    ("validation", 17, 22, None, False),
])
def test_resume_checks_changed_training_sample_policy(
    configured, tmp_path, monkeypatch, split, terminal, recorded, policy, reject,
):
    rewards = {}
    metadata = {
        "dataset_sha256": "dataset", "reward_sha256": reward_manifest_digest(rewards),
        "base_checkpoint": configured.section("vla")["base_checkpoint"],
        "stats_key": "test", "replay_policy": policy,
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata))
    manifest = {"dataset_sha256": "dataset", "episodes": [{
        "split": split, "terminal_step": terminal, "recorded_action_count": recorded,
    }]}
    components = SimpleNamespace(action_head=torch.nn.Linear(1, 1), stats_key="test")

    class WeightsReached(Exception):
        pass

    def load_weights(*args, **kwargs):
        raise WeightsReached

    monkeypatch.setattr(torch, "load", load_weights)
    expected = pytest.raises(ValueError, match="Start a new run") if reject else pytest.raises(WeightsReached)
    with expected:
        _restore_checkpoint(tmp_path, components, None, None, None,
                            torch.device("cpu"), configured, manifest, rewards)
