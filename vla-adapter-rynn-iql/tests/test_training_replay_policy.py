from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vla_rynn_iql.data import REPLAY_POLICY, training_replay_policy
from vla_rynn_iql.rewards import reward_manifest_digest
from vla_rynn_iql.training import _restore_checkpoint, _save_checkpoint


@pytest.mark.parametrize("include", [True, False])
def test_checkpoint_records_selected_replay_policy(configured, tmp_path, monkeypatch, include):
    configured.raw["data"]["include_post_success"] = include
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
    assert json.loads((checkpoint / "checkpoint.json").read_text())["replay_policy"] == training_replay_policy(include)


@pytest.mark.parametrize("split,terminal,recorded,policy,reject", [
    ("train", 17, 22, None, True),
    ("train", 17, 22, REPLAY_POLICY, False),
    ("train", 17, 22, "confirmed_success_v1", True),
    ("train", None, 22, None, False),
    ("train", 21, 22, None, False),
    ("validation", 17, 22, None, False),
])
@pytest.mark.parametrize("include", [True, False])
def test_resume_checks_changed_training_sample_policy(
    configured, tmp_path, monkeypatch, split, terminal, recorded, policy, reject, include,
):
    configured.raw["data"]["include_post_success"] = include
    if split == "train" and terminal is not None and recorded > terminal + 1:
        reject = policy != training_replay_policy(include)
    rewards = {}
    metadata = {
        "dataset_sha256": "dataset", "reward_sha256": reward_manifest_digest(rewards),
        "base_checkpoint": configured.section("model")["base_checkpoint"],
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
