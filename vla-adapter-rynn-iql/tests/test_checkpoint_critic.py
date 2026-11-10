import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vla_rynn_iql.checkpoint_critic import export_critic
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql.data import prepare_dataset, training_replay_policy
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.iql import QNetwork, ValueNetwork
from vla_rynn_iql.model_artifacts import parent_snapshot
from vla_rynn_iql.models import model_signature
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index, reward_manifest_digest
from vla_rynn_iql.server_replay import build_replay_cache
from test_replay import FakeAnnotator, _stats


@pytest.fixture
def case(configured, tmp_path):
    config = configured
    config.raw["training"].update(method="iql", train_steps=10000)
    config.raw["iql"].update(beta=.3, max_advantage_weight=20., critic_image_size=16)
    config.raw["data"].update(bootstrap_failed_end=True)
    config.raw["reward"].update(zero_success_potential=True, gamma=.99)
    manifest_path = prepare_dataset(config).manifest
    manifest = json.loads(manifest_path.read_text())
    annotate_manifest(config, FakeAnnotator())
    reward = load_reward_index(config)
    reward_path = tmp_path / "reward.json"
    reward_path.write_text(json.dumps(reward))
    cache, _ = build_replay_cache(config, tmp_path / "cache", _stats(7), _stats(8), reward)
    checkpoint = tmp_path / "step_00010000"
    checkpoint.mkdir()
    contract = model_contract(config.section("model"))
    parent = parent_snapshot(config.section("model"), "a" * 64)
    metadata = {"schema_version": 2, "step": 10000, "config_sha256": config.digest,
        "algorithm": "iql", "model_config": model_signature(config.raw), "model_contract": contract,
        "model_parent": parent, "model_identity": {"test": "frozen"},
        "base_checkpoint": config.section("model")["base_checkpoint"], "stats_key": parent["stats_key"],
        "dataset_sha256": manifest["dataset_sha256"], "reward_sha256": reward_manifest_digest(reward),
        "training_layout": {"chunking": manifest["chunking"], "actor_horizon": contract["io"]["replay_horizon"],
                            "transition_horizon": manifest["action_horizon"]},
        "replay_policy": training_replay_policy(True), "bootstrap_failed_end": True}
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    (checkpoint / "effective_config.yaml").write_text(yaml.safe_dump(config.raw))
    dims = (manifest["action_horizon"], manifest["action_dim"], manifest["proprio_dim"])
    modules = {"q1": QNetwork(*dims), "q2": QNetwork(*dims), "value": ValueNetwork(dims[2])}
    for name, module in modules.items():
        for p in module.parameters(): p.data.zero_()
        module.head[-1].bias.data.fill_({"q1": 2., "q2": 3., "value": 1.}[name])
    state = {f"{name}.{key}": value for name, module in modules.items() for key, value in module.state_dict().items()}
    torch.save({"schema_version": 2, "step": 10000, "algorithm": "iql",
                "algorithm_state": {"model": state, "q_optimizer": "never restore", "value_optimizer": "never restore"},
                "actor_optimizer": "never restore"}, checkpoint / "trainer.pt")
    return checkpoint, manifest_path, reward_path, cache, tmp_path / "out.json"


def run(case):
    return export_critic(*case, expected_trainer_sha256=sha256_file(case[0] / "trainer.pt"), batch_size=2)


def test_real_cache_and_final_critic_read_only_exact_actual_weights(case, monkeypatch):
    def forbidden(*a, **kw): pytest.fail("Must not create actor, optimizer, or training state")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr("vla_rynn_iql.models.model_backend", forbidden)
    before = {p: sha256_file(p) for p in case[0].parent.rglob("*") if p.is_file()}
    state = torch.random.get_rng_state().clone()
    result = run(case)
    assert torch.equal(state, torch.random.get_rng_state())
    assert result["seed"] is not None and result["step"] == 10000
    assert result["deduplicated"] and result["split"] == "train" and result["beta"] == .3
    assert result["groups"]["failed_parent"]["count"] > 0
    assert result["groups"]["human_pre_success"]["count"] > 0
    for row in result["rows"]:
        assert (row["q1"], row["q2"], row["v"], row["advantage"]) == (2., 3., 1., 1.)
        assert row["weight"] == float(torch.exp(torch.tensor(.3)))
    assert result["d2_second_condition"] is False
    assert all(sha256_file(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError): run(case)


@pytest.mark.parametrize("failure", ["sha", "intermediate", "reward", "member", "cache_data", "beta"])
def test_fail_closed_before_output(case, failure):
    checkpoint, manifest_path, reward_path, cache, output = case
    if failure == "sha":
        with pytest.raises(ValueError, match="SHA"):
            export_critic(*case, expected_trainer_sha256="0" * 64)
        return
    if failure == "intermediate":
        p = checkpoint / "checkpoint.json"; meta = json.loads(p.read_text()); meta["step"] = 9999; p.write_text(json.dumps(meta))
    elif failure == "reward":
        reward_path.write_text(json.dumps({"dataset_sha256": "wrong"}))
    elif failure == "member":
        p = cache / "cache.json"; meta = json.loads(p.read_text()); meta["records"][0]["run_id"] = "wrong"; p.write_text(json.dumps(meta))
    elif failure == "cache_data":
        with (cache / "pixels.npy").open("r+b") as f: f.seek(-1, 2); f.write(b"x")
    else:
        p = checkpoint / "effective_config.yaml"; raw = yaml.safe_load(p.read_text()); raw["iql"]["beta"] = 3.; p.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, KeyError)):
        run(case)
    assert not output.exists()
