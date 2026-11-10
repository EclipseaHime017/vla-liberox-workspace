import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_critic_diagnostics import diagnostic_inputs, one_cpu_thread
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql.critic_tail_values import export_tail_values
from vla_rynn_iql.data import load_manifest
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.iql import ValueNetwork
from vla_rynn_iql.models import replay_stats
from vla_rynn_iql.replay import ActionDataset
from vla_rynn_iql.rewards import reward_derivation_config


@pytest.fixture
def frozen(configured, tmp_path):
    observations = next(Path(configured.raw["paths"]["dataset_sources"][0]).glob("**/root/episodes/episode_000/trajectory_observations.npz"))
    with np.load(observations) as source:
        arrays = {k: source[k] for k in source.files}
    arrays["agentview_image"][-1] = 255
    arrays["wrist_image"][-1] = 128
    np.savez_compressed(observations, **arrays)
    config, _ = diagnostic_inputs.__wrapped__(configured, tmp_path)
    manifest = load_manifest(config)
    recipe = reward_derivation_config(config.raw["reward"])
    result = tmp_path / "frozen"
    result.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        value = ValueNetwork(manifest["proprio_dim"]).eval()
    torch.save({"schema_version": 1, "kind": "iql_critic_diagnostic", "horizon": manifest["action_horizon"],
        "action_dim": manifest["action_dim"], "proprio_dim": manifest["proprio_dim"], "seed": 17,
        "iql_config": config.raw["iql"], "reward_config": recipe,
        "critic": {"model": {"value." + k: v for k, v in value.state_dict().items()}}}, result / "critic.pt")
    (result / "summary.json").write_text(json.dumps({"kind": "iql_critic_diagnostic",
        "training_config_sha256": config.digest, "dataset_sha256": manifest["dataset_sha256"],
        "reward_config": recipe, "reward_manifest_sha256": config.raw["reward"]["manifest_sha256"],
        "model_config": config.raw["model"], "model_contract": model_contract(config.raw["model"]),
        "critic_horizon": manifest["action_horizon"], "seed": 17}))
    replay = ActionDataset(config, *replay_stats(config.raw["model"]), "train")
    replay.image_size = config.raw["iql"]["critic_image_size"]
    rows, last = [], None
    with torch.no_grad():
        for i in range(len(replay)):
            sample = replay._sample(i, transitions=True)
            rows.append(dict(split="train", run_id=sample["run_id"], start=sample["start"],
                end=sample["start"] + int(sample["chunk_length"]),
                v=float(value(sample["pixels"][None], sample["proprio"][None]))))
            if sample["run_id"] == "root" and rows[-1]["end"] == 17:
                last = float(value(sample["next_pixels"][None], sample["next_proprio"][None]))
    with (result / "qv.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return config, result, last


def test_real_frozen_endpoint_export_and_w1_consumption(frozen, tmp_path, monkeypatch):
    config, result, expected = frozen
    from vla_rynn_iql import iql, models, rewards
    def forbidden(*args, **kwargs):
        raise AssertionError("No actor, critic training or reward regeneration")
    monkeypatch.setattr(iql.PixelIQL, "update", forbidden)
    monkeypatch.setattr(models, "model_backend", forbidden)
    monkeypatch.setattr(rewards, "materialize_reward_manifest", forbidden)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    rng = torch.random.get_rng_state().clone()
    payload = export_tail_values(config, result, tmp_path / "tail.json", critic_sha256=sha256_file(result / "critic.pt"))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert before == {p: sha256_file(p) for p in before}
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["observation_step"] == 17 and row["last_chunk_start"] == 16
    assert row["tail_v"] == pytest.approx(expected)
    assert not np.isclose(row["tail_v"], row["reference_start_v"], atol=1e-7)
    script = Path(__file__).resolve().parents[1] / "scripts/screen_rewards.py"
    spec = importlib.util.spec_from_file_location("screen_tail_test", script)
    screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
    dataset = Path(config.raw["paths"]["work_dir"]) / "dataset_manifest.json"
    reward = Path(config.raw["reward"]["manifest_path"])
    options = dict(gamma=.8, split="all", bootstrap_failures=True, tail_values=payload,
                   tail_critic_sha256=payload["critic_sha256"])
    report = screen.analyze(dataset, reward, **options)
    assert next(r for r in report["inputs"] if r["run_id"] == "root")["tail_value"] == expected
    for field, bad in (("gamma", .9), ("dataset_manifest_sha256", "a" * 64),
                       ("critic_sha256", "a" * 64), ("split", "validation"),
                       ("primitive_discount", True), ("reward_manifest_sha256", "a" * 64)):
        changed = copy.deepcopy(payload); changed[field] = bad
        with pytest.raises(ValueError):
            screen.analyze(dataset, reward, **(options | {"tail_values": changed}))
    for field, bad in (("observation_step", 16), ("observations_sha256", "a" * 64), ("tail_v", float("nan"))):
        changed = copy.deepcopy(payload); changed["rows"][0][field] = bad
        with pytest.raises(ValueError):
            screen.analyze(dataset, reward, **(options | {"tail_values": changed}))
    with pytest.raises(FileExistsError):
        export_tail_values(config, result, tmp_path / "tail.json", critic_sha256=payload["critic_sha256"])


@pytest.mark.parametrize("failure", ["checksum", "summary", "csv", "source"])
def test_frozen_identity_rejected(frozen, tmp_path, failure):
    config, result, _ = frozen
    digest = sha256_file(result / "critic.pt")
    if failure == "checksum":
        digest = "0" * 64
    elif failure == "summary":
        p = result / "summary.json"; data = json.loads(p.read_text()); data["dataset_sha256"] = "wrong"
        p.write_text(json.dumps(data))
    elif failure == "csv":
        p = result / "qv.csv"; rows = list(csv.DictReader(p.open()))
        for r in rows: r["v"] = "999"
        with p.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    else:
        p = Path(next(e for e in load_manifest(config)["episodes"] if e["terminal_step"] is None)["observations_path"])
        p.write_bytes(p.read_bytes() + b"modified")
    with pytest.raises(ValueError):
        export_tail_values(config, result, tmp_path / "tail.json", critic_sha256=digest)
    assert not (tmp_path / "tail.json").exists()
