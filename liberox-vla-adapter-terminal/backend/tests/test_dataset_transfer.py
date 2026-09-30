import json
from pathlib import Path

import yaml

from backend.app.services.dataset_transfer import export_dataset
from test_global_reward_inheritance import setup_recording, native_rynn
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.portable_dataset import read_bundle, materialize_bundle
from vla_rynn_iql.rewards import load_reward_index
from vla_rynn_iql.io import sha256_file


def test_headless_export_legacy_and_global_labels(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    native_rynn(jobs, dataset, run)
    # Robometer is diagnostic-only and copied independently, never read as IQL reward.
    episode = Path(run["trajectory"]).parent
    (episode / "robometer_evaluation.json").write_text('{"fixture": "independent output"}')
    protected = {p: sha256_file(p) for p in Path(run["output_dir"]).rglob("*") if p.is_file()}
    config = tmp_path / "export-config.yaml"
    raw = jobs._load_base_config()
    raw["data"]["project_id"] = "test"
    config.write_text(yaml.safe_dump(raw))
    destination = tmp_path / "transfer/task/test"
    export_dataset(jobs.datasets.root.parent, dataset["id"], destination, config,
                   jobs.ui_config.offline_rl_root, required_rewards=("rynnvalue", "stage"))
    bundle = read_bundle(destination, verify=True)
    assert {"rynnvalue", "stage", "sparse"} <= set(bundle["rewards"])
    assert (destination / "runs/run/episodes/episode_000/robometer_evaluation.json").read_bytes() == (
        episode / "robometer_evaluation.json").read_bytes()
    assert protected == {p: sha256_file(p) for p in Path(run["output_dir"]).rglob("*") if p.is_file()}
    # No GUI, model evaluator or original source path is needed by the server.
    Path(run["output_dir"]).rename(tmp_path / "offline")
    migrated = materialize_bundle(destination, tmp_path / "server/inputs", raw, "rynnvalue")
    migrated["reward"]["gamma"] = bundle["rewards"]["rynnvalue"]["reward"]["gamma"]
    result = load_reward_index(LoadedConfig(config, migrated))
    assert len(result["episodes"]) == 1
