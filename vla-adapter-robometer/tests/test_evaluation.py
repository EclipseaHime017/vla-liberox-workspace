from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_adapter_robometer.config import DEFAULT_CONFIG, load_config
from vla_adapter_robometer.evaluation import evaluate_selection, evaluation_steps, prefix_indices


def test_sampling_includes_first_and_last():
    np.testing.assert_array_equal(evaluation_steps(501, 20, 3), np.r_[evaluation_steps(501, 20, 3)[:-1], 500])
    assert evaluation_steps(501, 20, 3)[0] == 0
    assert evaluation_steps(501, 20, 3)[-1] == 500
    np.testing.assert_array_equal(prefix_indices(3), [0, 1, 2, 3])


def test_config_is_strict(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["model"]["mystery"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown config.model"):
        load_config(path)


def test_complete_trajectory_is_evaluated(tmp_path: Path):
    episode = tmp_path / "episode"
    episode.mkdir()
    trajectory = episode / "trajectory.npz"
    observations = episode / "trajectory_observations.npz"
    manifest = episode / "run.json"
    np.savez_compressed(trajectory, time_seconds=np.arange(11) / 20, done=np.r_[False, True, np.ones(8, bool)])
    np.savez_compressed(observations, agentview_image=np.zeros((11, 4, 4, 3), np.uint8))
    manifest.write_text(json.dumps({"task": "do task"}))
    def artifact(path):
        return {"path": str(path), "sha256": "unused", "size": path.stat().st_size}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "dataset_sha256": "selection", "members": [{"run_id": "run", "artifacts": {
            "trajectory": artifact(trajectory), "observations": artifact(observations),
            "manifest": artifact(manifest),
        }}],
    }))
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"].update({"selection_manifest": str(selection), "output_dir": str(tmp_path / "out"), "robometer_root": str(tmp_path)})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    loads = []
    class Fake:
        commit = "fake"
        load_seconds = 0
        def __init__(self, _): loads.append(1)
        def __call__(self, frames, steps, prompt):
            assert len(frames) == 11 and steps[-1] == 10 and prompt == "do task"
            return np.linspace(0, 1, len(steps)), np.linspace(0, 1, len(steps))
    result = evaluate_selection(load_config(config_path), Fake)
    payload = json.loads(result.read_text())
    with np.load(payload["episodes"][0]["annotation_path"]) as values:
        assert values["observation_steps"][-1] == 10
    assert len(loads) == 1
    # A new dataset version or execution batch size must not load the model.
    shutil.copytree(tmp_path / "out" / "values", tmp_path / "new" / "values")
    raw["paths"]["output_dir"] = str(tmp_path / "new")
    raw["evaluation"]["batch_size"] = 1
    config_path.write_text(yaml.safe_dump(raw))
    cached = json.loads(evaluate_selection(load_config(config_path), Fake).read_text())
    assert len(loads) == 1
    assert cached["cache_stats"]["skipped"] == 1
    assert Path(cached["episodes"][0]["annotation_path"]).parent == tmp_path / "new" / "values"
    evaluate_selection(load_config(config_path), Fake, overwrite=True)
    assert len(loads) == 2
    raw["evaluation"]["fps"] = 4.0
    config_path.write_text(yaml.safe_dump(raw))
    evaluate_selection(load_config(config_path), Fake)
    assert len(loads) == 3
