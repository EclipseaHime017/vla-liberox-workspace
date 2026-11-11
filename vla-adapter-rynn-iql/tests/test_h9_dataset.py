import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "liberox-vla-adapter-terminal"),
               str(Path(__file__).resolve().parents[2] / "liberox-vla-adapter-terminal/scripts")]
from backend.app.services.h9_reward_dataset import register_h9_dataset, ARMS
from backend.app.services.dataset_transfer import export_dataset
from backend.app.services.training_dataset_service import TrainingDatasetService, _stable_hash
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.rewards import materialize_reward_manifest, load_pinned_reward_index
from test_gamma_dataset import frozen


def inputs(configured, tmp_path, arm):
    path, version, _ = frozen(configured, tmp_path, "final")
    project = tmp_path / "data/projects/project"
    (project / "datasets").mkdir(parents=True)
    new = project / "datasets/original"
    path.parent.rename(new)
    path = new / "dataset.json"
    # Source fixture's version config/reward paths move with its directory.
    for key in ("config_path", "reward_manifest_path"):
        version[key] = str(new / "annotations/original-version" / Path(version[key]).name)
    ds = json.loads(path.read_text()); ds["project_id"] = "libero_x_vla"
    prepared_path = Path(version["prepared_manifest_path"])
    prepared = json.loads(prepared_path.read_text())
    by_id = {e["run_id"]: e for e in prepared["episodes"]}
    for member in ds["members"]:
        ep = by_id[member["run_id"]]
        member.update(resume_step=int(ep.get("resume_step") or 0), end_step=ep["recorded_action_count"])
    ds["dataset_sha256"] = _stable_hash(TrainingDatasetService._immutable_payload(ds))
    prepared["source_dataset_sha256"] = ds["dataset_sha256"]
    prepared_path.write_text(json.dumps(prepared))
    version.update(dataset_sha256=ds["dataset_sha256"], prepared_manifest_sha256=sha256_file(prepared_path))
    (new / "annotations/original-version/version.json").write_text(json.dumps(version))
    path.write_text(json.dumps(ds))
    a, kappa = ARMS[arm]
    configured.raw["reward"].update(gamma=.99, alpha=.3, base_coefficient=a,
        shaping_weight=kappa, final_normalization="none", zero_success_potential=True,
        accumulate_primitive_steps=False, stage_exponent=2)
    configured.raw["data"].update(include_post_success=False, bootstrap_failed_end=True)
    configured.raw["iql"]["beta"] = .3
    config_path = tmp_path / "numeric.yaml"
    config_path.write_text(yaml.safe_dump(configured.raw))
    numeric = load_train_config(config_path)
    rewards = materialize_reward_manifest(numeric)
    prepared = Path(numeric.raw["paths"]["work_dir"]) / "dataset_manifest.json"
    return dict(dataset_manifest=path, source_version_id="original-version", config_path=config_path,
        config_sha256=sha256_file(config_path), prepared_sha256=sha256_file(prepared),
        reward_sha256=sha256_file(rewards), arm=arm, name="H9 " + arm)


@pytest.mark.parametrize("arm", list(ARMS))
def test_h9_register_official_full_export_and_pinned_roundtrip(configured, tmp_path, monkeypatch, arm):
    args = inputs(configured, tmp_path, arm)
    protected = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr("vla_rynn_iql.rewards.materialize_reward_manifest",
                        lambda *a, **k: pytest.fail("Registration/export must not regenerate"))
    result = register_h9_dataset(**args)
    published = result["version"]
    config = load_train_config(Path(published["config_path"]))
    loaded = load_pinned_reward_index(config)
    assert loaded["reward_config"]["base_coefficient"] == ARMS[arm][0]
    assert loaded["reward_config"]["shaping_weight"] == ARMS[arm][1]
    assert config.raw["data"]["bootstrap_failed_end"] is True
    assert config.raw["data"]["include_post_success"] is False
    destination = tmp_path / "export"
    # Real public exporter, including copying recordings and Final snapshot binding.
    export_dataset(args["dataset_manifest"].parents[2], result["dataset_id"], destination,
                   Path(published["config_path"]), Path(__file__).resolve().parents[1])
    metadata = list(destination.rglob("trajectory_reward.final.json"))
    assert len(metadata) == 2
    for file in metadata:
        item = json.loads(file.read_text())
        assert item["reward_config"] == loaded["reward_config"]
        assert sha256_file(file.parent / item["values_file"]) == item["values_sha256"]
        assert item["stage_annotations_snapshot"]
    exported = json.loads((destination / "dataset.json").read_text())
    assert exported["members"] == json.loads(args["dataset_manifest"].read_text())["members"]
    assert exported["include_post_success"] is False
    assert all(p.read_bytes() == data for p, data in protected.items())
    if os.environ.get("H9_SERVER_ROOT"):
        server = Path(os.environ["H9_SERVER_ROOT"])
        run = subprocess.run([sys.executable, "-c", SERVER_IMPORT, str(server),
            str(destination), str(tmp_path / "server-import")], capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(server / "src")}, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr


def test_isolated_cli_uses_existing_runtime(configured, tmp_path):
    args = inputs(configured, tmp_path, "T3-4")
    root = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "isolated"
    for relative in ("liberox-vla-adapter-terminal/scripts/register_h9_dataset.py",
                     "liberox-vla-adapter-terminal/backend/app/services/h9_reward_dataset.py"):
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    command = [sys.executable, str(isolated / "liberox-vla-adapter-terminal/scripts/register_h9_dataset.py"),
               "--runtime-root", str(root), "--export-to", str(tmp_path / "export")]
    for key, value in args.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout.splitlines()[-1])["export_path"] == str(tmp_path / "export")


def test_actor_beta_is_not_a_reward_recipe_constraint(configured, tmp_path):
    args = inputs(configured, tmp_path, "T3-4")
    raw = yaml.safe_load(args["config_path"].read_text())
    manifest = Path(raw["paths"]["work_dir"]) / "rewards/reward_manifest.json"
    before = manifest.read_bytes()
    raw["iql"]["beta"] = .75
    args["config_path"].write_text(yaml.safe_dump(raw))
    args["config_sha256"] = sha256_file(args["config_path"])
    result = register_h9_dataset(**args)
    saved = yaml.safe_load(Path(result["version"]["config_path"]).read_text())
    assert saved["iql"]["beta"] == .75
    assert result["version"]["reward_config"] == json.loads(before)["reward_config"]
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("fault", ["hash", "split", "recipe", "w5", "mixed"])
def test_h9_rejects_before_publication(configured, tmp_path, fault):
    args = inputs(configured, tmp_path, "T3-4")
    raw = yaml.safe_load(args["config_path"].read_text())
    work = Path(raw["paths"]["work_dir"])
    if fault == "hash":
        args["reward_sha256"] = "0" * 64
    elif fault == "split":
        p = work / "dataset_manifest.json"
        value = json.loads(p.read_text()); value["episodes"][0]["split"] = "validation"
        p.write_text(json.dumps(value)); args["prepared_sha256"] = sha256_file(p)
    elif fault == "mixed":
        p = work / "rewards/reward_manifest.json"
        value = json.loads(p.read_text()); value["episodes"][0]["saved_reward_config"] = {"source": "final"}
        p.write_text(json.dumps(value)); args["reward_sha256"] = sha256_file(p)
    else:
        if fault == "recipe":
            raw["reward"]["base_coefficient"] = .7
        else:
            raw["data"]["bootstrap_failed_end"] = False
        args["config_path"].write_text(yaml.safe_dump(raw))
        args["config_sha256"] = sha256_file(args["config_path"])
    with pytest.raises(ValueError):
        register_h9_dataset(**args)
    assert len(list(args["dataset_manifest"].parents[1].glob("*/dataset.json"))) == 1


SERVER_IMPORT = r'''
import json, sys
from pathlib import Path
import numpy as np
import yaml
from vla_rynn_iql.config import LoadedConfig, load_train_config
from vla_rynn_iql.server_config import load_server_config
from vla_rynn_iql.server_pipeline import training_settings
from vla_rynn_iql.server_inputs import discover_tasks, prepare_global_inputs
from vla_rynn_iql.rewards import load_reward_index
from vla_rynn_iql.fusion_rewards import fused_reward_arrays
root, source, target = map(Path, sys.argv[1:])
target.mkdir()
settings = yaml.safe_load((root / 'configs/server_pipeline.yaml').read_text())
base = load_train_config(root / 'configs/training/iql.yaml').raw
base['reward'].update(source='final', base_coefficient=9., alpha=.99)
base_path = target / 'training.yaml'
base_path.write_text(yaml.safe_dump(base))
settings.update(training_config=str(base_path), runs_root=str(source),
                output_root=str(target / 'output'), cache_root=str(target / 'cache'))
settings['overrides']['reward'] = {'gamma': .99, 'zero_success_potential': True}
settings['overrides']['data'] = {'bootstrap_failed_end': True, 'include_post_success': False}
path = target / 'server.yaml'
path.write_text(yaml.safe_dump(settings))
server = load_server_config(path)
task, = discover_tasks(source, 'libero_x_vla')
raw = training_settings(server, task)
saved = json.loads(next(source.rglob('trajectory_reward.final.json')).read_text())['reward_config']
assert raw['reward']['base_coefficient'] == saved['base_coefficient']
assert raw['reward']['alpha'] == .3
assert raw['reward']['zero_success_potential'] is True
assert raw['data']['bootstrap_failed_end'] is True
assert raw['data']['include_post_success'] is False
raw = prepare_global_inputs(task, target / 'run', raw, 'final')
config = LoadedConfig(path, raw)
rewards = load_reward_index(config)
manifest = json.loads((Path(raw['paths']['work_dir']) / 'dataset_manifest.json').read_text())
for episode, entry in zip(manifest['episodes'], rewards['episodes']):
    assert entry['saved_reward_config'] == saved
    with np.load(entry['reward_path']) as values:
        arrays = {k: values[k] for k in values.files}
        expected = fused_reward_arrays(episode, saved, arrays)
        np.testing.assert_array_equal(values['final_reward'], expected['final_reward'])
        assert 'final_reward_scale' not in values
print('H9 server real import PASS')
'''
