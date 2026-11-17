"""CPU-only generation/export contract, optionally verified by a server checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from studio.api.models import DatasetAnnotationRequest
from studio.workers.finalize_reward_version import seal
from test_dataset_transfer import export, local_stage
from test_global_reward_inheritance import setup_recording, native_rynn, digest
from core.config import load_train_config
from core.data import prepare_dataset
from rewards.fusion import import_saved_model_inputs
from rewards.manifest import materialize_reward_manifest


@pytest.mark.parametrize("value", [-1, True, "1", float("inf"), float("nan")])
def test_invalid_base_coefficient_rejected_by_api_or_service(tmp_path, monkeypatch, value):
    jobs, dataset, _ = setup_recording(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        request = DatasetAnnotationRequest(source="final", base_coefficient=value)
        jobs.start_annotation(dataset["id"], **request.model_dump(exclude_none=True))


@pytest.mark.parametrize("coefficient", [None, .6, 1.])
@pytest.mark.parametrize("alpha", [.25, .75])
def test_generation_export_and_optional_server_import(tmp_path, tmp_path_factory, monkeypatch, coefficient, alpha):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    # No model is loaded: this test covers reward/training configuration only.
    from studio.services.offline_job_service import core_module
    monkeypatch.setattr(core_module("model_assets"),
                        "pin_model", lambda settings: settings)
    native_rynn(jobs, dataset, run)
    local_stage(jobs, dataset)
    defaults = jobs.reward_configuration(dataset["id"])["final"]
    assert defaults["base_coefficient"] is None
    assert defaults["final_normalization"] == "initial_chunk_v1"
    request = DatasetAnnotationRequest(source="final", base_coefficient=coefficient, alpha=alpha,
                                       shaping_weight=.1, final_normalization="none")
    job = jobs.start_annotation(dataset["id"], **request.model_dump(exclude_none=True))
    config = load_train_config(job["config_path"])
    assert config.raw["reward"].get("base_coefficient") == coefficient
    assert config.raw["reward"]["final_normalization"] == "none"
    prepare_dataset(config)
    import_saved_model_inputs(config, job["output_path"] / "model_inputs.json")
    materialize_reward_manifest(config)
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    training = jobs.start_training(dataset["id"], {"reward_source": "final"})
    assert load_train_config(training["config_path"]).raw["reward"].get("base_coefficient") == coefficient
    with pytest.raises(ValueError):
        jobs.start_training(dataset["id"], {"reward_source": "final", "reward_base_coefficient": .5})
    destination = tmp_path_factory.mktemp("exports") / "dataset"
    export(jobs, dataset, destination)
    metadata_path = next(destination.rglob("trajectory_reward.final.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["reward_config"].get("base_coefficient") == coefficient
    assert metadata["reward_config"].get("final_normalization", "none") == "none"
    protected = {p: digest(p) for p in destination.rglob("*") if p.is_file()}
    with np.load(metadata_path.parent / metadata["values_file"]) as arrays:
        a = 1 - alpha if coefficient is None else coefficient
        np.testing.assert_allclose(arrays["final_reward"], a * arrays["sparse_reward"] +
                                   alpha * arrays["stage_chunk_reward"] + .1 * arrays["pbrs_shaping_reward"], atol=1e-6)
        assert "final_reward_scale" not in arrays
    # Bind the real published snapshot while the platform base has conflicting
    # new settings. A legacy recipe must retain its own implicit (1-alpha).
    from studio.services.trajectory_reward_snapshot import bind_reward_snapshot
    from studio.services.global_reward_binding import bind_global_rewards
    bind_reward_snapshot(Path(version["prepared_manifest_path"]), Path(version["reward_manifest_path"]))
    old_load = jobs._load_base_config
    def changed_base(*args, **kwargs):
        raw = old_load(*args, **kwargs)
        raw["reward"].update(source="final", base_coefficient=9., alpha=.99)
        return raw
    jobs._load_base_config = changed_base
    bound = bind_global_rewards(jobs, dataset, "final")
    bound_config = load_train_config(Path(bound["config_path"]))
    assert bound_config.raw["reward"].get("base_coefficient") == coefficient
    assert bound_config.raw["reward"]["alpha"] == alpha
    server_root = os.environ.get("W6_SERVER_ROOT")
    if server_root:
        result = subprocess.run([sys.executable, "-c", SERVER_IMPORT, server_root, str(destination),
                                 str(tmp_path / "server")], text=True, capture_output=True,
                                env={**os.environ, "PYTHONPATH": str(Path(server_root) / "src")})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "server import/gamma/success-potential PASS" in result.stdout
    assert protected == {p: digest(p) for p in destination.rglob("*") if p.is_file()}


SERVER_IMPORT = r'''
import json, sys
from pathlib import Path
import numpy as np
import yaml
from core.config import LoadedConfig, validate_train_config, load_train_config
from cluster.config import load_server_config
from cluster.pipeline import training_settings
from cluster.inputs import discover_tasks, prepare_global_inputs
from rewards.manifest import load_reward_index
from rewards.fusion import fused_reward_arrays
root, source, target = map(Path, sys.argv[1:])
target.mkdir()
settings = yaml.safe_load((root / 'configs/server_pipeline.yaml').read_text())
base = load_train_config(root / 'configs/methods/iql.yaml').raw
base['reward'].update(source='final', base_coefficient=9., alpha=.99)
base_path = target / 'training.yaml'
base_path.write_text(yaml.safe_dump(base))
settings.update(training_config=str(base_path), runs_root=str(source),
                output_root=str(target / 'output'), cache_root=str(target / 'cache'))
settings['overrides']['reward'] = {'gamma': .92}
path = target / 'server.yaml'
path.write_text(yaml.safe_dump(settings))
server = load_server_config(path)
task, = discover_tasks(source, 'test')
raw = training_settings(server, task)
saved = json.loads(next(source.rglob('trajectory_reward.final.json')).read_text())['reward_config']
assert raw['reward'].get('base_coefficient') == saved.get('base_coefficient')
assert raw['reward']['alpha'] == saved['alpha']
assert raw['reward']['final_normalization'] == 'none'
# Project ID is test fixture metadata, not a production server override.
raw['data']['project_id'] = 'test'
for zero in (False, True):
    local = json.loads(json.dumps(raw))
    local['reward']['zero_success_potential'] = zero
    local = validate_train_config(local, path).raw
    local = prepare_global_inputs(task, target / f'run-{zero}', local, 'final')
    config = LoadedConfig(path, local)
    rewards = load_reward_index(config)
    manifest = json.loads((Path(local['paths']['work_dir']) / 'dataset_manifest.json').read_text())
    for episode, entry in zip(manifest['episodes'], rewards['episodes']):
        assert entry['saved_reward_config'].get('base_coefficient') == saved.get('base_coefficient')
        recipe = {**entry['saved_reward_config'], 'gamma': .92, 'zero_success_potential': zero}
        with np.load(entry['reward_path']) as values:
            signals = {k: values[k] for k in values.files}
            expected = fused_reward_arrays(episode, recipe, signals)
            np.testing.assert_array_equal(values['final_reward'], expected['final_reward'])
            wrong = fused_reward_arrays(episode, {**recipe, 'base_coefficient': 9.}, signals)
            assert not np.array_equal(values['final_reward'], wrong['final_reward'])
            assert 'final_reward_scale' not in values
print('server import/gamma/success-potential PASS')
'''
