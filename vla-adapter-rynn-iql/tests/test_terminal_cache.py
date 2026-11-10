from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index
from vla_rynn_iql.server_replay import build_replay_cache, CachedReplayDataset
from test_replay import FakeAnnotator, _stats


def test_bootstrap_switch_has_distinct_cache_identity(configured, tmp_path):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    reward = load_reward_index(configured)
    old, _ = build_replay_cache(configured, tmp_path / "cache", _stats(7), _stats(8), reward)
    configured.raw["data"]["bootstrap_failed_end"] = True
    new, hit = build_replay_cache(configured, tmp_path / "cache", _stats(7), _stats(8), reward)
    assert old != new and not hit
    before, after = CachedReplayDataset(old), CachedReplayDataset(new)
    differences = [i for i in range(len(before)) if before[i]["bootstrap_mask"] != after[i]["bootstrap_mask"]]
    assert len(differences) == 1
    assert after[differences[0]]["run_id"] == "root"


def test_server_overrides_preserve_endpoint_switches(configured, tmp_path):
    import yaml
    from test_server_inputs import global_runs
    from test_server_pipeline import configuration
    from vla_rynn_iql.server_config import load_server_config
    from vla_rynn_iql.server_pipeline import training_settings
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    raw = yaml.safe_load(server.path.read_text())
    raw["overrides"].setdefault("data", {})["bootstrap_failed_end"] = True
    raw["overrides"].setdefault("reward", {})["zero_success_potential"] = True
    server.path.write_text(yaml.safe_dump(raw))
    settings = training_settings(load_server_config(server.path), dataset)
    assert settings["data"]["bootstrap_failed_end"] is True
    assert settings["reward"]["zero_success_potential"] is True
