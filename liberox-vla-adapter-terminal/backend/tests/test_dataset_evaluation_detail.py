import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.services.dataset_evaluation_detail import attach_dataset_context


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    times = np.arange(21, dtype=float) / 20
    result = {"run": {"id": "r", "task_id": "task"}, "series": {"time_seconds": times.tolist()},
              "rynnvalue_evaluation": {"global": True}, "evaluation": {"global": True},
              "robometer_evaluation": None}
    datasets, versions = {}, {}
    for dataset_id, power in (("a", 2), ("b", 4)):
        directory = tmp_path / dataset_id
        directory.mkdir()
        scores = -1 + (np.arange(21) / 20) ** power
        arrays = directory / "reward.npz"
        np.savez_compressed(arrays, boundary_steps=[0, 8, 16, 20],
                            final_reward=scores[[8, 16, 20]], stage_score=scores,
                            observation_steps=np.arange(21), time_seconds=times)
        recipe = {"source": "stage", "stage_exponent": power, "gamma": .99,
                  "accumulate_primitive_steps": False}
        manifest_path = directory / "reward_manifest.json"
        manifest_path.write_text(json.dumps({"complete": True, "reward_config": recipe,
            "episodes": [{"run_id": "r", "reward_path": str(arrays), "reward_sha256": digest(arrays)}]}))
        version = {"id": "v1", "evaluator": "stage", "parameters": recipe, "status": "READY",
                   "reward_manifest_path": str(manifest_path), "reward_manifest_sha256": digest(manifest_path)}
        versions[(dataset_id, "v1")] = version
        datasets[dataset_id] = {"id": dataset_id, "name": dataset_id, "members": [{"run_id": "r"}],
                                "reward_version_id": "v1", "evaluation_versions": [version]}
    service = SimpleNamespace(list=lambda _: list(datasets.values()),
                              get=lambda key, **_: datasets[key],
                              get_version=lambda key, version: versions[(key, version)])
    return result, service, datasets, versions


def test_same_trajectory_has_independent_dataset_curves(tmp_path):
    result, service, _, _ = fixture(tmp_path)
    a = attach_dataset_context(copy.deepcopy(result), service, "a", "v1")
    b = attach_dataset_context(copy.deepcopy(result), service, "b", "v1")
    assert a["reward_evaluation"]["stage_scores"][10] != b["reward_evaluation"]["stage_scores"][10]
    assert a["dataset_context"]["config"]["stage_exponent"] == 2
    assert b["dataset_context"]["config"]["stage_exponent"] == 4
    assert len(a["reward_evaluation"]["stage_scores"]) == 21
    assert a["reward_evaluation"]["chunk_end_steps"][-1] == 20
    assert a["rynnvalue_evaluation"] == {"global": True}
    assert a["evaluation_sources"]["rynnvalue"]["origin"] == "global"
    assert a["evaluation_sources"]["stage"]["origin"] == "dataset"
    assert len(a["available_dataset_contexts"]) == 2
    global_detail = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert global_detail["rynnvalue_evaluation"] == {"global": True}


def test_detail_never_recalculates_and_rejects_corrupted_artifacts(tmp_path, monkeypatch):
    result, service, _, versions = fixture(tmp_path)
    first = attach_dataset_context(copy.deepcopy(result), service, "a", "v1")
    # Once published, plotted values come from the file even if formula code changes.
    import backend.app.services.stage_annotation_service as stage
    monkeypatch.setattr(stage.StageAnnotationService, "_module", lambda _: pytest.fail("must not evaluate"))
    assert attach_dataset_context(copy.deepcopy(result), service, "a", "v1") == first
    manifest = json.loads(Path(versions[("a", "v1")]["reward_manifest_path"]).read_text())
    Path(manifest["episodes"][0]["reward_path"]).write_bytes(b"corrupt")
    broken = attach_dataset_context(copy.deepcopy(result), service, "a", "v1")
    assert "hash mismatch" in broken["evaluation_sources"]["stage"]["error"]
    assert broken["rynnvalue_evaluation"] == {"global": True}


def test_context_requires_membership_and_ready_version(tmp_path):
    result, service, datasets, versions = fixture(tmp_path)
    with pytest.raises(ValueError, match="requires dataset"):
        attach_dataset_context(copy.deepcopy(result), service, None, "v1")
    versions[("a", "v1")]["status"] = "RUNNING"
    detail = attach_dataset_context(copy.deepcopy(result), service, "a", "v1")
    assert "not ready" in detail["evaluation_sources"]["stage"]["error"]
    datasets["a"]["members"] = []
    with pytest.raises(ValueError, match="not a member"):
        attach_dataset_context(copy.deepcopy(result), service, "a", "v1")


def test_unannotated_dataset_uses_independent_global_evaluations(tmp_path):
    result, service, datasets, _ = fixture(tmp_path)
    datasets["a"]["reward_version_id"] = None
    datasets["a"]["evaluation_versions"] = []
    detail = attach_dataset_context(result, service, "a", None)
    assert detail["dataset_context"]["status"] == "GLOBAL_FALLBACK"
    assert detail["reward_evaluations"]["stage"]["reward_config"]["stage_exponent"] == 4
    assert detail["rynnvalue_evaluation"] == {"global": True}


def global_fixture(tmp_path):
    result, service, datasets, versions = fixture(tmp_path)
    episode = tmp_path / "episode"
    episode.mkdir()
    trajectory = episode / "trajectory.npz"
    observations = episode / "trajectory_observations.npz"
    np.savez(trajectory, done=np.zeros(20, dtype=bool), env_action=np.zeros((20, 7)))
    np.savez(observations, agentview_image=np.zeros((21, 2, 2, 3), dtype=np.uint8))
    result["run"]["trajectory"] = str(trajectory)
    result.update(evaluation=None, rynnvalue_evaluation=None)
    # Reverse insertion order deliberately: first means earliest completion,
    # not whichever dataset is currently at the top of the list.
    service.list = lambda *_: [datasets["b"], datasets["a"]]
    for dataset_id in ("a", "b"):
        version = versions[(dataset_id, "v1")]
        version["completed_at"] = "2026-09-14T10:00:00+00:00" if dataset_id == "a" else "2026-09-14T11:00:00+00:00"
        prepared = tmp_path / dataset_id / "dataset_manifest.json"
        prepared.write_text(json.dumps({"dataset_sha256": dataset_id, "episodes": [{
            "run_id": "r", "trajectory_path": str(trajectory), "observations_path": str(observations),
            "trajectory_sha256": digest(trajectory), "observations_sha256": digest(observations),
        }]}))
        version["prepared_manifest_path"] = str(prepared)
        manifest_path = Path(version["reward_manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset_sha256"] = dataset_id
        manifest_path.write_text(json.dumps(manifest))
        version["reward_manifest_sha256"] = digest(manifest_path)
    return result, service, datasets, versions


def test_global_uses_first_evaluation_and_survives_dataset_removal(tmp_path):
    from backend.app.services.trajectory_reward_snapshot import ensure_first_reward_snapshot

    result, service, datasets, versions = global_fixture(tmp_path)
    first = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert first["global_evaluation"]["config"]["stage_exponent"] == 2
    assert first["dataset_context"] is None
    assert first["reward_evaluation"]["stage_scores"][10] == pytest.approx(-.75)
    assert first["global_evaluation_pending"] is True
    ensure_first_reward_snapshot(result["run"], service)  # The API queues this off-thread.
    second_dataset = attach_dataset_context(copy.deepcopy(result), service, "b", None)
    assert second_dataset["reward_evaluation"]["stage_scores"][10] == pytest.approx(-.9375)
    # Remove only the fixture manifests. The global result owns its copied array.
    for version in versions.values():
        Path(version["reward_manifest_path"]).unlink()
    service.list = lambda *_: []
    persisted = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert persisted["reward_evaluation"] == first["reward_evaluation"]


def test_cold_global_detail_never_hashes_observation_or_writes_snapshot(tmp_path, monkeypatch):
    import backend.app.services.trajectory_reward_snapshot as snapshots

    result, service, _, _ = global_fixture(tmp_path)
    monkeypatch.setattr(snapshots, "_hash", lambda _: pytest.fail("cold detail must not scan source files"))
    detail = attach_dataset_context(result, service, None, None)
    assert detail["reward_evaluation"]["stage_scores"]
    assert detail["global_evaluation_pending"] is True
    assert not Path(result["run"]["trajectory"]).with_name(snapshots.SIDECAR).exists()


def test_explicit_global_overwrite_refreshes_curve_without_mutating_dataset(tmp_path):
    from backend.app.services.trajectory_reward_snapshot import bind_reward_snapshot

    result, service, _, versions = global_fixture(tmp_path)
    original = attach_dataset_context(copy.deepcopy(result), service, None, None)
    later = versions[("b", "v1")]
    path = Path(later["reward_manifest_path"])
    before = path.read_bytes()
    bind_reward_snapshot(Path(later["prepared_manifest_path"]), path,
                         overwrite=True, origin="manual", evaluation_id="manual")
    updated = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert updated["global_evaluation"]["origin"] == "global"
    assert updated["reward_evaluation"]["stage_scores"] != original["reward_evaluation"]["stage_scores"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("change_content", [False, True])
def test_global_revalidation_never_substitutes_another_dataset(tmp_path, change_content):
    from backend.app.services.trajectory_reward_snapshot import ensure_first_reward_snapshot

    result, service, _, _ = global_fixture(tmp_path)
    ensure_first_reward_snapshot(result["run"], service)
    observations = Path(result["run"]["trajectory"]).with_name("trajectory_observations.npz")
    if change_content:
        np.savez(observations, agentview_image=np.ones((21, 2, 2, 3), dtype=np.uint8))
    else:
        observations.touch()
    before = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert before["global_evaluation_pending"] is True
    assert before["reward_evaluation"] is None
    ensure_first_reward_snapshot(result["run"], service)
    after = attach_dataset_context(copy.deepcopy(result), service, None, None)
    assert after["global_evaluation_pending"] is False
    if change_content:
        assert after["reward_evaluation"] is None
        assert after["global_evaluation_error"]
    else:
        assert after["global_evaluation_error"] is None
        assert after["reward_evaluation"]["stage_scores"][10] == pytest.approx(-.75)
