import copy
import json

import pytest
import torch
import yaml

from vla_rynn_iql.io import sha256_file, stable_hash
from vla_rynn_iql.model_artifacts import decode_child, export_policy, parent_snapshot, validate_policy
from vla_rynn_iql.models import base_model_config
from vla_rynn_iql.pi05_assets import identity_digest, OPENPI_COMMIT


def make_policy(tmp_path, base_id, algorithm="iql"):
    settings = base_model_config(base_id)
    root = tmp_path / "source"
    root.mkdir()
    names = ({"action_head": "action_head.pt", "proprio_projector": "proprio_projector.pt"}
             if settings["family"] == "vla_adapter" else {"actor": "actor.pt", "base_identity": "pi05_identity.json"})
    for name in names.values():
        torch.save({"weight": torch.eye(3)}, root / name)
    revision = "a" * 64
    if settings["family"] == "pi05":
        identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                    "source": "gs://openpi-assets/checkpoints/pi05_libero",
                    "config_name": "pi05_libero", "native_action_horizon": 10,
                    "native_action_dim": 32, "discrete_state_input": True,
                    "conversion_precision": "bfloat16",
                    "files": {"model.safetensors": "a" * 64, "assets/libero/norm_stats.json": "b" * 64}}
        (root / names["base_identity"]).write_text(json.dumps(identity))
        revision = identity_digest(identity)
    parent = parent_snapshot(settings, revision)
    geometry = {"base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
                "action_horizon": 8, "action_dim": 7, "proprio_dim": 8}
    payload = {"schema_version": 5, "family": settings["family"], "model_config": settings,
               "parent": parent, **geometry, **names,
               "policy_id": "source", "label": "Trained policy", "training_step": 2,
               "algorithm": algorithm, "dataset_sha256": "d" * 64,
               "reward_sha256": "e" * 64 if algorithm == "iql" else None,
               "component_sha256": {key: sha256_file(root / name) for key, name in names.items()},
               "compatibility_sha256": stable_hash(geometry)}
    if settings["family"] == "vla_adapter":
        payload["backbone"] = None
    source = root / "policy.yaml"
    source.write_text(yaml.safe_dump(payload))
    return source, payload


@pytest.mark.parametrize("base_id", ["base", "pi05-libero-base"])
@pytest.mark.parametrize("algorithm", ["bc", "iql"])
def test_deployment_copy_rebinds_location_and_preserves_parent_content(tmp_path, base_id, algorithm):
    source, payload = make_policy(tmp_path, base_id, algorithm)
    root, parent = source.parent, payload["parent"]
    before = {path.name: sha256_file(path) for path in root.iterdir()}
    deployed = export_policy(source, tmp_path / "deployment", "/different/workspace/models/weights")
    copied = yaml.safe_load(deployed.read_text())
    assert copied["policy_id"] == deployed.parent.name == "deployment"
    validate_policy(deployed, copied)
    if base_id == "base":
        from vla_rynn_iql.vla_adapter import load_overlay, validate_overlay
        overlay = load_overlay(deployed)
        validate_overlay(overlay, copied["base_checkpoint"], copied["stats_key"])
    native, rebound = decode_child(copied)
    assert rebound["revision"] == parent["revision"]
    assert rebound["contract"] == parent["contract"]
    assert rebound["sha256"] != parent["sha256"]
    assert native["base_checkpoint"] == native["model_config"]["base_checkpoint"] == rebound["checkpoint"]
    assert copied["component_sha256"] == payload["component_sha256"]
    assert before == {path.name: sha256_file(path) for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        export_policy(source, deployed.parent, "another/path")
    bad = copy.deepcopy(payload)
    bad["parent"]["revision"] = "b" * 64
    source.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="snapshot hash"):
        export_policy(source, tmp_path / "bad", "another/path")


@pytest.mark.parametrize("base_id", ["base", "pi05-libero-base"])
@pytest.mark.parametrize("field,value", [
    ("label", ""), ("policy_id", "different"), ("training_step", True),
    ("dataset_sha256", "bad"), ("compatibility_sha256", "a" * 64),
    ("extra", "unknown"),
])
def test_export_rejects_invalid_full_manifest(tmp_path, base_id, field, value):
    source, payload = make_policy(tmp_path, base_id)
    payload[field] = value
    source.write_text(yaml.safe_dump(payload))
    destination = tmp_path / "deployment"
    with pytest.raises(ValueError):
        export_policy(source, destination, "another/path")
    assert not destination.exists()


def test_export_rejects_pi05_identity_not_bound_to_parent(tmp_path):
    source, payload = make_policy(tmp_path, "pi05-libero-base")
    identity = source.parent / payload["base_identity"]
    identity.write_text(json.dumps({"weights": "different"}))
    payload["component_sha256"]["base_identity"] = sha256_file(identity)
    source.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="identity conflicts"):
        export_policy(source, tmp_path / "deployment", "another/path")
