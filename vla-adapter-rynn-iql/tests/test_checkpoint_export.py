import copy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from vla_rynn_iql import checkpoint_export as exporter
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.model_artifacts import validate_policy
from vla_rynn_iql.models import model_signature
from test_policy_export import make_policy


def make_checkpoint(tmp_path, monkeypatch, base_id="pi05-libero-base", step=2000, mode="full"):
    source, policy = make_policy(tmp_path, base_id, algorithm="bc")
    settings = copy.deepcopy(policy["model_config"])
    settings["backbone"] = mode
    config = load_train_config(method="bc", family=settings["family"], overrides={"model": settings})
    parent = copy.deepcopy(policy["parent"])
    settings = config.section("model")
    checkpoint = tmp_path / f"step_{step:08d}"
    checkpoint.mkdir()
    family = settings["family"]
    if family == "pi05":
        from vla_rynn_iql import pi05 as backend
        identity = json.loads((source.parent / "pi05_identity.json").read_text())
    else:
        from vla_rynn_iql import vla_adapter as backend
        identity = None
    original = SimpleNamespace(model=torch.nn.Linear(2, 2), action_head=torch.nn.Linear(2, 2),
                               proprio_projector=torch.nn.Linear(2, 2), identity=identity, parent=parent)
    backend.save_actor(original, settings, checkpoint)
    metadata = {"schema_version": 2, "step": step, "config_sha256": config.digest,
                "algorithm": "bc", "model_config": model_signature(config.raw),
                "model_parent": parent, "model_identity": identity,
                "model_contract": parent["contract"], "stats_key": parent["stats_key"],
                "base_checkpoint": settings["base_checkpoint"],
                "dataset_sha256": "d" * 64, "reward_sha256": None}
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    (checkpoint / "effective_config.yaml").write_text(yaml.safe_dump(config.raw))
    # Export must not deserialize optimizer/trainer state at all.
    (checkpoint / "trainer.pt").write_bytes(b"not a torch archive")
    loaded = SimpleNamespace(model=torch.nn.Linear(2, 2), action_head=torch.nn.Linear(2, 2),
                             proprio_projector=torch.nn.Linear(2, 2), identity=identity,
                             parent=copy.deepcopy(parent))
    calls = []

    def load_components(saved_config, *, training):
        calls.append((saved_config, training))
        assert saved_config.section("model")["base_revision"] == parent["revision"]
        return loaded

    monkeypatch.setattr(exporter, "model_backend", lambda _: SimpleNamespace(
        load_components=load_components, restore_actor=backend.restore_actor, export_actor=backend.export_actor))
    return checkpoint, original, loaded, calls


@pytest.mark.parametrize("base_id", ["base", "pi05-libero-base"])
@pytest.mark.parametrize("step", [2000, 5000])
@pytest.mark.parametrize("mode", ["frozen", "full"])
def test_checkpoint_exports_exact_actor_and_does_not_publish_latest(tmp_path, monkeypatch, base_id, step, mode):
    checkpoint, original, loaded, calls = make_checkpoint(tmp_path, monkeypatch, base_id, step, mode)
    before = {p.name: sha256_file(p) for p in checkpoint.iterdir()}
    output = tmp_path / "registry" / f"policy-{step}"
    result = exporter.export_checkpoint(checkpoint, output)
    payload = yaml.safe_load(result.read_text())
    validate_policy(result, payload)
    assert payload["training_step"] == step and len(calls) == 1 and calls[0][1] is True
    assert not (output.parent / "latest").exists()
    assert before == {p.name: sha256_file(p) for p in checkpoint.iterdir()}
    names = ["model"] if base_id != "base" else ["action_head", "proprio_projector"]
    for name in names:
        for key, tensor in getattr(original, name).state_dict().items():
            assert torch.equal(tensor, getattr(loaded, name).state_dict()[key])
    provenance = json.loads((output / "checkpoint_export.json").read_text())
    assert provenance["step"] == step and "trainer.pt" not in provenance["source_sha256"]
    with pytest.raises(FileExistsError):
        exporter.export_checkpoint(checkpoint, output)
    assert len(calls) == 1


@pytest.mark.parametrize("field,value,error", [
    ("step", 5000, "directory step"), ("step", True, "positive integer"),
    ("config_sha256", "bad", "configuration hash"),
    ("model_parent", None, "frozen parent"), ("algorithm", "iql", "algorithm"),
    ("model_config", {}, "model settings"), ("model_contract", {}, "contract"),
])
def test_rejects_checkpoint_metadata_mismatch_before_model_load(tmp_path, monkeypatch, field, value, error):
    checkpoint, _, _, calls = make_checkpoint(tmp_path, monkeypatch)
    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        exporter.export_checkpoint(checkpoint, tmp_path / "out")
    assert calls == [] and not (tmp_path / "out").exists()


def test_loaded_wrong_parent_is_rejected(tmp_path, monkeypatch):
    checkpoint, _, loaded, _ = make_checkpoint(tmp_path, monkeypatch)
    loaded.parent["revision"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot hash"):
        exporter.export_checkpoint(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failed_restore_does_not_leave_published_policy(tmp_path, monkeypatch):
    checkpoint, _, _, _ = make_checkpoint(tmp_path, monkeypatch)
    torch.save({"wrong": torch.ones(1)}, checkpoint / "actor.pt")
    with pytest.raises(ValueError, match="actor keys"):
        exporter.export_checkpoint(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_checkpoint_change_during_export_is_rejected(tmp_path, monkeypatch):
    checkpoint, _, loaded, _ = make_checkpoint(tmp_path, monkeypatch)
    from vla_rynn_iql import pi05

    def changing_export(*args, **kwargs):
        result = pi05.export_actor(*args, **kwargs)
        (checkpoint / "actor.pt").write_bytes(b"changed")
        return result

    monkeypatch.setattr(exporter, "model_backend", lambda _: SimpleNamespace(
        load_components=lambda *a, **kw: loaded, restore_actor=pi05.restore_actor,
        export_actor=changing_export))
    with pytest.raises(ValueError, match="changed while exporting"):
        exporter.export_checkpoint(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_lora_checkpoint_is_restored_then_merged_without_changing_source(tmp_path, monkeypatch):
    checkpoint, original, loaded, calls = make_checkpoint(tmp_path, monkeypatch, "base", mode="lora")
    from vla_rynn_iql import vla_adapter

    class TinyAdapter(torch.nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
            self.delta = torch.nn.Parameter(torch.full((2, 2), delta))

        def merge_and_unload(self, *, safe_merge):
            assert safe_merge
            result = torch.nn.Linear(2, 2, bias=False)
            result.weight.data.copy_(self.weight + self.delta)
            return result

    original.model, loaded.model = TinyAdapter(0.5), TinyAdapter(0.0)
    settings = yaml.safe_load((checkpoint / "effective_config.yaml").read_text())["model"]
    vla_adapter.save_actor(original, settings, checkpoint)
    before = sha256_file(checkpoint / "backbone.pt")
    result = exporter.export_checkpoint(checkpoint, tmp_path / "merged")
    merged = torch.load(result.parent / "backbone.pt", weights_only=True)
    assert torch.equal(merged["weight"], torch.eye(2) + 0.5)
    assert before == sha256_file(checkpoint / "backbone.pt")
    assert len(calls) == 1
