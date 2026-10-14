from pathlib import Path

import pytest

from vla_rynn_iql import checkpoint_assets, model_storage


def test_direct_download_preserves_revision_and_reuses_without_network(tmp_path, monkeypatch):
    import huggingface_hub

    root = tmp_path / "models"
    monkeypatch.setattr(model_storage, "MODELS_ROOT", root)
    revision = "a" * 40
    calls = []

    def download(repo, *, revision, local_dir, allow_patterns):
        calls.append(repo)
        assert repo == "example/policy" and revision == "a" * 40
        assert allow_patterns is None
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    destination = model_storage.download_repository("example/policy", revision)
    assert destination == root / "example-policy"
    assert not (destination / "model.safetensors").is_symlink()
    assert checkpoint_assets.resolve_revision("example/policy") == revision
    with checkpoint_assets.checkpoint_view("example/policy", revision) as view:
        assert (Path(view) / "model.safetensors").read_bytes() == b"weights"
        (Path(view) / "config.json").write_text('{"modified": true}')
        assert (destination / "config.json").read_text() == "{}"
        assert not (Path(view) / model_storage.SOURCE_FILE).exists()
    assert len(calls) == 1
    with pytest.raises(ValueError, match="revision/download scope"):
        model_storage.download_repository("example/policy", "b" * 40)
    (destination / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="files changed"):
        checkpoint_assets.checkpoint_view("example/policy", revision)


def test_failed_download_is_not_published(tmp_path, monkeypatch):
    import huggingface_hub

    root = tmp_path / "models"
    monkeypatch.setattr(model_storage, "MODELS_ROOT", root)
    def download(*args, **kwargs):
        raise RuntimeError("download interrupted")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="interrupted"):
        model_storage.download_repository("example/policy", "a" * 40)
    assert not (root / "example-policy").exists()


def test_unknown_existing_directory_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(model_storage, "MODELS_ROOT", tmp_path)
    (tmp_path / "example-policy").mkdir()
    with pytest.raises(ValueError, match="source metadata"):
        model_storage.download_repository("example/policy", "a" * 40)


def test_openpi_cache_does_not_redirect_other_huggingface_consumers(monkeypatch):
    import os

    monkeypatch.setenv("HF_HOME", "/unchanged/huggingface")
    monkeypatch.setenv("OPENPI_DATA_HOME", "/obsolete/openpi")
    model_storage.configure_openpi_cache()
    assert os.environ["OPENPI_DATA_HOME"] == str(model_storage.OPENPI_CACHE_ROOT)
    assert os.environ["HF_HOME"] == "/unchanged/huggingface"
    assert model_storage.MODELS_ROOT == Path(__file__).resolve().parents[2] / "models"


def test_legacy_path_resolution_is_scoped_to_this_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(model_storage, "MODELS_ROOT", tmp_path / "models")
    assert model_storage.local_model_directory(tmp_path / "weights/model") == tmp_path / "models/model"
    assert model_storage.local_model_directory("/another/project/weights/model") == Path("/another/project/weights/model")
    old = tmp_path / "weights/explicit"
    old.mkdir(parents=True)
    assert model_storage.local_model_directory(old) == old
