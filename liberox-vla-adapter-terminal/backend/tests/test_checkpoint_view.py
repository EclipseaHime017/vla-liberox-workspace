from pathlib import Path
import pytest

from backend.app.policies.catalog import PolicyCatalog
from backend.app.policies.checkpoint_view import checkpoint_view


def test_selection_pins_base_and_loader_cannot_mutate_source(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"example"}')
    (base / "model.safetensors").write_bytes(b"model")
    catalog = PolicyCatalog(tmp_path / "registry", base_models={
        "base": {"base_checkpoint": str(base), "stats_key": "stats"}})
    entry = catalog.select("base")
    with checkpoint_view(entry) as view:
        copied = Path(view) / "config.json"
        copied.write_text('{"auto_map":"patched"}')
        assert (base / "config.json").read_text() == '{"model_type":"example"}'
        assert not (Path(view) / "model.safetensors").is_symlink()
        assert (Path(view) / "model.safetensors").read_bytes() == b"model"
    assert catalog.select("base").content_sha256 == entry.content_sha256
    (base / "model.safetensors").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_view(entry)
    assert catalog.select("base").content_sha256 != entry.content_sha256


def test_listing_does_not_resolve_or_download_remote_model(tmp_path, monkeypatch):
    import huggingface_hub
    def network_forbidden(*_, **__):
        raise AssertionError("list requested network")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", network_forbidden)
    catalog = PolicyCatalog(tmp_path, base_models={
        "base": {"base_checkpoint": "owner/not-downloaded", "stats_key": "stats"}})
    assert catalog.list_policies()[0]["base_revision"] is None
