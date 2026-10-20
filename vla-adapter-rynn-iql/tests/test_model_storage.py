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


def test_digest_cache_reuses_content_and_detects_replaced_bytes(tmp_path, monkeypatch):
    import os
    source = tmp_path / "weights.bin"
    source.write_bytes(b"first")
    before = source.stat()
    original = model_storage.sha256_file
    calls = []

    def count(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(model_storage, "sha256_file", count)
    first = model_storage.cached_sha256(source)
    assert model_storage.cached_sha256(source) == first and len(calls) == 1
    source.write_bytes(b"other")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert model_storage.cached_sha256(source) != first and len(calls) == 2
    for entry in model_storage.digest_cache_root().glob("*.json"):
        entry.write_text("broken")
    assert model_storage.cached_sha256(source) == original(source) and len(calls) == 3


def test_digest_cache_can_be_unwritable_and_rejects_changing_files(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"blocked")
    assert model_storage.cached_sha256(source, blocked) == model_storage.sha256_file(source)
    original = model_storage.sha256_file

    def changed(path):
        digest = original(path)
        path.write_bytes(b"changed")
        return digest

    monkeypatch.setattr(model_storage, "sha256_file", changed)
    with pytest.raises(ValueError, match="changed while hashing"):
        model_storage.cached_sha256(source, tmp_path / "new-cache")


def test_cold_digest_is_computed_only_once_across_processes(tmp_path, monkeypatch):
    import multiprocessing
    import time
    context = multiprocessing.get_context("fork")
    barrier, calls = context.Barrier(4), context.Value("i", 0)
    source = tmp_path / "weights.bin"
    source.write_bytes(b"shared weights")
    original = model_storage.sha256_file

    def count(path):
        with calls.get_lock():
            calls.value += 1
        time.sleep(.05)
        return original(path)

    def worker():
        barrier.wait(timeout=10)
        assert model_storage.cached_sha256(source) == original(source)

    monkeypatch.setattr(model_storage, "sha256_file", count)
    processes = [context.Process(target=worker) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert calls.value == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
