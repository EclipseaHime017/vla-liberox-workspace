from importlib import metadata

import pytest

from vla_adapter_robometer.runtime import public_version, validate_runtime_versions


def test_public_version_ignores_cuda_suffix():
    assert public_version("2.8.0+cu128") == "2.8.0"


def test_runtime_reports_the_whole_incompatible_matrix(monkeypatch):
    versions = {
        "torch": "2.7.0+cu128", "torchvision": "0.23.0",
        "torchao": "0.13.0+cu128", "xformers": "0.0.32.post2",
        "transformers": "4.57.1", "trl": "0.20.0",
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match=r"torch==2.7.0\+cu128.*expected 2.8.0"):
        validate_runtime_versions()
