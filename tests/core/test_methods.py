"""Method catalog: declared flags, compatible model families, and rejected combinations."""
from __future__ import annotations

from dataclasses import replace

import pytest

from core import methods
from core.config import load_train_config
from core.registry import family_capabilities


def test_catalog_lists_methods_in_display_order_with_their_flags():
    catalog = methods.method_catalog(family_capabilities())
    assert [item["id"] for item in catalog] == ["iql", "bc"]
    iql, bc = catalog
    assert (iql["label"], iql["summary"], iql["requires_rewards"], iql["critic"]) == ("IQL", "奖励加权后训练", True, "expectile")
    assert (bc["label"], bc["summary"], bc["requires_rewards"], bc["critic"]) == ("BC", "等权行为克隆", False, None)
    assert set(iql["families"]) == set(bc["families"]) == set(family_capabilities())


def test_families_declare_actor_capabilities():
    capabilities = family_capabilities()
    assert capabilities["pi05"] == {"sample"}
    assert capabilities["vla_adapter"] == frozenset()


@pytest.fixture
def sampling_method(monkeypatch):
    """A method whose actor must draw several action chunks per observation (as IDQL will)."""
    method = replace(methods.METHODS["bc"], actor_capabilities=frozenset({"sample"}))
    monkeypatch.setitem(methods.METHODS, "bc", method)
    return method


def test_catalog_offers_a_sampling_method_only_to_families_that_sample(sampling_method):
    catalog = {item["id"]: item for item in methods.method_catalog(family_capabilities())}
    assert catalog["bc"]["families"] == ["pi05"]
    assert catalog["bc"]["actor_capabilities"] == ["sample"]


def test_config_rejects_a_method_the_model_family_cannot_train(sampling_method):
    with pytest.raises(ValueError, match="needs actor capabilities \\['sample'\\]"):
        load_train_config(method="bc", family="vla_adapter")
    assert load_train_config(method="bc", family="pi05").raw["training"]["method"] == "bc"
