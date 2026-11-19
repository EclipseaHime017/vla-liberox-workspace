"""Method catalog: declared flags, compatible model families, and rejected combinations."""
from __future__ import annotations

from dataclasses import replace

import pytest

from core import methods
from core.config import load_train_config
from core.registry import family_capabilities


def test_catalog_lists_methods_in_display_order_with_their_flags():
    catalog = methods.method_catalog(family_capabilities())
    assert [item["id"] for item in catalog] == ["iql", "bc", "ifql", "vf"]
    iql, bc, ifql, vf = catalog
    assert (iql["label"], iql["summary"], iql["requires_rewards"], iql["critic"]) == ("IQL", "奖励加权后训练", True, "expectile")
    assert (bc["label"], bc["summary"], bc["requires_rewards"], bc["critic"]) == ("BC", "等权行为克隆", False, None)
    assert set(iql["families"]) == set(bc["families"]) == set(family_capabilities())
    assert (ifql["label"], ifql["requires_rewards"], ifql["critic"], ifql["exports_critic"]) == ("IFQL", True, "expectile", True)
    assert ifql["families"] == ["pi05"]
    assert (vf["critic"], vf["exports_critic"], vf["actor_samples_in_training"], vf["families"]) == (
        "value_flow", True, True, ["pi05"])
    assert not any(item["actor_samples_in_training"] for item in (iql, bc, ifql))


def test_methods_declare_the_critic_fields_they_accept():
    iql, bc, ifql = (methods.METHODS[name] for name in ("iql", "bc", "ifql"))
    assert bc.critic_keys == ()
    assert {"beta", "max_advantage_weight", "critic_warmup_steps"} <= set(iql.critic_keys)
    # IFQL's actor is plain flow BC: no advantage weighting or warmup.
    assert set(ifql.critic_keys) == set(methods.EXPECTILE_CRITIC_KEYS)


def test_selector_registry_matches_the_torch_free_declaration():
    from train.methods.selectors import SELECTORS
    assert tuple(SELECTORS) == methods.SELECTOR_KINDS


def test_ifql_configuration_validates_its_own_section():
    raw = load_train_config(method="ifql", family="pi05").raw
    assert raw["ifql"]["num_samples"] == 32 and raw["ifql"]["expectile"] == 0.9
    assert raw["iql"] == {} and raw["training"]["method"] == "ifql"
    assert methods.critic_warmup_steps(raw) == 0
    with pytest.raises(ValueError, match="needs actor capabilities"):
        load_train_config(method="ifql", family="vla_adapter")
    from core.config import validate_train_config
    from pathlib import Path
    for key, value in (("num_samples", 0), ("expectile", 1.5), ("critic_optimizer", "sgd")):
        broken = {**raw, "ifql": {**raw["ifql"], key: value}}
        with pytest.raises((TypeError, ValueError)):
            validate_train_config(broken, Path("ifql.yaml"))


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


def test_vf_configuration_validates_its_own_section():
    raw = load_train_config(method="vf", family="pi05").raw
    assert raw["vf"]["clip_flow_returns"] is True and raw["ifql"] == {} == raw["iql"]
    from core.config import validate_train_config
    from pathlib import Path
    for key, value in (("ret_agg", "max"), ("confidence_weight_temp", 0), ("clip_flow_returns", "yes"),
                       ("num_flow_steps", 0), ("critic_optimizer", "sgd")):
        broken = {**raw, "vf": {**raw["vf"], key: value}}
        with pytest.raises((TypeError, ValueError)):
            validate_train_config(broken, Path("vf.yaml"))
