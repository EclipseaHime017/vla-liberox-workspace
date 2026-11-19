"""Platform training parameters follow each method's declared critic fields (IFQL, VF vs IQL)."""
import pytest

from test_dataset_reward_versions import setup_jobs
from studio.services.offline_job_service import OfflineJobService


def test_ifql_defaults_offer_its_critic_fields_and_sample_count(tmp_path):
    jobs, _ = setup_jobs(tmp_path)
    defaults = jobs.defaults(algorithm="ifql", model_family="pi05")
    assert defaults["algorithm"] == "ifql"
    # IFQL's actor is plain flow BC: no critic warmup or advantage weighting.
    assert "critic_warmup_steps" not in defaults["basic"]
    assert defaults["advanced"]["expectile"] == 0.9 and defaults["advanced"]["num_samples"] == 32
    assert not {"beta", "max_advantage_weight"} & defaults["advanced"].keys()
    assert defaults["fixed"]["critic_image_size"] == 128
    iql = jobs.defaults(algorithm="iql")
    assert {"beta", "max_advantage_weight", "critic_lr"} <= iql["advanced"].keys()
    assert "critic_warmup_steps" in iql["basic"] and "num_samples" not in iql["advanced"]


@pytest.mark.parametrize("parameters,message", [
    ({"algorithm": "ifql", "beta": 3.0}, r"IFQL does not accept \['beta'\]"),
    ({"algorithm": "ifql", "critic_warmup_steps": 10}, "IFQL does not accept"),
    ({"algorithm": "iql", "num_samples": 8}, r"IQL does not accept \['num_samples'\]"),
    ({"algorithm": "bc", "expectile": 0.7}, "does not accept critic parameters"),
])
def test_methods_reject_fields_they_do_not_declare(parameters, message):
    with pytest.raises(ValueError, match=message):
        OfflineJobService._validate_training_parameters(parameters)


def test_ifql_accepts_its_own_fields():
    OfflineJobService._validate_training_parameters(
        {"algorithm": "ifql", "num_samples": 8, "expectile": 0.9, "critic_lr": 3e-4, "target_tau": 0.005})


def test_vf_defaults_offer_flow_critic_fields_and_its_parameters(tmp_path):
    jobs, _ = setup_jobs(tmp_path)
    defaults = jobs.defaults(algorithm="vf", model_family="pi05")
    advanced = defaults["advanced"]
    assert "critic_warmup_steps" not in defaults["basic"]
    assert advanced["critic_optimizer"] == "adam" and advanced["critic_lr"] == 0.0003
    assert not {"value_lr", "expectile", "beta"} & advanced.keys()
    assert (advanced["num_samples"], advanced["train_num_samples"], advanced["num_flow_steps"]) == (16, 16, 10)
    assert advanced["ret_agg"] == advanced["q_agg"] == "mean"


@pytest.mark.parametrize("parameters,message", [
    ({"algorithm": "vf", "expectile": 0.7}, r"VF does not accept \['expectile'\]"),
    ({"algorithm": "vf", "value_lr": 1e-4}, "VF does not accept"),
    ({"algorithm": "ifql", "bcfm_lambda": 1.0}, r"IFQL does not accept \['bcfm_lambda'\]"),
])
def test_vf_and_ifql_keep_their_fields_apart(parameters, message):
    with pytest.raises(ValueError, match=message):
        OfflineJobService._validate_training_parameters(parameters)
    OfflineJobService._validate_training_parameters(
        {"algorithm": "vf", "ret_agg": "min", "num_samples": 8, "train_num_samples": 4, "critic_lr": 1e-4})
