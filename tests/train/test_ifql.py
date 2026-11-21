"""IFQL: expectile critic beside a plain flow-BC actor, exported for best-of-N inference."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core.artifacts import validate_policy
from core.config import LoadedConfig, load_train_config
from core.io import sha256_file
from train.methods.base import build_algorithm, critic_artifacts, save_critic
from train.encoding import candidate_batch, critic_pixels, encode_actions, encode_proprio
from train.methods.ifql import ImplicitFlowQLearning
from train.methods.selectors import CRITIC_FILE, choose, load_critic

QUANTILE = {"q01": [-1.0] * 8, "q99": [1.0] * 8, "codec": "openpi_quantile"}


def ifql_config(tmp_path=None) -> LoadedConfig:
    raw = load_train_config(method="ifql", family="pi05").raw
    return LoadedConfig(Path(tmp_path or ".") / "ifql.yaml", raw)


def transition_batch(size=3, horizon=10):
    torch.manual_seed(0)
    return {"pixels": torch.randint(0, 255, (size, 6, 32, 32), dtype=torch.uint8),
            "next_pixels": torch.randint(0, 255, (size, 6, 32, 32), dtype=torch.uint8),
            "proprio": torch.randn(size, 8), "next_proprio": torch.randn(size, 8),
            "actions": torch.randn(size, horizon, 7), "action_mask": torch.ones(size, horizon),
            "reward": -torch.rand(size), "bootstrap_mask": torch.ones(size),
            "chunk_length": torch.full((size,), horizon)}


def test_ifql_builds_from_the_catalog_and_trains_only_its_critic():
    config = ifql_config()
    algorithm = build_algorithm(config, torch.device("cpu"))
    assert isinstance(algorithm, ImplicitFlowQLearning)
    assert algorithm.agent.expectile == 0.9 and algorithm.agent.tau == 0.005
    context, metrics = algorithm.update(transition_batch(), step=0)
    # The critic only ranks sampled chunks at inference; it never weights the actor.
    assert context is None and "advantage_weight_mean" not in metrics
    assert {"q_loss", "value_loss", "q_mean", "value_mean", "advantage_mean"} <= metrics.keys()
    losses = torch.tensor([1.0, 3.0], requires_grad=True)
    assert algorithm.weight_actor_losses(losses, context).item() == 2.0
    restored = build_algorithm(config, torch.device("cpu"))
    restored.restore(copy.deepcopy(algorithm.checkpoint()))
    for left, right in zip(algorithm.agent.parameters(), restored.agent.parameters()):
        torch.testing.assert_close(left, right)


def test_exported_critic_scores_with_the_online_min_q(tmp_path):
    config = ifql_config(tmp_path)
    algorithm = build_algorithm(config, torch.device("cpu"))
    algorithm.update(transition_batch(), step=0)
    save_critic(algorithm, config, tmp_path)
    scorer = load_critic(tmp_path / CRITIC_FILE, "cpu")
    assert scorer.settings == {"horizon": 10, "action_dim": 7, "proprio_dim": 8, "image_size": 128}
    batch = transition_batch()
    inputs = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
    with torch.no_grad():
        expected = torch.minimum(algorithm.agent.q1(*inputs), algorithm.agent.q2(*inputs))
    torch.testing.assert_close(scorer(batch), expected)
    # Exports are tensors and plain settings only, loadable without pickled code.
    payload = torch.load(tmp_path / CRITIC_FILE, weights_only=True)
    assert payload["kind"] == "expectile" and not any(key.startswith("target_") for key in payload["state"])


def test_candidate_batch_encodes_like_training_replay():
    rng = np.random.default_rng(1)
    agent, wrist = (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2))
    raw_proprio = rng.normal(size=8).astype(np.float32)
    candidates = rng.uniform(-1, 1, size=(4, 10, 7)).astype(np.float32)
    batch = candidate_batch(agent, wrist, raw_proprio, candidates, image_size=128, horizon=5,
                            action_stats=QUANTILE, proprio_stats=QUANTILE, process_action=lambda action: action)
    torch.testing.assert_close(batch["pixels"][2], critic_pixels(agent, wrist, 128))
    np.testing.assert_allclose(batch["proprio"][0].numpy(), encode_proprio(raw_proprio[None], QUANTILE, QUANTILE)[0])
    np.testing.assert_allclose(batch["actions"][3, :5].numpy(), encode_actions(candidates[3, :5], QUANTILE))
    # The critic sees the planned chunk's first `horizon` steps, as in training replay.
    assert batch["actions"].shape == (4, 5, 7)
    assert batch["action_mask"].sum(1).tolist() == [5, 5, 5, 5]


class RecordingScorer(torch.nn.Module):
    """Scores a chunk by its first action's x; records what it was shown."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.settings = {"horizon": 10, "action_dim": 7, "proprio_dim": 8, "image_size": 16}
        self.seen = None

    def forward(self, batch):
        self.seen = batch
        return batch["actions"][:, 0, 0] + self.anchor


def test_selection_scores_the_whole_planned_chunk_after_post_processing():
    candidates = np.zeros((3, 10, 7), dtype=np.float32)
    candidates[:, 0, 0] = [0.2, 3.0, 0.5]       # candidate 1 is clipped to 1.0 before scoring
    candidates[:, 9, 1] = [0.1, 0.2, 0.3]       # the last planned step reaches the critic
    scorer = RecordingScorer()
    kwargs = dict(agent=np.zeros((8, 8, 3), np.uint8), wrist=np.zeros((8, 8, 3), np.uint8),
                  raw_proprio=np.zeros(8, np.float32), action_stats=QUANTILE, proprio_stats=QUANTILE,
                  process_action=lambda action: np.clip(action, -1, 1))
    index, scores = choose(scorer, {"kind": "argmax", "num_samples": 3}, candidates, **kwargs)
    assert index == 1
    assert scores.tolist() == pytest.approx([0.2, 1.0, 0.5], abs=1e-5)
    assert scorer.seen["action_mask"].sum(1).tolist() == [10, 10, 10]
    np.testing.assert_allclose(scorer.seen["actions"][:, 9, 1].numpy(),
                               encode_actions(candidates[:, 9], QUANTILE)[:, 1], atol=1e-6)
    assert scorer.seen["pixels"].shape == (3, 6, 16, 16)
    # A planned chunk shorter than the critic's horizon is refused, never padded.
    with pytest.raises(ValueError, match="at least `horizon` steps"):
        choose(scorer, {"kind": "argmax", "num_samples": 3}, np.zeros((3, 8, 7), np.float32), **kwargs)


def test_export_carries_critic_and_selector_and_validates(tmp_path):
    from test_policy_export import make_policy
    config = ifql_config(tmp_path)
    algorithm = build_algorithm(config, torch.device("cpu"))
    checkpoint = tmp_path / "checkpoints" / "step_00000002"
    checkpoint.mkdir(parents=True)
    save_critic(algorithm, config, checkpoint)
    source, payload = make_policy(tmp_path, "pi05-liberox-base", "ifql")
    fields = critic_artifacts(config, checkpoint, source.parent)
    assert fields["selector"] == {"kind": "argmax", "num_samples": 32}
    payload.update(critic=fields["critic"], selector=fields["selector"], reward_sha256="e" * 64,
                   component_sha256={**payload["component_sha256"], **fields["component_sha256"]})
    source.write_text(yaml.safe_dump(payload))
    validate_policy(source, payload)
    assert sha256_file(source.parent / CRITIC_FILE) == sha256_file(checkpoint / CRITIC_FILE)
    for broken in ({**payload, "selector": {"kind": "softmax", "num_samples": 4}},
                   {**payload, "selector": {"kind": "argmax", "num_samples": 0}},
                   {key: value for key, value in payload.items() if key != "selector"}):
        with pytest.raises(ValueError):
            validate_policy(source, broken)
    (checkpoint / CRITIC_FILE).unlink()
    with pytest.raises(ValueError, match="lacks its exported critic"):
        critic_artifacts(config, checkpoint, tmp_path)


class FakeCache:
    def __init__(self, value):
        self.value = value

    def batch_repeat_interleave(self, repeats):
        self.value = self.value.repeat_interleave(repeats, dim=0)


class FakeFlowModel(torch.nn.Module):
    """Implements the OpenPI calls the shared-prefix sampler makes, with a toy vector field."""

    def __init__(self):
        from types import SimpleNamespace
        super().__init__()
        self.config = SimpleNamespace(action_horizon=3, action_dim=2)
        self.backbone = torch.nn.Linear(1, 1)
        self.modes_during_prefix = []
        language = SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=language), forward=self._forward)
        self.prefix_passes = 0

    def _preprocess_observation(self, observation, train):
        assert train is False
        return None, None, None, None, observation.state

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embeddings = self._state * 10
        return embeddings, torch.ones(len(embeddings), 4, dtype=torch.bool), None

    def _prepare_attention_masks_4d(self, masks):
        return masks

    def _forward(self, **kwargs):
        self.prefix_passes += 1
        self.modes_during_prefix.append((self.training, self.backbone.training))
        return None, FakeCache(kwargs["inputs_embeds"][0].sum(-1))

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        assert len(state) == len(prefix_pad_masks) == len(past_key_values.value) == len(x_t)
        return (past_key_values.value + state.sum(-1))[:, None, None] * timestep[:, None, None] - x_t

    def reference(self, state, noise, num_steps):
        """Per-candidate Euler integration without any prefix sharing."""
        x_t, dt, time = noise, -1.0 / num_steps, 1.0
        while time >= -dt / 2:
            v_t = (state.sum(-1) * 10 + state.sum(-1))[:, None, None] * time - x_t
            x_t, time = x_t + dt * v_t, time + dt
        return x_t


def test_shared_prefix_sampling_matches_per_candidate_integration(monkeypatch):
    from types import SimpleNamespace
    from train.models import pi05
    install_module = __import__("test_evaluation_policies").install_module
    install_module(monkeypatch, "openpi.models_pytorch.pi0_pytorch", make_att_2d_masks=lambda pad, att: pad)
    model = FakeFlowModel()
    state = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
    model._state = state
    noise = torch.randn(6, 3, 2)
    result = pi05._sample_shared_prefix(model, torch.device("cpu"), SimpleNamespace(state=state), 3, 10, noise=noise)
    # One prefix pass serves every candidate; candidates are grouped per observation.
    assert model.prefix_passes == 1
    expected = model.reference(state.repeat_interleave(3, dim=0), noise, 10)
    torch.testing.assert_close(result, expected)


def test_shared_prefix_sampling_runs_in_eval_mode_and_restores_each_module(monkeypatch):
    # Training with gradient checkpointing drops the prefix KV cache; Value Flows samples mid-training.
    from types import SimpleNamespace
    from train.models import pi05
    install_module = __import__("test_evaluation_policies").install_module
    install_module(monkeypatch, "openpi.models_pytorch.pi0_pytorch", make_att_2d_masks=lambda pad, att: pad)
    model = FakeFlowModel()
    model.train()
    model.backbone.eval()  # A frozen backbone stays in eval mode while the rest trains.
    state = torch.tensor([[1.0, 2.0]])
    model._state = state
    pi05._sample_shared_prefix(model, torch.device("cpu"), SimpleNamespace(state=state), 2, 4,
                               noise=torch.zeros(2, 3, 2))
    assert model.modes_during_prefix == [(False, False)]
    assert model.training and not model.backbone.training


def test_distributed_ifql_matches_single_device_and_exports_unwrapped_critic(tmp_path):
    import torch.distributed as dist
    from train.methods.distributed import DistributedAlgorithm
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        config = ifql_config(tmp_path)
        torch.manual_seed(3)
        single = build_algorithm(config, torch.device("cpu"))
        torch.manual_seed(3)
        parallel = DistributedAlgorithm(config, torch.device("cpu"))
        assert isinstance(parallel.inner.agent.q1, torch.nn.parallel.DistributedDataParallel)
        for step in range(2):
            single.update(transition_batch(), step)
            parallel.update(transition_batch(), step)
        exported, reference = parallel.export_critic(), single.export_critic()
        assert exported["state"].keys() == reference["state"].keys()
        for key, value in reference["state"].items():
            torch.testing.assert_close(exported["state"][key], value, atol=2e-6, rtol=2e-5)
        for optimizer in parallel.optimizers:
            optimizer.consolidate_state_dict(to=0)
        assert parallel.checkpoint()["model"].keys() == single.checkpoint()["model"].keys()
    finally:
        dist.destroy_process_group()
