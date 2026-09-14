from __future__ import annotations

import json
import importlib.metadata
import logging
import math
import os
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import yaml
from PIL import Image

from .config import LoadedConfig, reward_source
from .data import load_manifest
from .io import atomic_json, sha256_file, stable_hash


LOG = logging.getLogger(__name__)

# Schema v6 separates expensive RynnValue inference from deterministic reward
# reduction.  Annotation artifacts contain only official model outputs;
# sparse/PBRS/final rewards live in a second cache keyed by their derivation
# parameters.
ANNOTATION_SCHEMA_VERSION = 6
REWARD_SCHEMA_VERSION = 1
REUSABLE_OFFICIAL_OUTPUT_SCHEMA_VERSIONS = frozenset({4, 5, 6})
OFFICIAL_INFERENCE_CONFIG_KEYS = frozenset({
    "model", "revision", "dtype", "max_frames",
    "robot_description", "camera_description",
})
REWARD_DERIVATION_CONFIG_KEYS = frozenset({
    "rynnvalue", "gamma", "shaping_weight", "accumulate_primitive_steps",
})
TRAINING_REDUCTION_CONFIG_KEYS = frozenset({"gamma", "accumulate_primitive_steps"})
OFFICIAL_OUTPUT_KEYS = (
    "absolute_temporal_distance_seconds",
    "absolute_value_entropy_nats",
    "absolute_value_logits",
    "relative_temporal_distance_seconds",
    "relative_value_logits",
)


def official_inference_config(reward_config: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that can change frozen RynnValue outputs."""
    return {
        key: reward_config[key]
        for key in sorted(OFFICIAL_INFERENCE_CONFIG_KEYS)
    }


def reward_derivation_config(reward_config: dict[str, Any]) -> dict[str, Any]:
    """Return cheap reward-reduction settings, independent of VLM inference."""
    result = {
        key: reward_config[key]
        for key in sorted(REWARD_DERIVATION_CONFIG_KEYS)
    }
    source = reward_source(reward_config)
    result["rynnvalue"] = source == "rynnvalue"
    # Preserve RynnValue's existing manifest/checkpoint identity exactly.
    if source != "rynnvalue":
        result["source"] = source
    if source == "stage":
        result["stage_exponent"] = float(reward_config.get("stage_exponent", 2.0))
    return result


def reward_implementation_fingerprint(source: str) -> str:
    """Invalidate cheap derived caches when formula code changes, never raw VLM outputs."""
    files = [Path(__file__)]
    if source == "stage":
        files.append(Path(__file__).with_name("stage_rewards.py"))
    return stable_hash({path.name: sha256_file(path) for path in files})


def _reward_generation_dir(root: Path) -> Path:
    """Every rebuild gets new files; old training references are never overwritten."""
    directory = root / "versions" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _episode_reward_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_sha256": episode["trajectory_sha256"],
        "observations_sha256": episode["observations_sha256"],
        "recorded_action_count": episode["recorded_action_count"],
        "reward_boundaries": episode["reward_boundaries"],
        "evaluation_chunks": episode.get("evaluation_chunks", episode["chunks"]),
    }


def _episode_timeline_arrays(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(episode["trajectory_path"], allow_pickle=False) as trajectory:
        times = np.asarray(trajectory["time_seconds"], dtype=np.float64)
        done = np.asarray(trajectory["done"], dtype=bool)
    return {"observation_steps": np.arange(len(times), dtype=np.int64),
            "time_seconds": times, "environment_done": done}


def reward_manifest_digest(index: dict[str, Any]) -> str:
    """Content identity for direct rewards; legacy RynnValue hashes stay unchanged."""
    if (index.get("reward_config", {}).get("source", "rynnvalue") == "rynnvalue"
            and not index.get("source_reward_manifest_sha256")):
        return stable_hash(index)
    identity = {
        "schema_version": index["schema_version"], "kind": index["kind"],
        "dataset_sha256": index["dataset_sha256"], "reward_config": index["reward_config"],
        "stage_annotations_sha256": index.get("stage_annotations_sha256"),
        "episodes": [{"run_id": item["run_id"], "reward_sha256": item["reward_sha256"],
                      "stage_annotation_sha256": item.get("stage_annotation_sha256")}
                     for item in index["episodes"]],
    }
    if "derivation_implementation_sha256" in index:
        identity["derivation_implementation_sha256"] = index["derivation_implementation_sha256"]
    if index.get("source_reward_manifest_sha256"):
        identity.update(source_reward_manifest_sha256=index["source_reward_manifest_sha256"],
                        source_reward_version_id=index["source_reward_version_id"])
    return stable_hash(identity)


def validate_rynnvalue_config_contract(config: Any, processor: Any) -> dict[str, Any]:
    """Validate the pinned RynnValue/Qwen value-head interface before loading weights."""
    if getattr(config, "model_type", None) != "rynn_value_lang":
        raise RuntimeError(
            "The reward checkpoint is not a RynnValue language model: "
            f"model_type={getattr(config, 'model_type', None)!r}"
        )
    head_config = getattr(config, "value_head_config", None)
    if head_config is None:
        raise RuntimeError(
            "RynnValue checkpoint has no value_head_config; refusing to fall back to "
            "the Qwen language-model head"
        )
    hidden_size = int(config.text_config.hidden_size)
    repeat = int(config.value_token_repeat)
    processor_repeat = int(processor.value_token_repeat)
    if repeat < 1 or processor_repeat != repeat:
        raise RuntimeError(
            "RynnValue processor/model value-token repeat mismatch: "
            f"processor={processor_repeat}, model={repeat}"
        )
    bins = int(config.value_tokenizer_config.bins)
    if bins < 2:
        raise RuntimeError(f"RynnValue value tokenizer has invalid bins={bins}")
    relative_head_config = getattr(config, "relative_value_head_config", None)
    relative_tokenizer_config = getattr(config, "relative_value_tokenizer_config", None)
    if relative_head_config is None or relative_tokenizer_config is None:
        raise RuntimeError(
            "RynnValue checkpoint has no relative temporal-distance head; the complete "
            "official output contract cannot be recorded"
        )
    relative_repeat = int(getattr(config, "relative_value_token_repeat", 0))
    processor_relative_repeat = int(getattr(processor, "relative_value_token_repeat", 0))
    if relative_repeat < 1 or processor_relative_repeat != relative_repeat:
        raise RuntimeError(
            "RynnValue processor/model relative-value-token repeat mismatch: "
            f"processor={processor_relative_repeat}, model={relative_repeat}"
        )
    relative_bins = int(relative_tokenizer_config.bins)
    if relative_bins < 2:
        raise RuntimeError(
            f"RynnValue relative value tokenizer has invalid bins={relative_bins}"
        )
    return {
        "model_type": config.model_type,
        "qwen_hidden_size": hidden_size,
        "value_token_repeat": repeat,
        "value_head_input_size": hidden_size * repeat,
        "value_head_type": str(head_config.head_type),
        "value_bins": bins,
        "value_head_count": int(config.num_value_heads),
        "relative_value_token_repeat": relative_repeat,
        "relative_value_head_input_size": hidden_size * relative_repeat,
        "relative_value_head_type": str(relative_head_config.head_type),
        "relative_value_bins": relative_bins,
    }


def validate_rynnvalue_runtime_dtype(
    model: Any,
    requested_dtype: Any,
    expected_head_input_size: int,
    expected_relative_head_input_size: int | None = None,
) -> dict[str, Any]:
    """Reject mixed FP32/BF16 models before the first expensive annotation call."""
    heads = getattr(model, "value_heads", None)
    if heads is None or len(heads) < 1:
        raise RuntimeError("Loaded RynnValue model has no dedicated value head")
    projection = heads[0].proj
    input_layer = getattr(projection, "input_layer", projection)
    weight = getattr(input_layer, "weight", None)
    if weight is None or weight.ndim != 2:
        raise RuntimeError("Loaded RynnValue value head has an unsupported input projection")
    if int(weight.shape[1]) != int(expected_head_input_size):
        raise RuntimeError(
            "RynnValue value-head input does not match the Qwen hidden-state contract: "
            f"loaded={int(weight.shape[1])}, expected={int(expected_head_input_size)}"
        )
    relative_weight = None
    if expected_relative_head_input_size is not None:
        relative_head = getattr(model, "relative_value_head", None)
        if relative_head is None:
            raise RuntimeError("Loaded RynnValue model has no dedicated relative value head")
        relative_projection = relative_head.proj
        relative_input = getattr(relative_projection, "input_layer", relative_projection)
        relative_weight = getattr(relative_input, "weight", None)
        if relative_weight is None or relative_weight.ndim != 2:
            raise RuntimeError(
                "Loaded RynnValue relative value head has an unsupported input projection"
            )
        if int(relative_weight.shape[1]) != int(expected_relative_head_input_size):
            raise RuntimeError(
                "RynnValue relative value-head input does not match the Qwen hidden-state "
                f"contract: loaded={int(relative_weight.shape[1])}, "
                f"expected={int(expected_relative_head_input_size)}"
            )

    dtype_counts: dict[str, int] = defaultdict(int)
    mismatches: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        dtype_counts[str(parameter.dtype)] += parameter.numel()
        if parameter.dtype != requested_dtype and len(mismatches) < 8:
            mismatches.append(f"{name}={parameter.dtype}")
    if mismatches:
        raise RuntimeError(
            "RynnValue contains floating parameters that were not converted to the requested "
            f"dtype {requested_dtype}: {', '.join(mismatches)}"
        )
    result = {
        "runtime_dtype": str(requested_dtype),
        "floating_parameter_dtypes": dict(dtype_counts),
        "value_head_input_shape": list(weight.shape),
        "value_head_dtype": str(weight.dtype),
    }
    if relative_weight is not None:
        result.update({
            "relative_value_head_input_shape": list(relative_weight.shape),
            "relative_value_head_dtype": str(relative_weight.dtype),
        })
    return result


class TemporalValueAnnotator(Protocol):
    metadata: dict[str, Any]

    def predict(self, prompt: str, frames: Sequence[np.ndarray]) -> dict[str, np.ndarray]: ...

    def analyze(self, prompt: str, frames: Sequence[np.ndarray]) -> dict[str, Any]: ...


def reward_view(frame: np.ndarray, orientation: str) -> np.ndarray:
    """Return an upright third-person image for RynnValue."""
    array = np.asarray(frame, dtype=np.uint8)
    if orientation == "libero_raw":
        return array[::-1].copy()
    if orientation == "vla_policy":
        # VLA input is raw framebuffer rotated by 180 degrees; undo the
        # horizontal part to obtain the normal display orientation.
        return array[:, ::-1].copy()
    raise ValueError(f"Unknown observation orientation: {orientation}")


def policy_view(frame: np.ndarray, orientation: str) -> np.ndarray:
    array = np.asarray(frame, dtype=np.uint8)
    if orientation == "libero_raw":
        return array[::-1, ::-1].copy()
    if orientation == "vla_policy":
        return array.copy()
    raise ValueError(f"Unknown observation orientation: {orientation}")


class RynnValueAnnotator:
    """Frozen official RynnValue model loaded from a pinned HF snapshot."""

    def __init__(self, config: LoadedConfig):
        reward = config.section("reward")
        checkout = Path(config.section("paths")["rynnvalue_root"]).resolve()
        package_init = checkout / "rynn_value" / "__init__.py"
        if not package_init.is_file():
            raise RuntimeError(
                f"RynnValue source checkout is invalid; missing {package_init}. "
                "Clone the pinned official repository instead of pip-installing it."
            )
        lock_path = Path(__file__).resolve().parents[2] / "configs" / "dependency-lock.yaml"
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["rynnvalue"]
        revision = reward["revision"]
        if revision != lock["hf_revision"]:
            raise RuntimeError(
                "RynnValue model request is not the pinned revision: "
                f"requested={revision}, expected={lock['hf_revision']}"
            )
        try:
            git_commit = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot verify RynnValue source checkout: {checkout}") from exc
        if git_commit != lock["git_commit"]:
            raise RuntimeError(
                "Official RynnValue checkout is not the pinned audited revision: "
                f"expected {lock['git_commit']}, got {git_commit}"
            )
        if dirty:
            raise RuntimeError(
                "Official RynnValue checkout has local modifications; restore a clean pinned "
                f"checkout before annotation:\n{dirty}"
            )
        sys.path.insert(0, str(checkout))
        try:
            import torch
            import rynn_value
            from rynn_value import (
                RynnValueLangConfig,
                RynnValueLangModel,
                RynnValueLangProcessor,
            )
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "RynnValue dependencies are unavailable. Use the rynnvalue-reward environment "
                "and install requirements-reward.txt."
            ) from exc
        loaded_checkout = Path(rynn_value.__file__).resolve().parents[1]
        if loaded_checkout != checkout:
            raise RuntimeError(
                "Imported RynnValue from an unexpected checkout: "
                f"configured={checkout}, imported={loaded_checkout}"
            )
        snapshot = Path(snapshot_download(repo_id=reward["model"], revision=revision)).resolve()
        # A snapshot directory is immutable and its final path component is the
        # resolved Hub commit hash, even if the requested revision was 'main'.
        resolved_revision = snapshot.name
        dtype = getattr(torch, reward["dtype"])
        hf_config = RynnValueLangConfig.from_pretrained(snapshot, local_files_only=True)
        hf_config._attn_implementation = "pred_slot_isolated_eager"
        self.processor = RynnValueLangProcessor.from_pretrained(snapshot, local_files_only=True)
        model_contract = validate_rynnvalue_config_contract(hf_config, self.processor)
        model = RynnValueLangModel.from_pretrained(
            snapshot, config=hf_config, dtype=dtype,
            local_files_only=True, low_cpu_mem_usage=True,
        )
        # The pinned upstream value-head constructors explicitly default to
        # float32. Match the official inference program and cast the complete
        # model so custom heads cannot remain FP32 beside a BF16 Qwen backbone.
        self.model = model.to(device=reward["device"], dtype=dtype).eval()
        runtime_contract = validate_rynnvalue_runtime_dtype(
            self.model,
            dtype,
            model_contract["value_head_input_size"],
            model_contract["relative_value_head_input_size"],
        )
        self.torch = torch
        self.device = reward["device"]
        model_files = sorted(
            path for path in snapshot.iterdir()
            if path.is_file() and path.suffix in {".json", ".py", ".safetensors"}
        )
        if resolved_revision != lock["hf_revision"]:
            raise RuntimeError(
                "RynnValue model snapshot is not the pinned revision: "
                f"resolved={resolved_revision}, expected={lock['hf_revision']}"
            )
        self.metadata = {
            "provider": "rynnvalue",
            "model": reward["model"],
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "official_code_commit": git_commit,
            "snapshot": str(snapshot),
            "source_checkout": str(checkout),
            "file_sha256": {path.name: sha256_file(path) for path in model_files},
            "dtype": reward["dtype"],
            "device": reward["device"],
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "model_contract": model_contract,
            "runtime_contract": runtime_contract,
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "huggingface-hub")
            },
        }
        self.robot_description = reward["robot_description"]
        self.camera_description = reward["camera_description"]
        self.max_frames = int(reward["max_frames"])
        self.batch_size = int(reward["annotation_batch_size"])
        self.value_head_count = int(model_contract["value_head_count"])

    def _inputs(self, prompt: str, frames: Sequence[np.ndarray]):
        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        return self.processor.process_episode(
            instruction=prompt, images=images,
            robot_description=self.robot_description,
            camera_description=self.camera_description,
        )

    def _prefix_inputs(
        self, prompt: str, frames: Sequence[np.ndarray], end_index: int
    ):
        # Match the official inference program: every prediction sees a
        # uniformly resampled prefix with a fixed number of image slots, and
        # only the last value slot is read.
        indices = np.linspace(0, end_index, self.max_frames, dtype=np.int64)
        return self._inputs(prompt, [frames[int(index)] for index in indices])

    def _batch_kwargs(self, samples: Sequence[Any]) -> dict[str, Any]:
        torch = self.torch
        return {
            "input_ids": torch.cat([sample["input_ids"] for sample in samples], dim=0)
            .to(self.device).long(),
            "attention_mask": torch.cat(
                [sample["attention_mask"] for sample in samples], dim=0
            ).to(self.device).long(),
            "pixel_values": torch.cat(
                [sample["pixel_values"].flatten(0, 1) for sample in samples], dim=0
            ).to(self.device),
            "image_grid_thw": torch.cat(
                [sample["image_grid_thw"].flatten(0, 1) for sample in samples], dim=0
            ).to(self.device).long(),
        }

    @staticmethod
    def _absolute_last_slots(tensor: Any, sample_count: int, head_count: int) -> Any:
        """Select the official demo's last prefix slot without averaging heads."""
        tensor = tensor.float()
        if tensor.dim() == 1 and head_count == 1:
            tensor = tensor.reshape(1, sample_count, -1)
        elif tensor.dim() == 2 and tensor.shape[0] == head_count:
            tensor = tensor.reshape(head_count, sample_count, -1)
        elif tensor.dim() == 3 and tensor.shape[:2] == (head_count, sample_count):
            pass
        else:
            raise ValueError(
                "Unexpected RynnValue absolute prediction shape: "
                f"{tuple(tensor.shape)} for samples={sample_count}, heads={head_count}"
            )
        return tensor[:, :, -1].transpose(0, 1).contiguous()

    @staticmethod
    def _absolute_last_logits(tensor: Any, sample_count: int, head_count: int) -> Any:
        tensor = tensor.float()
        if tensor.dim() == 2 and head_count == 1:
            tensor = tensor.reshape(1, sample_count, -1, tensor.shape[-1])
        elif tensor.dim() == 3 and tensor.shape[0] == head_count:
            tensor = tensor.reshape(head_count, sample_count, -1, tensor.shape[-1])
        elif tensor.dim() == 4 and tensor.shape[:2] == (head_count, sample_count):
            pass
        else:
            raise ValueError(
                "Unexpected RynnValue absolute-logit shape: "
                f"{tuple(tensor.shape)} for samples={sample_count}, heads={head_count}"
            )
        return tensor[:, :, -1, :].permute(1, 0, 2).contiguous()

    @staticmethod
    def _relative_last_slots(tensor: Any, sample_count: int) -> Any:
        tensor = tensor.float()
        if tensor.dim() == 1:
            tensor = tensor.reshape(sample_count, -1)
        elif tensor.dim() == 2 and tensor.shape[0] == sample_count:
            pass
        else:
            raise ValueError(
                "Unexpected RynnValue relative prediction shape: "
                f"{tuple(tensor.shape)} for samples={sample_count}"
            )
        return tensor[:, -1].contiguous()

    @staticmethod
    def _relative_last_logits(tensor: Any, sample_count: int) -> Any:
        tensor = tensor.float()
        if tensor.dim() == 2:
            tensor = tensor.reshape(sample_count, -1, tensor.shape[-1])
        elif tensor.dim() == 3 and tensor.shape[0] == sample_count:
            pass
        else:
            raise ValueError(
                "Unexpected RynnValue relative-logit shape: "
                f"{tuple(tensor.shape)} for samples={sample_count}"
            )
        return tensor[:, -1, :].contiguous()

    def predict(self, prompt: str, frames: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        absolute_values: list[np.ndarray] = []
        absolute_entropies: list[np.ndarray] = []
        absolute_logits: list[np.ndarray] = []
        relative_values: list[np.ndarray] = []
        relative_logits: list[np.ndarray] = []
        samples = []
        for end_index in range(len(frames)):
            samples.append(self._prefix_inputs(prompt, frames, end_index))
            if len(samples) < self.batch_size and end_index + 1 < len(frames):
                continue
            with self.torch.inference_mode():
                output = self.model(**self._batch_kwargs(samples))
            if output.value.pred_value is None or output.value.entropy is None:
                raise ValueError("RynnValue did not return its official absolute value outputs")
            if output.value.logits is None:
                raise ValueError("RynnValue did not return absolute value-head logits")
            if output.relative.pred_value is None or output.relative.logits is None:
                raise ValueError("RynnValue did not return its official relative value outputs")
            count = len(samples)
            absolute_values.append(
                self._absolute_last_slots(
                    output.value.pred_value, count, self.value_head_count
                ).detach().cpu().numpy()
            )
            absolute_entropies.append(
                self._absolute_last_slots(
                    output.value.entropy, count, self.value_head_count
                ).detach().cpu().numpy()
            )
            absolute_logits.append(
                self._absolute_last_logits(
                    output.value.logits, count, self.value_head_count
                ).detach().cpu().numpy()
            )
            relative_values.append(
                self._relative_last_slots(output.relative.pred_value, count)
                .detach().cpu().numpy()
            )
            relative_logits.append(
                self._relative_last_logits(output.relative.logits, count)
                .detach().cpu().numpy()
            )
            samples.clear()
        return {
            "absolute_temporal_distance_seconds": np.concatenate(absolute_values).astype(np.float32),
            "absolute_value_entropy_nats": np.concatenate(absolute_entropies).astype(np.float32),
            "absolute_value_logits": np.concatenate(absolute_logits).astype(np.float32),
            "relative_temporal_distance_seconds": np.concatenate(relative_values).astype(np.float32),
            "relative_value_logits": np.concatenate(relative_logits).astype(np.float32),
        }

    def analyze(self, prompt: str, frames: Sequence[np.ndarray]) -> dict[str, Any]:
        inputs = self._prefix_inputs(prompt, frames, len(frames) - 1)
        kwargs = self._batch_kwargs([inputs])
        input_ids = kwargs["input_ids"]
        kwargs.update({
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        })
        eos = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        kwargs["eos_token_id"] = eos
        kwargs["pad_token_id"] = eos
        with self.torch.inference_mode():
            generated = self.model.generate(**kwargs)
        text = self.processor.tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        def match(pattern: str) -> str | None:
            found = re.search(pattern, text, flags=re.IGNORECASE)
            return found.group(1).strip() if found else None

        return {
            "generated_text": text,
            "generated_token_ids": generated[0, input_ids.shape[1]:]
            .detach().cpu().to(self.torch.int64).tolist(),
            # Parsing is explicitly display-only. The exact official generation
            # above remains the persisted source of truth.
            "parsed_for_display": {
                "description": match(r"-\s*Video Description:\s*(.+)"),
                "match": match(r"-\s*Match:\s*(Yes|No)"),
                "success": match(r"-\s*Success:\s*(Yes|No)"),
            },
        }


def validate_official_outputs(
    outputs: dict[str, np.ndarray], boundary_count: int
) -> dict[str, np.ndarray]:
    required = {
        "absolute_temporal_distance_seconds": 2,
        "absolute_value_entropy_nats": 2,
        "absolute_value_logits": 3,
        "relative_temporal_distance_seconds": 1,
        "relative_value_logits": 2,
    }
    if not isinstance(outputs, dict):
        raise TypeError("RynnValue annotator must return the complete official output mapping")
    if set(outputs) != set(required):
        raise ValueError(
            "RynnValue official output keys do not match the recorded contract: "
            f"{sorted(outputs)}"
        )
    normalized: dict[str, np.ndarray] = {}
    for name, dimensions in required.items():
        value = np.asarray(outputs[name], dtype=np.float32)
        if value.ndim != dimensions or value.shape[0] != boundary_count:
            raise ValueError(
                f"RynnValue {name} shape {value.shape} does not match "
                f"{boundary_count} boundaries and {dimensions} dimensions"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"RynnValue {name} contains NaN or Inf")
        normalized[name] = value
    if normalized["absolute_temporal_distance_seconds"].shape != normalized[
        "absolute_value_entropy_nats"
    ].shape:
        raise ValueError("RynnValue absolute distance/entropy head shapes do not match")
    if normalized["absolute_value_logits"].shape[:2] != normalized[
        "absolute_temporal_distance_seconds"
    ].shape:
        raise ValueError("RynnValue absolute logits do not match decoded head outputs")
    return normalized


def sparse_macro_reward(done: np.ndarray, start: int, length: int) -> float:
    """Return the paper sparse reward for one action-chunk transition."""
    terminal = start + length - 1
    if length < 1 or start < 0 or terminal >= len(done):
        raise ValueError(
            f"Invalid macro-action interval [{start}, {start + length}) for {len(done)} actions"
        )
    return 0.0 if bool(done[terminal]) else -1.0


def sparse_primitive_return(
    done: np.ndarray, start: int, length: int, gamma: float,
) -> float:
    """Return the discounted primitive-step costs inside one chunk."""
    end = start + length
    if length < 1 or start < 0 or end > len(done):
        raise ValueError(
            f"Invalid action-chunk interval [{start}, {end}) for {len(done)} actions"
        )
    return float(sum(
        (gamma ** offset) * (0.0 if bool(done[start + offset]) else -1.0)
        for offset in range(length)
    ))


def chunk_reward_components(
    done: np.ndarray,
    start: int,
    length: int,
    value_start: float,
    value_end: float,
    gamma: float,
    shaping_weight: float,
    accumulate_primitive_steps: bool = False,
    rynnvalue: bool = True,
) -> tuple[float, float, float]:
    """Return sparse, raw PBRS shape, and final rewards for one chunk."""
    sparse = (
        sparse_primitive_return(done, start, length, gamma)
        if accumulate_primitive_steps
        else sparse_macro_reward(done, start, length)
    )
    phi_start, phi_end = -float(value_start), -float(value_end)
    duration = length if accumulate_primitive_steps else 1
    pbrs_shaping = (gamma ** duration) * phi_end - phi_start
    dense = shaping_weight * pbrs_shaping if rynnvalue else 0.0
    return sparse, pbrs_shaping, sparse + dense


def _official_inference_config_matches(
    previous: dict[str, Any], requested: dict[str, Any],
) -> bool:
    return all(previous.get(key) == requested.get(key) for key in OFFICIAL_INFERENCE_CONFIG_KEYS)


def _reusable_manifest_entry(
    episode: dict[str, Any], reward_cfg: dict[str, Any],
    annotation_manifest: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
    """Reuse official heads from a previous annotation manifest in this work dir."""
    if not isinstance(annotation_manifest, dict):
        return None
    previous_cfg = annotation_manifest.get("annotation_config")
    if not isinstance(previous_cfg, dict):
        # v4/v5 reward manifests stored inference and reduction settings together.
        previous_cfg = annotation_manifest.get("reward_config")
    if not isinstance(previous_cfg, dict) or not _official_inference_config_matches(
        previous_cfg, reward_cfg,
    ):
        return None
    entry = next(
        (item for item in annotation_manifest.get("episodes", [])
         if item.get("run_id") == episode["run_id"]),
        None,
    )
    if not isinstance(entry, dict):
        return None
    path = Path(str(entry.get("annotation_path") or ""))
    if (
        path.is_symlink() or not path.is_file()
        or entry.get("annotation_sha256") != sha256_file(path)
    ):
        return None
    official_metadata = entry.get("official_outputs")
    annotator_metadata = entry.get("annotator") or annotation_manifest.get("annotator")
    if not isinstance(official_metadata, dict) or not isinstance(annotator_metadata, dict):
        return None
    expected_key = stable_hash({
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "run": episode["run_id"], "trajectory": episode["trajectory_sha256"],
        "observations": episode["observations_sha256"], "prompt": episode["prompt"],
        "boundaries": episode["reward_boundaries"],
        "annotation_config": official_inference_config(reward_cfg), "model": annotator_metadata,
    })
    if entry.get("source_key") != expected_key:
        # Shared versions may have equal run IDs / boundary counts but different
        # source files. Legacy sidecars retain their own checked migration path.
        return None
    if official_metadata.get("prefix_image_slots") != int(reward_cfg["max_frames"]):
        return None
    expected_boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
    try:
        with np.load(path, allow_pickle=False) as arrays:
            boundaries = np.asarray(arrays["boundary_steps"], dtype=np.int64)
            if not np.array_equal(boundaries, expected_boundaries):
                return None
            official_outputs = validate_official_outputs(
                {name: arrays[name] for name in OFFICIAL_OUTPUT_KEYS}, len(boundaries),
            )
    except (KeyError, OSError, ValueError):
        return None
    analysis = official_metadata.get("analysis")
    if not isinstance(analysis, dict):
        return None
    return official_outputs, analysis, annotator_metadata


def _reusable_official_sidecar(
    episode: dict[str, Any], reward_cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
    """Load hash-checked v4/v5 official heads without running RynnValue again."""
    trajectory = Path(episode["trajectory_path"])
    sidecar = trajectory.parent / "rynnvalue_evaluation.json"
    values = trajectory.parent / "rynnvalue_evaluation.npz"
    if not sidecar.is_file() or sidecar.is_symlink() or not values.is_file() or values.is_symlink():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") not in REUSABLE_OFFICIAL_OUTPUT_SCHEMA_VERSIONS
        or payload.get("run_id") != episode["run_id"]
        or payload.get("trajectory_sha256") != episode["trajectory_sha256"]
        or payload.get("observations_sha256") != episode["observations_sha256"]
        or payload.get("values_sha256") != sha256_file(values)
    ):
        return None
    annotator_metadata = payload.get("annotator")
    official_metadata = payload.get("official_outputs")
    if not isinstance(annotator_metadata, dict) or not isinstance(official_metadata, dict):
        return None
    requested_revision = annotator_metadata.get("requested_revision")
    resolved_revision = annotator_metadata.get("resolved_revision")
    previous_inference_cfg = payload.get("annotation_config")
    if not isinstance(previous_inference_cfg, dict):
        previous_inference_cfg = payload.get("reward_config")
    if not isinstance(previous_inference_cfg, dict):
        return None
    if (
        annotator_metadata.get("model") != reward_cfg.get("model")
        or reward_cfg.get("revision") not in {requested_revision, resolved_revision}
        or official_metadata.get("prefix_image_slots") != int(reward_cfg["max_frames"])
        or previous_inference_cfg.get("robot_description") != reward_cfg.get("robot_description")
        or previous_inference_cfg.get("camera_description") != reward_cfg.get("camera_description")
    ):
        return None
    expected_boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
    try:
        with np.load(values, allow_pickle=False) as arrays:
            boundaries = np.asarray(arrays["boundary_steps"], dtype=np.int64)
            if not np.array_equal(boundaries, expected_boundaries):
                return None
            official_outputs = validate_official_outputs(
                {name: arrays[name] for name in OFFICIAL_OUTPUT_KEYS}, len(boundaries)
            )
    except (KeyError, OSError, ValueError):
        return None
    analysis = official_metadata.get("analysis")
    if not isinstance(analysis, dict):
        return None
    return official_outputs, analysis, annotator_metadata


def shaped_chunk_reward(
    done: np.ndarray,
    start: int,
    length: int,
    value_start: float,
    value_end: float,
    gamma: float,
    shaping_weight: float,
    accumulate_primitive_steps: bool = False,
) -> float:
    return chunk_reward_components(
        done, start, length, value_start, value_end, gamma, shaping_weight,
        accumulate_primitive_steps,
    )[2]


def annotate_manifest(
    config: LoadedConfig,
    annotator: TemporalValueAnnotator | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    manifest = load_manifest(config)
    reward_cfg = config.section("reward")
    annotation_cfg = official_inference_config(reward_cfg)
    active_annotator = annotator
    manifest_annotator_metadata: dict[str, Any] | None = None

    def live_annotator() -> TemporalValueAnnotator:
        nonlocal active_annotator
        if active_annotator is None:
            active_annotator = RynnValueAnnotator(config)
        return active_annotator

    work_dir = Path(config.section("paths")["work_dir"])
    annotation_dir = work_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = None
    for previous_manifest_path in (
        annotation_dir / "annotation_manifest.json",
        work_dir / "rewards" / "reward_manifest.json",
    ):
        try:
            candidate = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate.get("dataset_sha256") == manifest["dataset_sha256"]:
            previous_manifest = candidate
            break
    if (
        not isinstance(previous_manifest, dict)
        or previous_manifest.get("dataset_sha256") != manifest["dataset_sha256"]
    ):
        previous_manifest = None
    cache_dir = Path(config.section("paths")["annotation_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        reusable = None
        if annotator is None and not overwrite:
            reusable = _reusable_manifest_entry(
                episode, reward_cfg, previous_manifest,
            ) or _reusable_official_sidecar(episode, reward_cfg)
        episode_annotator_metadata = (
            reusable[2] if reusable is not None else live_annotator().metadata
        )
        if manifest_annotator_metadata is None:
            manifest_annotator_metadata = episode_annotator_metadata
        elif stable_hash(manifest_annotator_metadata) != stable_hash(episode_annotator_metadata):
            # Every reusable episode has already passed the requested model /
            # revision / inference-contract checks.  Do not discard valuable
            # evaluations merely because provenance such as snapshot paths or
            # package metadata differs across machines.  Per-episode metadata
            # remains authoritative when the dataset contains mixed provenance.
            manifest_annotator_metadata = {}
        source_key = stable_hash({
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "run": episode["run_id"],
            "trajectory": episode["trajectory_sha256"],
            "observations": episode["observations_sha256"],
            "prompt": episode["prompt"],
            "boundaries": episode["reward_boundaries"],
            "annotation_config": annotation_cfg,
            "model": episode_annotator_metadata,
        })
        output = cache_dir / f"{source_key}.npz"
        meta_path = cache_dir / f"{source_key}.json"
        if not overwrite and meta_path.is_file():
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            cached_output = Path(str(current.get("annotation_path") or output)) if isinstance(current, dict) else output
            if (
                isinstance(current, dict)
                and current.get("source_key") == source_key
                and not cached_output.is_symlink() and cached_output.is_file()
                and current.get("annotation_sha256") == sha256_file(cached_output)
            ):
                index.append(current)
                continue
        boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
        if reusable is not None:
            official_outputs, analysis, _ = reusable
            LOG.info("Reusing official RynnValue outputs for %s", episode["run_id"])
        else:
            current_annotator = live_annotator()
            with np.load(episode["observations_path"], allow_pickle=False) as images:
                raw = images["agentview_image"]
                frames = [
                    reward_view(raw[int(index)], episode["observation_orientation"])
                    for index in boundaries
                ]
            official_outputs = validate_official_outputs(
                current_annotator.predict(episode["prompt"], frames), len(frames)
            )
            analysis = current_annotator.analyze(episode["prompt"], frames)
        # Keep previously pinned official outputs intact on forced evaluation.
        output = cache_dir / f"{source_key}-{uuid.uuid4().hex}.npz"
        temporary = cache_dir / f".{output.name}.{os.getpid()}.npz"
        try:
            np.savez_compressed(
                temporary,
                boundary_steps=boundaries,
                **official_outputs,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "run_id": episode["run_id"], "source_key": source_key,
            "annotation_path": str(output.resolve()), "annotation_sha256": sha256_file(output),
            "environment_success": episode["success"],
            "annotator": episode_annotator_metadata,
            "official_outputs": {
                "inference_method": "prefix_uniform_last_slot",
                "prefix_image_slots": int(reward_cfg["max_frames"]),
                "absolute_slot": "last_value_slot_of_each_uniformly_resampled_prefix",
                "relative_slot": "between_last_two_slots_of_each_uniformly_resampled_prefix",
                "boundary_count": len(boundaries),
                "array_keys": sorted(official_outputs),
                "analysis": analysis,
            },
        }
        atomic_json(meta_path, metadata)
        index.append(metadata)
        LOG.info("Annotated %s (%d boundaries)", episode["run_id"], len(boundaries))
        atomic_json(annotation_dir / "annotation_manifest.json", {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "kind": "rynnvalue_annotation",
            "dataset_sha256": manifest["dataset_sha256"],
            "annotation_config": annotation_cfg,
            "annotator": manifest_annotator_metadata or {},
            "complete": False, "episodes": index,
        })
    index_path = annotation_dir / "annotation_manifest.json"
    atomic_json(index_path, {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "kind": "rynnvalue_annotation",
        "dataset_sha256": manifest["dataset_sha256"],
        "annotation_config": annotation_cfg,
        "annotator": manifest_annotator_metadata or {},
        "complete": True, "episodes": index,
    })
    return index_path


def load_annotation_index(config: LoadedConfig) -> dict[str, Any]:
    """Load official RynnValue outputs, migrating v4/v5 caches without a new forward."""
    manifest = load_manifest(config)
    path = (
        Path(config.section("paths")["work_dir"])
        / "annotations" / "annotation_manifest.json"
    )
    if not path.is_file():
        LOG.info("No separated annotation manifest; migrating reusable RynnValue outputs")
        path = annotate_manifest(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_config = official_inference_config(config.section("reward"))
    if (
        payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or payload.get("kind") != "rynnvalue_annotation"
        or payload.get("complete") is not True
        or payload.get("dataset_sha256") != manifest["dataset_sha256"]
        or payload.get("annotation_config") != expected_config
    ):
        raise ValueError(
            "RynnValue annotation cache does not match the prepared data or inference "
            "configuration; run annotate_rewards.py. Reward-only settings do not require "
            "another model evaluation."
        )
    if len(payload.get("episodes", [])) != len(manifest.get("episodes", [])):
        raise ValueError("RynnValue annotation manifest does not cover every episode")
    for episode in payload["episodes"]:
        annotation_path = Path(str(episode.get("annotation_path") or ""))
        if (
            annotation_path.is_symlink()
            or not annotation_path.is_file()
            or episode.get("annotation_sha256") != sha256_file(annotation_path)
        ):
            raise ValueError(
                f"RynnValue annotation cache is missing or corrupted for {episode.get('run_id')}"
            )
    return payload


def materialize_reward_manifest(config: LoadedConfig, *, force: bool = False) -> Path:
    """Build the cheap reward cache from immutable official RynnValue outputs."""
    if config.section("reward").get("manifest_path"):
        # Explicit version consumers must never mutate or regenerate their input.
        index = load_pinned_reward_index(config)
        return Path(index.get("training_adaptation_manifest_path") or config.section("reward")["manifest_path"])
    if reward_source(config.section("reward")) != "rynnvalue":
        return _materialize_direct_rewards(config, force=force)
    manifest = load_manifest(config)
    annotation_index = load_annotation_index(config)
    reward_cfg = config.section("reward")
    derivation_cfg = reward_derivation_config(reward_cfg)
    implementation = reward_implementation_fingerprint("rynnvalue")
    reward_dir = Path(config.section("paths")["work_dir"]) / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    index_path = reward_dir / "reward_manifest.json"

    if not force and index_path.is_file():
        try:
            current = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if (
            isinstance(current, dict)
            and current.get("schema_version") == REWARD_SCHEMA_VERSION
            and current.get("kind") == "derived_iql_reward"
            and current.get("complete") is True
            and current.get("dataset_sha256") == manifest["dataset_sha256"]
            and current.get("annotation_manifest_sha256") == stable_hash(annotation_index)
            and current.get("reward_config") == derivation_cfg
            and current.get("derivation_implementation_sha256") == implementation
            and len(current.get("episodes", [])) == len(manifest.get("episodes", []))
        ):
            valid = True
            for episode in current["episodes"]:
                reward_path = Path(str(episode.get("reward_path") or ""))
                if (
                    reward_path.is_symlink()
                    or not reward_path.is_file()
                    or episode.get("reward_sha256") != sha256_file(reward_path)
                ):
                    valid = False
                    break
            if valid:
                return index_path

    generation_dir = _reward_generation_dir(reward_dir)
    annotations = {
        str(item["run_id"]): item for item in annotation_index["episodes"]
    }
    index: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        annotation = annotations.get(str(episode["run_id"]))
        if annotation is None:
            raise KeyError(f"RynnValue annotation is missing for {episode['run_id']}")
        annotation_path = Path(annotation["annotation_path"])
        expected_boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
        with np.load(annotation_path, allow_pickle=False) as source:
            boundaries = np.asarray(source["boundary_steps"], dtype=np.int64)
            if not np.array_equal(boundaries, expected_boundaries):
                raise ValueError(
                    f"RynnValue annotation boundaries do not match {episode['run_id']}"
                )
            official_outputs = validate_official_outputs(
                {name: source[name] for name in OFFICIAL_OUTPUT_KEYS}, len(boundaries)
            )
        absolute = official_outputs["absolute_temporal_distance_seconds"]
        if absolute.shape[1] != 1:
            raise ValueError(
                "The pinned RynnValue reward path expects one official absolute value head; "
                f"got {absolute.shape[1]}. Raw heads were not averaged."
            )
        value_lookup = {
            int(boundary): float(value)
            for boundary, value in zip(boundaries, absolute[:, 0])
        }
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        evaluation_chunks = episode.get("evaluation_chunks", episode["chunks"])
        components = [
            chunk_reward_components(
                done,
                int(chunk["start"]),
                int(chunk["length"]),
                value_lookup[int(chunk["start"])],
                value_lookup[int(chunk["end"])],
                float(derivation_cfg["gamma"]),
                float(derivation_cfg["shaping_weight"]),
                bool(derivation_cfg["accumulate_primitive_steps"]),
                bool(derivation_cfg["rynnvalue"]),
            )
            for chunk in evaluation_chunks
        ]
        sparse = np.asarray([item[0] for item in components], dtype=np.float32)
        shaping = np.asarray([item[1] for item in components], dtype=np.float32)
        final = np.asarray([item[2] for item in components], dtype=np.float32)
        reward_key = stable_hash({
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "derivation_implementation_sha256": implementation,
            "run_id": episode["run_id"],
            "annotation_sha256": annotation["annotation_sha256"],
            "chunks": evaluation_chunks,
            "terminal_step": episode["terminal_step"],
            "reward_config": derivation_cfg,
        })
        output = generation_dir / f"{reward_key}.npz"
        temporary = generation_dir / f".{reward_key}.{os.getpid()}.npz"
        try:
            # The combined file preserves the existing UI/binding contract while
            # the canonical annotation cache remains reward-agnostic.
            np.savez_compressed(
                temporary,
                boundary_steps=boundaries,
                **_episode_timeline_arrays(episode),
                **official_outputs,
                sparse_reward=sparse,
                pbrs_shaping_reward=shaping,
                dense_reward=(
                    float(derivation_cfg["shaping_weight"]) * shaping
                    if bool(derivation_cfg["rynnvalue"])
                    else np.zeros_like(shaping)
                ),
                pbrs_chunk_reward=final,
                final_reward=final,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {
            "schema_version": REWARD_SCHEMA_VERSION,
            "run_id": episode["run_id"],
            "source_key": reward_key,
            **_episode_reward_metadata(episode),
            # Compatibility: older consumers use annotation_path for the file that
            # contains final rewards and official outputs together.
            "annotation_path": str(output.resolve()),
            "annotation_sha256": sha256_file(output),
            "reward_path": str(output.resolve()),
            "reward_sha256": sha256_file(output),
            "official_annotation_path": str(annotation_path.resolve()),
            "official_annotation_sha256": annotation["annotation_sha256"],
            "environment_success": episode["success"],
            "annotator": annotation.get("annotator") or annotation_index.get("annotator") or {},
            "official_outputs": annotation.get("official_outputs") or {},
            "pbrs_reward": {
                "array_keys": [
                    "sparse_reward", "pbrs_shaping_reward", "dense_reward",
                    "pbrs_chunk_reward",
                ],
                "description": "Deterministic reward cache derived without RynnValue forward",
                **derivation_cfg,
            },
        }
        index.append(metadata)
        atomic_json(index_path, {
            "schema_version": REWARD_SCHEMA_VERSION,
            "kind": "derived_iql_reward",
            "dataset_sha256": manifest["dataset_sha256"],
            "annotation_manifest_sha256": stable_hash(annotation_index),
            "annotation_config": annotation_index["annotation_config"],
            "reward_config": derivation_cfg,
            "derivation_implementation_sha256": implementation,
            "annotator": annotation_index.get("annotator") or {},
            "complete": False,
            "episodes": index,
        })
    atomic_json(index_path, {
        "schema_version": REWARD_SCHEMA_VERSION,
        "kind": "derived_iql_reward",
        "dataset_sha256": manifest["dataset_sha256"],
        "annotation_manifest_sha256": stable_hash(annotation_index),
        "annotation_config": annotation_index["annotation_config"],
        "reward_config": derivation_cfg,
        "derivation_implementation_sha256": implementation,
        "annotator": annotation_index.get("annotator") or {},
        "complete": True,
        "episodes": index,
    })
    # The top-level manifest is the CLI's mutable cache pointer. Each generation
    # also has an immutable copy suitable for explicit training version pinning.
    atomic_json(generation_dir / "reward_manifest.json", json.loads(index_path.read_text()))
    LOG.info(
        "Materialized %d reward arrays from cached RynnValue outputs (%s)",
        len(index), derivation_cfg,
    )
    return index_path


def load_reward_index(config: LoadedConfig) -> dict[str, Any]:
    """Load or cheaply rebuild rewards for the active training semantics."""
    if config.section("reward").get("manifest_path"):
        return load_pinned_reward_index(config)
    path = materialize_reward_manifest(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("Derived reward manifest is incomplete")
    return payload


def load_pinned_reward_index(config: LoadedConfig) -> dict[str, Any]:
    """Read fixed semantic rewards; adapt only training discount/reduction privately.

    We intentionally do not compare the current formula implementation: an old
    version denotes its saved numbers, even after code or live keyframes change.
    """
    reward = config.section("reward")
    path = Path(reward["manifest_path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != reward["manifest_sha256"]:
        raise ValueError("Pinned reward manifest is missing or its hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = load_manifest(config)
    if (payload.get("schema_version") != REWARD_SCHEMA_VERSION
            or payload.get("kind") != "derived_iql_reward" or payload.get("complete") is not True):
        raise ValueError("Pinned reward version is incomplete or unsupported")
    if payload.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("Pinned reward version belongs to a different prepared dataset")
    requested_recipe = reward_derivation_config(reward)
    stored_recipe = payload.get("reward_config")
    if not isinstance(stored_recipe, dict) or {
        key: value for key, value in stored_recipe.items() if key not in TRAINING_REDUCTION_CONFIG_KEYS
    } != {
        key: value for key, value in requested_recipe.items() if key not in TRAINING_REDUCTION_CONFIG_KEYS
    }:
        raise ValueError("Training reward parameters conflict with the pinned reward version")
    if (reward_source(reward) == "rynnvalue"
            and payload.get("annotation_config") != official_inference_config(reward)):
        raise ValueError("Training model-evaluation parameters conflict with the pinned reward version")
    if payload.get("version_id", reward["version_id"]) != reward["version_id"]:
        raise ValueError("Pinned reward version ID does not match")
    expected = {str(item["run_id"]): item for item in manifest["episodes"]}
    entries = payload.get("episodes", [])
    if (not isinstance(entries, list) or len(entries) != len(expected)
            or {str(item.get("run_id")) for item in entries} != set(expected)):
        raise ValueError("Pinned reward members do not match the prepared dataset")
    for entry in entries:
        episode = expected[str(entry["run_id"])]
        for path_key, hash_key in (("trajectory_path", "trajectory_sha256"),
                                   ("observations_path", "observations_sha256")):
            source_path = Path(episode[path_key])
            if (not source_path.is_file() or sha256_file(source_path) != episode[hash_key]):
                raise ValueError(f"Pinned reward source data changed: {episode['run_id']} ({path_key})")
        values_path = Path(entry["reward_path"])
        if (values_path.is_symlink() or not values_path.is_file()
                or sha256_file(values_path) != entry["reward_sha256"]):
            raise ValueError(f"Pinned reward arrays are missing or corrupted: {episode['run_id']}")
        # Legacy manifest adapters may lack this metadata, but current versions
        # record all indices so even equal-size, wrongly ordered chunks fail.
        for name, value in _episode_reward_metadata(episode).items():
            if name in entry and entry[name] != value:
                raise ValueError(f"Pinned reward {name} mismatch: {episode['run_id']}")
        with np.load(values_path, allow_pickle=False) as arrays:
            final = arrays["final_reward"] if "final_reward" in arrays else arrays["pbrs_chunk_reward"]
            chunks = episode.get("evaluation_chunks", episode["chunks"])
            if final.shape != (len(chunks),) or not np.isfinite(final).all():
                raise ValueError(f"Pinned reward chunk values are invalid: {episode['run_id']}")
            if not np.array_equal(arrays["boundary_steps"], episode["reward_boundaries"]):
                raise ValueError(f"Pinned reward boundary mismatch: {episode['run_id']}")
        if entry.get("annotation_path") != entry["reward_path"]:
            raise ValueError("Pinned reward replay alias must reference the validated reward arrays")
    snapshot_path = payload.get("stage_annotations_path")
    if snapshot_path:
        snapshot = Path(snapshot_path)
        if (snapshot.is_symlink() or not snapshot.is_file()
                or stable_hash(json.loads(snapshot.read_text())) != payload.get("stage_annotations_sha256")):
            raise ValueError("Pinned Stage keyframe snapshot is missing or corrupted")
    if stored_recipe != requested_recipe:
        return _adapt_pinned_training_rewards(config, payload, manifest, requested_recipe)
    return payload


def _adapt_pinned_training_rewards(config: LoadedConfig, pinned: dict[str, Any],
                                  prepared: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """Change gamma/cumulative using saved signals, never rerun a semantic evaluator."""
    from .stage_rewards import stage_chunk_reward

    work = Path(config.section("paths")["work_dir"]).resolve()
    target_root = (Path(config.section("paths")["output_dir"]) / "reward_adaptations").resolve()
    protected = [work, Path(config.section("reward")["manifest_path"]).resolve().parent]
    if (work.parent / "version.json").is_file():
        protected.append(work.parent)
    if any(target_root.is_relative_to(path) or path.is_relative_to(target_root) for path in protected):
        raise ValueError("Training reward adaptation output must be outside the sealed evaluation directory")
    source = reward_source(config.section("reward"))
    gamma, cumulative = float(recipe["gamma"]), bool(recipe["accumulate_primitive_steps"])
    implementation = reward_implementation_fingerprint(source)
    entries = {str(item["run_id"]): item for item in pinned["episodes"]}
    generated = []
    # Validate all saved semantic signals before creating an output directory.
    for episode in prepared["episodes"]:
        entry = entries[str(episode["run_id"])]
        with np.load(entry["reward_path"], allow_pickle=False) as values:
            arrays = {name: values[name] for name in values.files}
        chunks = episode.get("evaluation_chunks", episode["chunks"])
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        if source == "stage":
            scores = arrays.get("stage_score")
            if scores is None or scores.shape != (len(done) + 1,) or not np.isfinite(scores).all():
                raise ValueError(f"Pinned Stage score timeline is unavailable: {episode['run_id']}")
            final = np.asarray([
                stage_chunk_reward(scores, int(chunk["start"]), int(chunk["length"]), gamma, cumulative)
                for chunk in chunks
            ], dtype=np.float32)
            arrays["stage_chunk_reward"] = final
        elif source == "rynnvalue":
            distances = arrays.get("absolute_temporal_distance_seconds")
            boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
            if (distances is None or distances.shape != (len(boundaries), 1)
                    or not np.isfinite(distances).all()):
                raise ValueError(f"Pinned RynnValue temporal-distance timeline is unavailable: {episode['run_id']}")
            lookup = dict(zip(boundaries.tolist(), distances[:, 0].tolist()))
            components = np.asarray([
                chunk_reward_components(done, int(chunk["start"]), int(chunk["length"]),
                    lookup[int(chunk["start"])], lookup[int(chunk["end"])], gamma,
                    float(recipe["shaping_weight"]), cumulative, True)
                for chunk in chunks
            ], dtype=np.float32)
            arrays["sparse_reward"], arrays["pbrs_shaping_reward"], final = components.T
            arrays["dense_reward"] = float(recipe["shaping_weight"]) * arrays["pbrs_shaping_reward"]
        else:
            final = np.asarray([
                sparse_primitive_return(done, int(chunk["start"]), int(chunk["length"]), gamma)
                if cumulative else sparse_macro_reward(done, int(chunk["start"]), int(chunk["length"]))
                for chunk in chunks
            ], dtype=np.float32)
            arrays["sparse_reward"] = final
        arrays.update(final_reward=final, pbrs_chunk_reward=final)
        if not np.isfinite(final).all():
            raise ValueError(f"Non-finite adapted reward: {episode['run_id']}")
        generated.append((entry, episode, arrays))

    directory = target_root / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    result = {**pinned, "reward_config": recipe, "episodes": [],
              "source_reward_manifest_sha256": config.section("reward")["manifest_sha256"],
              "source_reward_version_id": config.section("reward")["version_id"],
              "source_derivation_implementation_sha256": pinned.get("derivation_implementation_sha256"),
              "derivation_implementation_sha256": implementation,
              "training_adaptation_manifest_path": str(directory / "reward_manifest.json")}
    for entry, episode, arrays in generated:
        key = stable_hash({"source_reward_sha256": entry["reward_sha256"],
                           "run_id": episode["run_id"], "reward_config": recipe,
                           "derivation_implementation_sha256": implementation})
        path = directory / f"{key}.npz"
        np.savez_compressed(path, **arrays)
        digest = sha256_file(path)
        adapted = {**entry, "source_key": key, "reward_path": str(path), "annotation_path": str(path),
                   "reward_sha256": digest, "annotation_sha256": digest}
        if "pbrs_reward" in adapted:
            adapted["pbrs_reward"] = {**adapted["pbrs_reward"], **recipe,
                                     "description": "Training-local reduction of pinned model outputs"}
        result["episodes"].append(adapted)
    atomic_json(directory / "reward_manifest.json", result)
    LOG.info("Adapted pinned %s rewards for training: gamma=%s, cumulative=%s; source version %s",
             source, gamma, cumulative, result["source_reward_version_id"])
    return result


def load_stage_annotations(config: LoadedConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate every selected member, including validation and deduplicated prefixes."""
    from .stage_rewards import STAGE_FILENAME, validate_stage_annotation

    frozen_path = config.section("data").get("stage_annotations_manifest")
    frozen = None
    if frozen_path is not None:
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        if (not isinstance(frozen, dict) or set(frozen) != {"schema_version", "annotations"}
                or frozen["schema_version"] != 1 or not isinstance(frozen["annotations"], dict)):
            raise ValueError("Invalid frozen stage annotations manifest")
    selected: dict[str, Any] = {}
    invalid: list[str] = []
    for episode in manifest["episodes"]:
        run_id = str(episode["run_id"])
        trajectory = Path(episode["trajectory_path"])
        try:
            if frozen is None and episode.get("observation_orientation") == "vla_policy":
                raise ValueError(
                    "Stage cannot reuse labels for a CSV/video-reconstructed trajectory; "
                    "provide the original trajectory.npz, observations and bound stage_annotation.json. "
                    "Annotation hashes are never silently rebound."
                )
            payload = (frozen["annotations"][run_id] if frozen is not None else
                       json.loads((trajectory.parent / STAGE_FILENAME).read_text(encoding="utf-8")))
            actual_sha = sha256_file(trajectory)
            if actual_sha != episode["trajectory_sha256"]:
                raise ValueError("Trajectory changed after Prepare; run Prepare again")
            with np.load(trajectory, allow_pickle=False) as arrays:
                done = np.asarray(arrays["done"])
            selected[run_id] = validate_stage_annotation(
                payload, run_id=run_id, trajectory_sha256=actual_sha, done=done,
                success_consecutive_steps=int(config.section("data")["success_consecutive_steps"]),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            invalid.append(f"{run_id}: {exc}")
    if invalid:
        raise ValueError("Stage annotations missing or invalid; training stopped:\n" + "\n".join(invalid))
    return {"schema_version": 1, "annotations": selected}


def _materialize_direct_rewards(config: LoadedConfig, *, force: bool) -> Path:
    """Sparse/Stage are independent reward sources and never need VLM evaluation."""
    from .stage_rewards import stage_annotation_context, stage_chunk_reward, stage_scores

    manifest = load_manifest(config)
    derivation = reward_derivation_config(config.section("reward"))
    source = derivation["source"]
    snapshot = load_stage_annotations(config, manifest) if source == "stage" else None
    snapshot_hash = stable_hash(snapshot) if snapshot is not None else None
    directory = Path(config.section("paths")["work_dir"]) / "rewards"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "reward_manifest.json"
    identity = {
        "schema_version": REWARD_SCHEMA_VERSION, "kind": "derived_iql_reward",
        "dataset_sha256": manifest["dataset_sha256"], "reward_config": derivation,
        "stage_annotations_sha256": snapshot_hash,
        "derivation_implementation_sha256": reward_implementation_fingerprint(source),
    }
    if index_path.exists() and not force:
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
            if (all(previous.get(key) == value for key, value in identity.items())
                    and previous.get("complete") is True
                    and len(previous["episodes"]) == len(manifest["episodes"])
                    and all(not Path(item["reward_path"]).is_symlink()
                            and Path(item["reward_path"]).is_file()
                            and sha256_file(Path(item["reward_path"])) == item["reward_sha256"]
                            for item in previous["episodes"])):
                if snapshot is None or (
                    Path(previous["stage_annotations_path"]).is_file()
                    and json.loads(Path(previous["stage_annotations_path"]).read_text()) == snapshot
                ):
                    return index_path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    generation_dir = _reward_generation_dir(directory)
    snapshot_path = generation_dir / f"stage_annotations_{snapshot_hash}.json" if snapshot else None
    if snapshot_path is not None:
        atomic_json(snapshot_path, snapshot)
    index = []
    gamma = float(derivation["gamma"])
    cumulative = bool(derivation["accumulate_primitive_steps"])
    for episode in manifest["episodes"]:
        chunks = episode.get("evaluation_chunks", episode["chunks"])
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        if source == "stage":
            annotation = snapshot["annotations"][str(episode["run_id"])]
            with np.load(episode["trajectory_path"], allow_pickle=False) as trajectory:
                context = stage_annotation_context(annotation, done=trajectory["done"],
                    success_consecutive_steps=int(config.section("data")["success_consecutive_steps"]),
                    exponent=derivation["stage_exponent"])
            scores = stage_scores(context)
            final = np.asarray([stage_chunk_reward(scores, int(chunk["start"]),
                int(chunk["length"]), gamma, cumulative) for chunk in chunks], dtype=np.float32)
            arrays = {"stage_score": scores, "stage_chunk_reward": final}
            annotation_hash = annotation["annotation_sha256"]
        else:
            final = np.asarray([
                (sparse_primitive_return(done, int(chunk["start"]), int(chunk["length"]), gamma)
                 if cumulative else sparse_macro_reward(done, int(chunk["start"]), int(chunk["length"])))
                for chunk in chunks
            ], dtype=np.float32)
            arrays = {"sparse_reward": final}
            annotation_hash = None
        key = stable_hash({**identity, "run_id": episode["run_id"], "chunks": chunks,
                           "trajectory_sha256": episode["trajectory_sha256"],
                           "annotation_sha256": annotation_hash})
        output = generation_dir / f"{key}.npz"
        temporary = generation_dir / f".{key}.{os.getpid()}.npz"
        try:
            np.savez_compressed(temporary, boundary_steps=np.asarray(episode["reward_boundaries"]),
                                **_episode_timeline_arrays(episode),
                                final_reward=final, pbrs_chunk_reward=final, **arrays)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        digest = sha256_file(output)
        index.append({"run_id": episode["run_id"], "reward_path": str(output.resolve()),
                      **_episode_reward_metadata(episode),
                      "reward_sha256": digest, "annotation_path": str(output.resolve()),
                      "annotation_sha256": digest, "stage_annotation_sha256": annotation_hash,
                      "source": source, "environment_success": episode["success"]})
    atomic_json(index_path, {**identity, "complete": True, "episodes": index,
                            "stage_annotations_path": str(snapshot_path.resolve()) if snapshot_path else None})
    atomic_json(generation_dir / "reward_manifest.json", json.loads(index_path.read_text()))
    LOG.info("Materialized %d %s reward arrays without reward-model evaluation", len(index), source)
    return index_path
