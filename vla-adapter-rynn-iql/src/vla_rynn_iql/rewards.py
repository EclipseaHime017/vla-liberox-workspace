from __future__ import annotations

import json
import importlib.metadata
import logging
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import yaml
from PIL import Image

from .config import LoadedConfig
from .data import load_manifest
from .io import atomic_json, sha256_file, stable_hash


LOG = logging.getLogger(__name__)

# Schema v5 stores the complete official RynnValue outputs and deterministic
# reward arrays. Reward semantics are selected by a boolean in reward_config;
# the schema number is not used as a reward-mode switch.
ANNOTATION_SCHEMA_VERSION = 5
REUSABLE_OFFICIAL_OUTPUT_SCHEMA_VERSIONS = frozenset({4, 5})
OFFICIAL_INFERENCE_CONFIG_KEYS = frozenset({
    "model", "revision", "dtype", "max_frames",
    "robot_description", "camera_description",
})
OFFICIAL_OUTPUT_KEYS = (
    "absolute_temporal_distance_seconds",
    "absolute_value_entropy_nats",
    "absolute_value_logits",
    "relative_temporal_distance_seconds",
    "relative_value_logits",
)


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
    return sparse, pbrs_shaping, sparse + shaping_weight * pbrs_shaping


def _official_inference_config_matches(
    previous: dict[str, Any], requested: dict[str, Any],
) -> bool:
    return all(previous.get(key) == requested.get(key) for key in OFFICIAL_INFERENCE_CONFIG_KEYS)


def _reusable_manifest_entry(
    episode: dict[str, Any], reward_cfg: dict[str, Any],
    reward_manifest: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
    """Reuse official heads from the previous reward reduction in this work dir."""
    if not isinstance(reward_manifest, dict):
        return None
    previous_cfg = reward_manifest.get("reward_config")
    if not isinstance(previous_cfg, dict) or not _official_inference_config_matches(
        previous_cfg, reward_cfg,
    ):
        return None
    entry = next(
        (item for item in reward_manifest.get("episodes", [])
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
    annotator_metadata = entry.get("annotator") or reward_manifest.get("annotator")
    if not isinstance(official_metadata, dict) or not isinstance(annotator_metadata, dict):
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
    previous_reward_cfg = payload.get("reward_config")
    if not isinstance(previous_reward_cfg, dict):
        return None
    if (
        annotator_metadata.get("model") != reward_cfg.get("model")
        or reward_cfg.get("revision") not in {requested_revision, resolved_revision}
        or official_metadata.get("prefix_image_slots") != int(reward_cfg["max_frames"])
        or previous_reward_cfg.get("robot_description") != reward_cfg.get("robot_description")
        or previous_reward_cfg.get("camera_description") != reward_cfg.get("camera_description")
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
    active_annotator = annotator
    manifest_annotator_metadata: dict[str, Any] | None = None

    def live_annotator() -> TemporalValueAnnotator:
        nonlocal active_annotator
        if active_annotator is None:
            active_annotator = RynnValueAnnotator(config)
        return active_annotator

    reward_dir = Path(config.section("paths")["work_dir"]) / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = reward_dir / "reward_manifest.json"
    try:
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_manifest = None
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
            raise ValueError("Prepared trajectories use incompatible RynnValue evaluator metadata")
        source_key = stable_hash({
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "run": episode["run_id"],
            "trajectory": episode["trajectory_sha256"],
            "observations": episode["observations_sha256"],
            "prompt": episode["prompt"],
            "boundaries": episode["reward_boundaries"],
            "reward_config": reward_cfg,
            "model": episode_annotator_metadata,
        })
        output = cache_dir / f"{source_key}.npz"
        meta_path = cache_dir / f"{source_key}.json"
        if not overwrite and output.is_file() and meta_path.is_file():
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            if (
                isinstance(current, dict)
                and current.get("source_key") == source_key
                and current.get("annotation_sha256") == sha256_file(output)
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
        # Reward semantics use the debounced terminal from the prepared manifest,
        # never transient raw done=True samples from the immutable source trajectory.
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            # The environment terminal is absorbing for reward semantics. The
            # recorded post-success tail remains available for diagnostics but
            # does not reintroduce the pre-success -1 step cost.
            done[int(episode["terminal_step"]):] = True
        reward_components = [
            chunk_reward_components(
                done, int(chunk["start"]), int(chunk["length"]),
                value_lookup[int(chunk["start"])], value_lookup[int(chunk["end"])],
                float(reward_cfg["gamma"]), float(reward_cfg["shaping_weight"]),
                bool(reward_cfg["accumulate_primitive_steps"]),
            ) for chunk in episode.get("evaluation_chunks", episode["chunks"])
        ]
        pbrs_shaping_rewards = np.asarray(
            [component[1] for component in reward_components], dtype=np.float32,
        )
        chunk_rewards = np.asarray(
            [component[2] for component in reward_components], dtype=np.float32,
        )
        temporary = cache_dir / f".{source_key}.{os.getpid()}.npz"
        try:
            np.savez_compressed(
                temporary,
                boundary_steps=boundaries,
                **official_outputs,
                pbrs_shaping_reward=pbrs_shaping_rewards,
                # Backward-compatible name: this is the combined final training reward.
                pbrs_chunk_reward=chunk_rewards,
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
            "pbrs_reward": {
                "array_keys": [
                    "pbrs_shaping_reward", "pbrs_chunk_reward",
                ],
                "description": (
                    "Unweighted RynnValue PBRS shape reward and combined final IQL reward"
                ),
                "accumulate_primitive_steps": bool(
                    reward_cfg["accumulate_primitive_steps"]
                ),
            },
        }
        atomic_json(meta_path, metadata)
        index.append(metadata)
        LOG.info("Annotated %s (%d boundaries)", episode["run_id"], len(boundaries))
        atomic_json(reward_dir / "reward_manifest.json", {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "dataset_sha256": manifest["dataset_sha256"],
            "accumulate_primitive_steps": bool(
                reward_cfg["accumulate_primitive_steps"]
            ),
            "reward_config": reward_cfg, "annotator": manifest_annotator_metadata or {},
            "complete": False, "episodes": index,
        })
    index_path = reward_dir / "reward_manifest.json"
    atomic_json(index_path, {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_sha256": manifest["dataset_sha256"],
        "accumulate_primitive_steps": bool(reward_cfg["accumulate_primitive_steps"]),
        "reward_config": reward_cfg, "annotator": manifest_annotator_metadata or {},
        "complete": True, "episodes": index,
    })
    return index_path


def load_reward_index(config: LoadedConfig) -> dict[str, Any]:
    path = Path(config.section("paths")["work_dir"]) / "rewards" / "reward_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Reward manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            "Reward manifest uses obsolete transition semantics: "
            f"expected schema v{ANNOTATION_SCHEMA_VERSION}, got "
            f"v{payload.get('schema_version')}. Re-run annotation; reusable official "
            "RynnValue outputs will be migrated without another model forward pass."
        )
    manifest_reward_cfg = payload.get("reward_config")
    if not isinstance(manifest_reward_cfg, dict):
        raise ValueError("Reward manifest has no valid reward_config")
    requested_cfg = config.section("reward")
    if manifest_reward_cfg != requested_cfg:
        if not _official_inference_config_matches(manifest_reward_cfg, requested_cfg):
            raise ValueError(
                "Reward manifest uses different RynnValue inference settings; run "
                "annotate_rewards.py before training"
            )
        LOG.info(
            "Reward semantics changed; rebuilding deterministic reward arrays from "
            "the existing RynnValue outputs"
        )
        path = annotate_manifest(config)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reward_config") != requested_cfg:
            raise ValueError("Rebuilt reward manifest does not match the requested semantics")
    if payload.get("complete") is not True:
        raise ValueError("Reward annotation manifest is incomplete")
    return payload
