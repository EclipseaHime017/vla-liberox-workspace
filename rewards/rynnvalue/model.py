"""The pinned official RynnValue model (checkout in third_party/RynnValue): load, validate, infer."""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import yaml
from PIL import Image

from core.config import LoadedConfig
from core.io import sha256_file
from core.paths import THIRD_PARTY


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
        lock_path = THIRD_PARTY / "lock-train.yaml"
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
