from __future__ import annotations

import json
import importlib.metadata
import logging
import math
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


class TemporalValueAnnotator(Protocol):
    metadata: dict[str, Any]

    def predict(self, prompt: str, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]: ...


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
            import rynn_value  # noqa: F401 - registers local audited Auto classes
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoModel, AutoProcessor
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
        hf_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
        hf_config._attn_implementation = "pred_slot_isolated_eager"
        self.processor = AutoProcessor.from_pretrained(
            snapshot, trust_remote_code=False, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            snapshot, config=hf_config, torch_dtype=dtype, trust_remote_code=False,
            local_files_only=True, low_cpu_mem_usage=True,
        ).to(reward["device"]).eval()
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
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "huggingface-hub")
            },
        }
        self.robot_description = reward["robot_description"]
        self.camera_description = reward["camera_description"]
        self.max_frames = int(reward["max_frames"])
        self.batch_size = int(reward["annotation_batch_size"])

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
    def _last_slot(tensor: Any, sample_count: int) -> Any:
        # This is the shape reduction used by the pinned official inference
        # script and supports both single and ensembled value heads.
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            tensor = tensor.reshape(sample_count, -1)
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)
        if tensor.dim() == 2 and tensor.shape[-1] > 1:
            tensor = tensor[:, -1]
        elif tensor.dim() == 2:
            tensor = tensor[:, 0]
        return tensor.float().reshape(-1)

    def predict(self, prompt: str, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        values: list[float] = []
        entropies: list[float] = []
        samples = []
        for end_index in range(len(frames)):
            samples.append(self._prefix_inputs(prompt, frames, end_index))
            if len(samples) < self.batch_size and end_index + 1 < len(frames):
                continue
            with self.torch.inference_mode():
                output = self.model(**self._batch_kwargs(samples))
            predicted = self._last_slot(output.value.pred_value, len(samples))
            entropy_value = getattr(output.value, "entropy", None)
            entropy = (
                self.torch.zeros_like(predicted)
                if entropy_value is None
                else self._last_slot(entropy_value, len(samples))
            )
            if predicted.numel() != len(samples) or entropy.numel() != len(samples):
                raise ValueError("RynnValue returned an unexpected prefix batch shape")
            values.extend(predicted.detach().cpu().tolist())
            entropies.extend(entropy.detach().cpu().tolist())
            samples.clear()
        return np.asarray(values, dtype=np.float32), np.asarray(entropies, dtype=np.float32)

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
            "text": text,
            "description": match(r"-\s*Video Description:\s*(.+)"),
            "match": match(r"-\s*Match:\s*(Yes|No)"),
            "success": match(r"-\s*Success:\s*(Yes|No)"),
        }


def annotation_windows(count: int, maximum: int, overlap: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError("At least one reward boundary is required")
    if count <= maximum:
        return [(0, count)]
    stride = maximum - overlap
    starts = list(range(0, max(1, count - maximum + 1), stride))
    final = count - maximum
    if starts[-1] != final:
        starts.append(final)
    return [(start, min(count, start + maximum)) for start in starts]


def annotate_values(
    annotator: TemporalValueAnnotator,
    prompt: str,
    frames: Sequence[np.ndarray],
    maximum: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(frames), dtype=np.float64)
    entropy_sums = np.zeros(len(frames), dtype=np.float64)
    counts = np.zeros(len(frames), dtype=np.int32)
    for start, end in annotation_windows(len(frames), maximum, overlap):
        values, entropy = annotator.predict(prompt, frames[start:end])
        if values.shape != (end - start,) or entropy.shape != (end - start,):
            raise ValueError("RynnValue returned an unexpected number of temporal predictions")
        if not np.isfinite(values).all() or not np.isfinite(entropy).all():
            raise ValueError("RynnValue returned NaN or Inf")
        sums[start:end] += values
        entropy_sums[start:end] += entropy
        counts[start:end] += 1
    if np.any(counts == 0):
        raise RuntimeError("Annotation windows did not cover every boundary")
    return (sums / counts).astype(np.float32), (entropy_sums / counts).astype(np.float32)


def sparse_chunk_return(done: np.ndarray, start: int, length: int, gamma: float) -> float:
    total = 0.0
    for offset in range(length):
        # Paper convention: -1 before completion, 0 at a successful terminal.
        step_cost = 0.0 if bool(done[start + offset]) else -1.0
        total += (gamma ** offset) * step_cost
    return total


def shaped_chunk_reward(
    done: np.ndarray,
    start: int,
    length: int,
    value_start: float,
    value_end: float,
    gamma: float,
    shaping_weight: float,
) -> float:
    phi_start, phi_end = -float(value_start), -float(value_end)
    return sparse_chunk_return(done, start, length, gamma) + shaping_weight * (
        (gamma ** length) * phi_end - phi_start
    )


def annotate_manifest(config: LoadedConfig, annotator: TemporalValueAnnotator | None = None) -> Path:
    manifest = load_manifest(config)
    reward_cfg = config.section("reward")
    annotator = annotator or RynnValueAnnotator(config)
    reward_dir = Path(config.section("paths")["work_dir"]) / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        output = reward_dir / f"{episode['run_id']}.npz"
        meta_path = reward_dir / f"{episode['run_id']}.json"
        source_key = stable_hash({
            "dataset": manifest["dataset_sha256"], "run": episode["run_id"],
            "trajectory": episode["trajectory_sha256"],
            "observations": episode["observations_sha256"],
            "prompt": episode["prompt"],
            "boundaries": episode["reward_boundaries"],
            "reward_config": reward_cfg,
            "model": annotator.metadata,
        })
        if output.is_file() and meta_path.is_file():
            current = json.loads(meta_path.read_text(encoding="utf-8"))
            if current.get("source_key") == source_key:
                index.append(current)
                continue
        with np.load(episode["observations_path"], allow_pickle=False) as images:
            raw = images["agentview_image"]
            boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
            frames = [reward_view(raw[int(index)], episode["observation_orientation"]) for index in boundaries]
        values, entropy = annotate_values(
            annotator, episode["prompt"], frames,
            int(reward_cfg["max_frames"]), int(reward_cfg["window_overlap"]),
        )
        analysis = None
        analyze = getattr(annotator, "analyze", None)
        if callable(analyze):
            maximum = int(reward_cfg["max_frames"])
            if len(frames) > maximum:
                sample = np.linspace(0, len(frames) - 1, maximum, dtype=np.int64)
                analysis_frames = [frames[int(index)] for index in sample]
            else:
                analysis_frames = frames
            analysis = analyze(episode["prompt"], analysis_frames)
        value_lookup = {int(boundary): float(value) for boundary, value in zip(boundaries, values)}
        # Reward semantics use the debounced terminal from the prepared manifest,
        # never transient raw done=True samples from the immutable source trajectory.
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"])] = True
        chunk_rewards = np.asarray([
            shaped_chunk_reward(
                done, int(chunk["start"]), int(chunk["length"]),
                value_lookup[int(chunk["start"])], value_lookup[int(chunk["end"])],
                float(reward_cfg["gamma"]), float(reward_cfg["shaping_weight"]),
            ) for chunk in episode["chunks"]
        ], dtype=np.float32)
        np.savez_compressed(
            output, boundaries=boundaries, remaining_seconds=values,
            entropy=entropy, chunk_reward=chunk_rewards,
        )
        metadata = {
            "schema_version": 1, "run_id": episode["run_id"], "source_key": source_key,
            "annotation_path": str(output.resolve()), "annotation_sha256": sha256_file(output),
            "environment_success": episode["success"], "annotator": annotator.metadata,
            "analysis": analysis,
        }
        atomic_json(meta_path, metadata)
        index.append(metadata)
        LOG.info("Annotated %s (%d boundaries)", episode["run_id"], len(boundaries))
    index_path = reward_dir / "reward_manifest.json"
    atomic_json(index_path, {
        "schema_version": 1, "dataset_sha256": manifest["dataset_sha256"],
        "reward_config": reward_cfg, "annotator": annotator.metadata,
        "episodes": index,
    })
    return index_path


def load_reward_index(config: LoadedConfig) -> dict[str, Any]:
    path = Path(config.section("paths")["work_dir"]) / "rewards" / "reward_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Reward manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
