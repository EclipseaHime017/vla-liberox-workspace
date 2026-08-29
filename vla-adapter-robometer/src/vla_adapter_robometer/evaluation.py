from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Config


SCHEMA_VERSION = 1
ARRAY_KEYS = frozenset({"observation_steps", "time_seconds", "progress_pred", "success_probs"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_steps(observation_count: int, control_hz: float, fps: float) -> np.ndarray:
    if observation_count < 1:
        raise ValueError("At least one observation is required")
    duration = (observation_count - 1) / control_hz
    count = max(1, int(math.floor(duration * fps)) + 1)
    steps = np.rint(np.arange(count, dtype=np.float64) * control_hz / fps).astype(np.int64)
    steps = np.unique(np.clip(steps, 0, observation_count - 1))
    if steps[0] != 0:
        steps = np.insert(steps, 0, 0)
    if steps[-1] != observation_count - 1:
        steps = np.append(steps, observation_count - 1)
    return steps


def prefix_indices(end_step: int, count: int = 4) -> np.ndarray:
    return np.linspace(0, end_step, count, dtype=np.int64)


def _manifest_prompt(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompt = payload.get("task") or payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Run manifest has no task prompt: {path}")
    return prompt.strip()


class OfficialRobometerAnnotator:
    def __init__(self, config: Config):
        raw = config.raw
        root = Path(raw["paths"]["robometer_root"])
        package = root / "robometer" / "__init__.py"
        if not package.is_file():
            raise FileNotFoundError(f"Official Robometer checkout is unavailable: {root}")
        sys.path.insert(0, str(root))
        import torch
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator
        from huggingface_hub import snapshot_download
        self.torch = torch
        self.device = torch.device(raw["model"]["device"])
        if not torch.cuda.is_available():
            raise RuntimeError("Robometer BF16 evaluation requires CUDA")
        started = time.monotonic()
        snapshot = snapshot_download(
            repo_id=raw["model"]["checkpoint"], revision=raw["model"]["revision"],
        )
        self.exp_config, self.tokenizer, self.processor, self.model = load_model_from_hf(
            model_path=snapshot, device=self.device,
        )
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()
        self.collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True,
        )
        self.batch_size = int(raw["evaluation"]["batch_size"])
        self.load_seconds = time.monotonic() - started
        try:
            self.commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except Exception:
            self.commit = "unknown"
        expected_commit = raw["model"]["robometer_commit"]
        if self.commit != expected_commit:
            raise RuntimeError(
                f"Official Robometer checkout mismatch: expected {expected_commit}, got {self.commit}"
            )

    def __call__(self, frames: np.ndarray, steps: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        outputs_progress: list[float] = []
        outputs_success: list[float] = []
        samples = []
        for step in steps:
            selected = frames[prefix_indices(int(step), 4)]
            trajectory = Trajectory(
                frames=selected, frames_shape=tuple(selected.shape), task=prompt,
                id=str(int(step)), metadata={"subsequence_length": len(selected)},
                video_embeddings=None,
            )
            samples.append(ProgressSample(trajectory=trajectory, sample_type="progress"))
        for start in range(0, len(samples), self.batch_size):
            batch = self.collator(samples[start:start + self.batch_size])
            inputs = batch["progress_inputs"]
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    inputs[key] = value.to(self.device)
            loss = getattr(self.exp_config, "loss", None)
            discrete = getattr(loss, "progress_loss_type", "l2").lower() == "discrete"
            bins = getattr(loss, "progress_discrete_bins", None) or getattr(self.exp_config.model, "progress_discrete_bins", 10)
            with self.torch.inference_mode(), self.torch.autocast(
                device_type="cuda", dtype=self.torch.bfloat16,
            ):
                result = compute_batch_outputs(
                    self.model, self.tokenizer, inputs, "progress", discrete, bins,
                )
            progress = result.get("progress_pred")
            success = (result.get("outputs_success") or {}).get("success_probs")
            if progress is None or success is None:
                raise ValueError("Robometer did not return progress_pred and success_probs")
            if len(progress) != len(success):
                raise ValueError("Robometer progress/success batch length mismatch")
            for progress_row, success_row in zip(progress, success):
                def as_numpy(value: Any) -> np.ndarray:
                    if hasattr(value, "detach"):
                        value = value.detach()
                    if hasattr(value, "cpu"):
                        value = value.cpu()
                    if hasattr(value, "numpy"):
                        value = value.numpy()
                    return np.asarray(value)

                progress_values = as_numpy(progress_row)
                success_values = as_numpy(success_row)
                if progress_values.size == 0 or success_values.size == 0:
                    raise ValueError("Robometer returned an empty progress or success output")
                outputs_progress.append(float(progress_values.reshape(-1)[-1]))
                outputs_success.append(float(success_values.reshape(-1)[-1]))
        return np.asarray(outputs_progress, np.float32), np.asarray(outputs_success, np.float32)


def evaluate_selection(
    config: Config,
    annotator_factory: Callable[[Config], Any] = OfficialRobometerAnnotator,
    *,
    overwrite: bool = False,
) -> Path:
    raw = config.raw
    selection_path = raw["paths"]["selection_manifest"]
    if selection_path is None:
        raise ValueError("paths.selection_manifest must be set")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    output = Path(raw["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    values_dir = output / "values"
    values_dir.mkdir(exist_ok=True)
    members = selection.get("members") or []
    if not members:
        raise ValueError("Selection contains no trajectories")
    annotator = annotator_factory(config)
    episodes = []
    started = time.monotonic()
    for index, member in enumerate(members, 1):
        run_id = str(member["run_id"])
        artifacts = member["artifacts"]
        trajectory = Path(artifacts["trajectory"]["path"]).resolve()
        observations = Path(artifacts["observations"]["path"]).resolve()
        manifest = Path(artifacts["manifest"]["path"]).resolve()
        target = values_dir / f"{run_id}.npz"
        metadata_path = values_dir / f"{run_id}.json"
        source_key = hashlib.sha256(json.dumps({
            "trajectory": sha256_file(trajectory), "observations": sha256_file(observations),
            "manifest": sha256_file(manifest), "config": config.digest,
        }, sort_keys=True).encode()).hexdigest()
        if not overwrite and target.is_file() and metadata_path.is_file():
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            if prior.get("source_key") == source_key and prior.get("values_sha256") == sha256_file(target):
                episodes.append(prior)
                print(f"ROBOMETER [{index}/{len(members)}] cached {run_id}", flush=True)
                continue
        with np.load(observations, allow_pickle=False) as archive:
            if "agentview_image" not in archive.files:
                raise ValueError(f"agentview_image is missing: {observations}")
            frames = archive["agentview_image"].copy()
        with np.load(trajectory, allow_pickle=False) as archive:
            times = archive["time_seconds"].astype(np.float64)
        if len(frames) != len(times):
            raise ValueError(f"Observation/time length mismatch for {run_id}")
        steps = evaluation_steps(len(frames), raw["evaluation"]["control_hz"], raw["evaluation"]["fps"])
        progress, success = annotator(frames, steps, _manifest_prompt(manifest))
        if len(progress) != len(steps) or len(success) != len(steps):
            raise ValueError(f"Robometer output length mismatch for {run_id}")
        if not np.isfinite(progress).all() or not np.isfinite(success).all():
            raise ValueError(f"Robometer returned NaN or Inf for {run_id}")
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, observation_steps=steps, time_seconds=times[steps],
            progress_pred=progress, success_probs=success,
        )
        os.replace(temporary, target)
        item = {
            "schema_version": SCHEMA_VERSION, "run_id": run_id,
            "source_key": source_key, "annotation_path": str(target),
            "values_sha256": sha256_file(target), "trajectory_sha256": sha256_file(trajectory),
            "observations_sha256": sha256_file(observations), "manifest_sha256": sha256_file(manifest),
            "sample_count": len(steps),
        }
        descriptor, metadata_temporary_name = tempfile.mkstemp(
            prefix=f".{run_id}.", suffix=".json.tmp", dir=values_dir,
        )
        os.close(descriptor)
        metadata_temporary = Path(metadata_temporary_name)
        metadata_temporary.write_text(json.dumps(item, indent=2), encoding="utf-8")
        os.replace(metadata_temporary, metadata_path)
        episodes.append(item)
        elapsed = time.monotonic() - started
        throughput = index / max(elapsed, 1e-9)
        eta = (len(members) - index) / max(throughput, 1e-9)
        memory = None
        if hasattr(annotator, "torch") and annotator.torch.cuda.is_available():
            memory = annotator.torch.cuda.max_memory_allocated() / 1024**3
        memory_text = "" if memory is None else f" | CUDA peak {memory:.2f} GiB"
        print(
            f"ROBOMETER [{index}/{len(members)}] {run_id} | {elapsed:.1f}s elapsed"
            f" | {throughput:.3f} trajectories/s | ETA {eta:.1f}s{memory_text}",
            flush=True,
        )
    result = {
        "schema_version": SCHEMA_VERSION, "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": selection.get("dataset_sha256"), "config_sha256": config.digest,
        "annotator": {
            "model": raw["model"]["checkpoint"], "revision": raw["model"]["revision"],
            "robometer_commit": getattr(annotator, "commit", "test"),
            "load_seconds": getattr(annotator, "load_seconds", None),
        },
        "evaluation_config": raw["evaluation"], "episodes": episodes,
    }
    result_path = output / "robometer_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".robometer_manifest.", suffix=".tmp", dir=output)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary_path, result_path)
    return result_path
