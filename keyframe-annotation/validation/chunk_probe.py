"""Standalone caption diagnostic; never writes platform results or annotations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "keyframe-annotation/src"))
from keyframe_annotation.config import load_config
from keyframe_annotation.data import Recording, atomic_json, sha256
from keyframe_annotation.runtime import QwenAnnotator

RUNS = ("364ad127c5ac", "6d106a62d168")
CAMERAS = ("agentview_image", "wrist_image")
SUBGOALS = "Grasp and lift the black bowl; place it on the flat stove and release it."
# Fixed before generating captions; contrasts are diagnostics, not a held-out test.
PROBES = {
    RUNS[0]: (0, 24, 48, 64, 80, 96, 112, 144, 168, 192, 208, 224, 240, 280, 360, 440, 496),
    RUNS[1]: (0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 296),
}
VARIANTS = ("video_dual", "video_main", "images_dual", "context_dual", "neutral_dual", "context_main")


def source_for(run_id):
    matches = list((ROOT / "dataset-root/projects/libero_x_vla/runs").glob(f"*/*__{run_id}/episodes/episode_000/trajectory.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one trajectory for {run_id}, found {len(matches)}")
    trajectory = matches[0]
    with np.load(trajectory, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        count = len(archive["done"])
        queries = archive["inference_query_step"].tolist()
    if meta["open_loop_steps"] != 8 or (queries and queries != list(range(0, count, 8))):
        raise ValueError("This diagnostic requires the recorded 8-step chunk convention")
    return {"run_id": run_id, "task_id": meta["task_id"], "prompt": meta["task"],
            "trajectory_path": str(trajectory), "observations_path": str(trajectory.with_name("trajectory_observations.npz")),
            "manifest_path": str(trajectory.parents[2] / "run.json"), "control_hz": None,
            "orientation": "libero_raw"}, "policy_queries" if queries else "manual_8_step_analysis_windows"


def caption_prompt(instruction, start, end, hz, variant):
    background = "" if variant == "neutral_dual" else f"Task: {instruction}\nSubgoals: {SUBGOALS}\nThese are intended goals, not observed facts.\n"
    context = "Earlier frames are context only. " if variant in ("context_dual", "context_main") else ""
    return (background + context + f"Describe only the visible action during observations {start}-{end} "
            f"({start/hz:.2f}-{end/hz:.2f} seconds) in at most 40 English words. "
            "Mention the object and its movement or lack of change. If contact or holding is unclear, say so. "
            "Do not predict what happens later. Return a short paragraph, not labels or JSON.")


def input_steps(start, end, hz, variant):
    # The entire observed clip is at most 2 seconds, including the focal chunk.
    first = max(0, end-round(2*hz)) if variant in ("context_dual", "context_main") else start
    return list(range(first, end+1))


def chunk_starts(run_id, count, variant, full_trajectory=False):
    if full_trajectory or variant == "video_dual":
        return list(range(0, count, 8))
    return list(PROBES[run_id])


def prepare_inputs(model, recording, prompt, steps, variant):
    cameras = (CAMERAS[0],) if variant in ("video_main", "context_main") else CAMERAS
    model.config = replace(model.config, cameras=cameras)
    if variant != "images_dual":
        return model._prepare_inputs(prompt, recording, steps)
    content, images = [], []
    for camera in cameras:
        for step in steps:
            content.extend([{"type": "text", "text": f"Camera {camera}, observation {step}, {step/recording.hz:.3f} seconds:"},
                            {"type": "image"}])
            images.append(recording.image(camera, step))
    content.append({"type": "text", "text": prompt})
    text = model.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return model.processor(text=[text], images=images, images_kwargs={"size": {
        "shortest_edge": model.config.image_max_pixels, "longest_edge": model.config.image_max_pixels}}, return_tensors="pt")


def contact_sheets(recording, directory):
    chunks = list(range(0, recording.count, 8))
    for offset in range(0, len(chunks), 8):
        rows = chunks[offset:offset+8]
        sheet = Image.new("RGB", (1536, 286*len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, start in enumerate(rows):
            end = min(start+8, recording.count)
            steps = (start, (start+end)//2, end)
            draw.text((4, row*286+3), f"chunk {start//8}: observations {start}-{end}; main start/middle/end | wrist start/middle/end", fill="black")
            for column, (camera, step) in enumerate((c, s) for c in CAMERAS for s in steps):
                sheet.paste(recording.image(camera, step).resize((256, 256)), (column*256, row*286+26))
        directory.mkdir(parents=True, exist_ok=True)
        sheet.save(directory / f"chunks_{offset:02d}_{offset+len(rows)-1:02d}.jpg", quality=93)


@contextmanager
def gpu_lock():
    path = ROOT / "dataset-root/projects/libero_x_vla/.gpu-task.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Platform GPU resource is busy; no process was interrupted") from exc
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["video_dual"])
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--full-trajectory", action="store_true", help="Evaluate every chunk for each selected variant, not just contrast probes")
    parser.add_argument("--limit", type=int, help="Optional smoke-test call limit; rerunning resumes compatible outputs")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs) or len(set(args.variants)) != len(args.variants):
        parser.error("Select each recording and variant only once")
    output = args.output.resolve()
    if ROOT / "tmp" not in output.parents:
        parser.error("Diagnostic output must be a dedicated directory below the workspace tmp/")
    config = replace(load_config(ROOT / "keyframe-annotation/configs/qwen3_vl.yaml"), max_new_tokens=128)
    records = []
    for run_id in args.runs:
        source, convention = source_for(run_id)
        recording = Recording(source, config)
        records.append(recording)
        manifest = {"source": source, "source_hashes": recording.hashes, "chunk_convention": convention,
                    "count": recording.count, "hz": recording.hz, "subgoals": SUBGOALS,
                    "chunks": [[s, min(s+8, recording.count)] for s in range(0, recording.count, 8)],
                    "contrast_starts": PROBES[run_id], "config": asdict(config),
                    "evaluation_starts": {v: chunk_starts(run_id, recording.count, v, args.full_trajectory) for v in args.variants},
                    "source_shapes": {c: list(x.shape) for c, x in recording.images.items()}}
        path = output / run_id / "input.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError(f"Input changed; choose a new diagnostic directory: {path}")
        atomic_json(path, manifest)
        if args.prepare_only:
            contact_sheets(recording, output / run_id / "sheets")
    if args.prepare_only:
        return
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in (Path(__file__),
        ROOT / "keyframe-annotation/src/keyframe_annotation/runtime.py",
        ROOT / "keyframe-annotation/src/keyframe_annotation/data.py"))).hexdigest()
    calls, started = 0, time.monotonic()
    with gpu_lock():
        model = QwenAnnotator(config)
        metadata = {**model.metadata, "script_sha256": code_hash}
        model_path = output / "model.json"
        if model_path.exists() and json.loads(model_path.read_text()) != metadata:
            raise ValueError("Model implementation or environment changed; use a new output directory")
        atomic_json(model_path, metadata)
        for variant in args.variants:
            for recording in records:
                run_id = recording.source["run_id"]
                starts = chunk_starts(run_id, recording.count, variant, args.full_trajectory)
                for start in starts:
                    end = min(start+8, recording.count)
                    steps = input_steps(start, end, recording.hz, variant)
                    prompt = caption_prompt(recording.source["prompt"], start, end, recording.hz, variant)
                    path = output / run_id / variant / f"{start:04d}.json"
                    identity = {"run_id": run_id, "variant": variant, "start": start, "end": end,
                                "steps": steps, "prompt": prompt, "script_sha256": code_hash}
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if any(previous.get(k) != v for k, v in identity.items()):
                            raise ValueError(f"Incompatible existing caption: {path}")
                        continue
                    if args.limit is not None and calls >= args.limit:
                        return
                    tick = time.monotonic()
                    batch = prepare_inputs(model, recording, prompt, steps, variant)
                    grids = {key: batch[key].tolist() for key in ("image_grid_thw", "video_grid_thw") if key in batch}
                    input_tokens = batch.input_ids.shape[1]
                    batch = batch.to(config.device)
                    model.torch.cuda.synchronize(config.device)
                    prefill_start = time.monotonic()
                    model.torch.cuda.reset_peak_memory_stats(config.device)
                    with model.torch.inference_mode():
                        generated = model.model.generate(**batch, do_sample=False, max_new_tokens=config.max_new_tokens,
                            repetition_penalty=config.repetition_penalty)
                    model.torch.cuda.synchronize(config.device)
                    tokens = generated[0, input_tokens:]
                    text = model.processor.decode(tokens, skip_special_tokens=True)
                    result = {**identity, "caption": text, "word_count": len(text.split()),
                              "exceeds_requested_words": len(text.split()) > 40,
                              "new_tokens": len(tokens), "hit_token_limit": len(tokens) >= config.max_new_tokens,
                              "input_tokens": input_tokens, "grids": grids,
                              "inference_seconds": time.monotonic()-prefill_start,
                              "total_seconds": time.monotonic()-tick, "peak_memory_gib": model.memory_gib()}
                    atomic_json(path, result)
                    calls += 1
                    print(json.dumps({k: result[k] for k in ("run_id", "variant", "start", "end", "caption", "total_seconds", "peak_memory_gib")}), flush=True)
                    del batch, generated, tokens
        for recording in records:
            for name, field in (("trajectory", "trajectory_path"), ("observations", "observations_path"), ("manifest", "manifest_path")):
                if sha256(Path(recording.source[field])) != recording.hashes[name]:
                    raise RuntimeError(f"Source changed during diagnostic: {field}")
        atomic_json(output / "completion.json", {"status": "COMPLETED", "variants": args.variants,
                    "run_ids": args.runs, "full_trajectory": args.full_trajectory,
                    "calls_this_invocation": calls, "elapsed_seconds": time.monotonic()-started,
                    "sources_unchanged": True, "script_sha256": code_hash})


if __name__ == "__main__":
    main()
