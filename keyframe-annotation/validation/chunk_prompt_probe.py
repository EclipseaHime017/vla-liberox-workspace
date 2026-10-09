"""Exploratory natural-question control after the preregistered chunk probe."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from chunk_probe import (ROOT, RUNS, PROBES, CAMERAS, SUBGOALS, source_for, gpu_lock,
                         load_config, Recording, QwenAnnotator, prepare_inputs, atomic_json, sha256)


def prompt(instruction, comparative=False):
    question = ("Compare the beginning and end of the clip: what changes in the gripper's motion, "
                "the target object's position and their contact?" if comparative else "What is the robot doing in this clip?")
    return f"Task: {instruction}\nSubgoals: {SUBGOALS}\n{question} Describe what you can see in at most 40 English words."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("minimal", "compare", "endpoints", "wide"),
                        default=["minimal", "compare", "endpoints", "wide"])
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / "tmp" not in output.parents:
        parser.error("Use a dedicated output directory below tmp/")
    config = replace(load_config(ROOT / "keyframe-annotation/configs/qwen3_vl.yaml"), max_new_tokens=128)
    records = [Recording(source_for(run_id)[0], config) for run_id in RUNS]
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("chunk_probe.py").read_bytes()).hexdigest()
    started = time.monotonic()
    with gpu_lock():
        model = QwenAnnotator(config)
        metadata = {**model.metadata, "script_sha256": code_hash}
        path = output / "model.json"
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError("Model/implementation changed; use a new diagnostic directory")
        atomic_json(path, metadata)
        for variant in args.variants:
            for recording in records:
                run_id = recording.source["run_id"]
                for start in PROBES[run_id]:
                    end = min(start+8, recording.count)
                    steps = ([start, end] if variant == "endpoints" else
                             list(range(max(0, end-round(recording.hz*2)) if variant == "wide" else start, end+1)))
                    question = prompt(recording.source["prompt"], variant in ("compare", "endpoints"))
                    path = output / run_id / variant / f"{start:04d}.json"
                    identity = {"run_id": run_id, "variant": variant, "start": start, "end": end,
                                "steps": steps, "prompt": question, "source_hashes": recording.hashes,
                                "script_sha256": code_hash}
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if any(previous.get(k) != v for k, v in identity.items()):
                            raise ValueError(f"Incompatible output: {path}")
                        continue
                    tick = time.monotonic()
                    batch = prepare_inputs(model, recording, question, steps,
                                           "images_dual" if variant == "endpoints" else "video_dual")
                    grids = {key: batch[key].tolist() for key in ("image_grid_thw", "video_grid_thw") if key in batch}
                    input_tokens = batch.input_ids.shape[1]
                    batch = batch.to(config.device)
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
                              "total_seconds": time.monotonic()-tick, "peak_memory_gib": model.memory_gib()}
                    atomic_json(path, result)
                    print(json.dumps({k: result[k] for k in ("run_id", "variant", "start", "caption", "total_seconds")}), flush=True)
                    del batch, generated, tokens
        for recording in records:
            for name, field in (("trajectory", "trajectory_path"), ("observations", "observations_path"), ("manifest", "manifest_path")):
                if sha256(Path(recording.source[field])) != recording.hashes[name]:
                    raise RuntimeError(f"Source changed during diagnostic: {field}")
        atomic_json(output / "completion.json", {"status": "COMPLETED", "variants": args.variants,
                    "elapsed_seconds": time.monotonic()-started, "sources_unchanged": True, "script_sha256": code_hash})


if __name__ == "__main__":
    main()
