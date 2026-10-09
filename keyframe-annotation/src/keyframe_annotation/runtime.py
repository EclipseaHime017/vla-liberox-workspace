"""Optional GPU boundary. Imported only by the standalone worker."""
from __future__ import annotations

import numpy as np


class QwenAnnotator:
    def __init__(self, config):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen annotation requires CUDA; no CPU or substitute-model fallback")
        torch.set_num_threads(4)
        torch.manual_seed(config.seed)
        self.torch, self.config = torch, config
        snapshot = snapshot_download(config.model_id, revision=config.revision, cache_dir=config.cache_dir,
                                     local_files_only=config.local_files_only)
        options = dict(local_files_only=True, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(snapshot, **options)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot, **options,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(config.device).eval()
        self.metadata = {"model_id": config.model_id, "revision": config.revision,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "dtype": "bfloat16", "device": config.device,
            "visual_input": {"plan": "images", "localization": "native_video_v1",
                             "camera_streams": "separate", "resample_frames": False,
                             "timestamps": "original_observation_steps / control_hz",
                             "temporal_patch_size": self.processor.video_processor.temporal_patch_size},
            "generation": {"do_sample": False, "repetition_penalty": config.repetition_penalty}}

    def _prepare_inputs(self, prompt, recording, steps):
        content, media = [], {}
        if len(steps) > 1:
            videos, metadata = [], []
            for camera in self.config.cameras:
                content.extend([{"type": "text", "text": f"Camera {camera}; chronological video, original trajectory timestamps:"},
                                {"type": "video"}])
                videos.append(np.stack([np.asarray(recording.image(camera, step)) for step in steps]))
                # The processor may pad its own timestamp list for temporal patches.
                metadata.append({"total_num_frames": recording.count + 1, "fps": recording.hz,
                                 "frames_indices": list(steps)})
            pixels = len(steps) * self.config.image_max_pixels
            media = {"videos": videos, "videos_kwargs": {
                "video_metadata": metadata, "do_sample_frames": False,
                "input_data_format": "channels_last",
                "size": {"shortest_edge": pixels, "longest_edge": pixels}}}
        elif steps:
            images = []
            for camera in self.config.cameras:
                content.extend([{"type": "text", "text": f"Observation {steps[0]}, {steps[0] / recording.hz:.3f} s, camera {camera}:"},
                                {"type": "image"}])
                images.append(recording.image(camera, steps[0]))
            media = {"images": images, "images_kwargs": {
                "size": {"shortest_edge": self.config.image_max_pixels,
                         "longest_edge": self.config.image_max_pixels}}}
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], return_tensors="pt", **media)

    def generate(self, prompt, recording, steps):
        inputs = self._prepare_inputs(prompt, recording, steps).to(self.config.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                                          repetition_penalty=self.config.repetition_penalty)
        return self.processor.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def memory_gib(self):
        return self.torch.cuda.max_memory_allocated(self.config.device) / 1024**3
