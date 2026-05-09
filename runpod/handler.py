"""
RunPod Serverless Handler for VoxCPM2 MindExpander.

Deploy this as a RunPod Serverless endpoint for real-time voice generation.

Setup:
  1. Create a RunPod Serverless endpoint
  2. Set the container image to include VoxCPM dependencies
  3. Mount a network volume with LoRA weights at /workspace/checkpoints/latest/
  4. Use this handler.py as the entrypoint

Environment variables:
  HF_HOME: HuggingFace cache directory
  LORA_PATH: Override path to LoRA weights directory
"""

import io
import json
import os
import time
from pathlib import Path

import runpod

# Global model instance (loaded once per worker)
_model = None
_model_loaded = False


def load_model():
    """Load VoxCPM2 model with LoRA weights."""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    import torch
    from voxcpm.core import VoxCPM
    from voxcpm.model.voxcpm2 import LoRAConfig

    print("[RunPod] Loading VoxCPM2 model...")
    start = time.time()

    # Find LoRA weights
    lora_path = os.getenv("LORA_PATH", "/workspace/checkpoints/latest")
    lora_dir = Path(lora_path)

    lora_cfg = None
    lora_weights = None

    safetensors = lora_dir / "lora_weights.safetensors"
    config_json = lora_dir / "lora_config.json"

    if safetensors.exists():
        print(f"[RunPod] Found LoRA weights: {safetensors}")
        if config_json.exists():
            with open(config_json) as f:
                cfg = json.load(f)
            lora_cfg = LoRAConfig(**cfg.get("lora_config", {}))
        else:
            lora_cfg = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
                r=32,
                alpha=16,
            )
        lora_weights = str(lora_dir)
    else:
        print(f"[RunPod] No LoRA weights at {safetensors}, using base model")

    _model = VoxCPM.from_pretrained(
        hf_model_id="openbmb/VoxCPM2",
        load_denoiser=False,
        optimize=False,
        lora_config=lora_cfg,
        lora_weights_path=lora_weights,
    )

    elapsed = time.time() - start
    print(f"[RunPod] Model loaded in {elapsed:.1f}s, sample_rate={_model.tts_model.sample_rate}")
    print(f"[RunPod] LoRA enabled: {_model.lora_enabled}")

    _model_loaded = True
    return _model


def handler(job):
    """
    RunPod serverless handler.

    Input:
      {
        "input": {
          "text": "Hello world",
          "voice": "default",           # optional
          "response_format": "wav",     # wav, mp3, flac
          "cfg_value": 2.0,             # optional CFG scale
          "inference_timesteps": 10,    # optional diffusion steps
          "normalize": false,           # optional text normalization
        }
      }

    Output:
      {
        "audio_base64": "...",
        "duration_seconds": 3.5,
        "sample_rate": 48000,
        "format": "wav",
        "generation_time_seconds": 2.1
      }
    """
    import base64
    import soundfile as sf

    job_input = job.get("input", {})
    text = job_input.get("text", "")
    response_format = job_input.get("response_format", "wav")
    cfg_value = job_input.get("cfg_value", 2.0)
    inference_timesteps = job_input.get("inference_timesteps", 10)
    normalize = job_input.get("normalize", False)

    if not text:
        return {"error": "text is required"}

    model = load_model()
    start = time.time()

    audio_np = model.generate(
        text=text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=False,
    )

    duration = len(audio_np) / model.tts_model.sample_rate
    elapsed = time.time() - start

    # Encode audio
    buf = io.BytesIO()
    sf.write(buf, audio_np, model.tts_model.sample_rate, format=response_format.upper())
    audio_bytes = buf.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_base64": audio_b64,
        "duration_seconds": round(duration, 2),
        "sample_rate": model.tts_model.sample_rate,
        "format": response_format,
        "generation_time_seconds": round(elapsed, 2),
        "text": text,
    }


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
