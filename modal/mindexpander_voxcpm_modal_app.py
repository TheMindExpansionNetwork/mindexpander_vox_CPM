"""
VoxCPM2 MindExpander — OpenAI-compatible TTS endpoint on Modal.

Usage:
  # Deploy (creates persistent endpoint):
  modal deploy modal/mindexpander_voxcpm_modal_app.py

  # Or run locally (ephemeral):
  modal run modal/mindexpander_voxcpm_modal_app.py

  # Upload LoRA weights to volume first:
  modal volume put voxcpm-lora /local/path/lora_weights.safetensors /latest/lora_weights.safetensors

  # Test:
  curl -X POST https://YOUR_URL/v1/audio/speech \
    -H "Authorization: Bearer YOUR_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model":"mindexpander-voxcpm2","input":"Hello from the clone","voice":"default"}' \
    --output test.wav
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "mindexpander-voxcpm"
HF_MODEL_ID = "openbmb/VoxCPM2"

# Modal volumes
LORA_VOLUME = "voxcpm-lora"
HF_CACHE_VOLUME = "huggingface-cache"
OUTPUTS_VOLUME = "outputs"

# GPU image with VoxCPM dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "transformers>=4.36.2",
        "einops",
        "inflect",
        "addict",
        "wetext",
        "datasets>=3,<4",
        "huggingface-hub",
        "pydantic",
        "safetensors",
        "soundfile",
        "librosa",
        "numpy<2",
        "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "git+https://github.com/OpenBMB/VoxCPM.git",
    )
)

# CPU-only image for health/status endpoints
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "uvicorn")
)

app = modal.App(APP_NAME)


def _read_bearer_token():
    """Read bearer token from Modal secret or env."""
    return os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")


@app.function(
    image=cpu_image,
    cpu=0.25,
    memory=512,
    timeout=300,
    scaledown_window=60,
    min_containers=0,
)
@modal.asgi_app(label=f"{APP_NAME}-api")
def api():
    """CPU-only ASGI app that routes to health + GPU generation."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response, JSONResponse

    web_app = FastAPI(title="MindExpander VoxCPM2 TTS API")

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": HF_MODEL_ID,
            "endpoint": "mindexpander-voxcpm2",
            "lora": True,
            "openai_compatible": "/v1/audio/speech",
        }

    @web_app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "mindexpander-voxcpm2",
                    "object": "model",
                    "owned_by": "mindexpander",
                    "permission": [],
                }
            ],
        }

    @web_app.post("/v1/audio/speech")
    async def generate_speech(request: Request):
        body = await request.json()
        text = body.get("input", "")
        voice = body.get("voice", "default")
        response_format = body.get("response_format", "wav")
        speed = body.get("speed", 1.0)
        cfg_value = body.get("cfg_value", 2.0)
        inference_timesteps = body.get("inference_timesteps", 10)

        if not text:
            raise HTTPException(status_code=400, detail="input is required")

        # Call GPU class method directly (enter() runs automatically on container start)
        runner = VoxCPMRunner()
        result = await runner.generate.remote.aio(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )

        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }.get(response_format, "audio/wav")

        return Response(
            content=result["audio_bytes"],
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                "X-Duration-Seconds": str(result.get("duration_seconds", 0)),
                "X-Sample-Rate": str(result.get("sample_rate", 48000)),
            },
        )

    @web_app.post("/generate")
    async def generate_simple(request: Request):
        """Simple generation endpoint (non-OpenAI)."""
        body = await request.json()
        text = body.get("text", body.get("input", ""))
        voice = body.get("voice", "default")

        if not text:
            raise HTTPException(status_code=400, detail="text or input required")

        runner = VoxCPMRunner()
        result = await runner.generate.remote.aio(text=text, voice=voice)
        return Response(
            content=result["audio_bytes"],
            media_type="audio/wav",
        )

    return web_app


@app.cls(
    image=gpu_image,
    gpu="RTX-PRO-6000",
    timeout=600,
    scaledown_window=120,
    min_containers=0,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True),
    },
    secrets=[],
)
class VoxCPMRunner:
    """GPU-backed VoxCPM2 model runner with LoRA support."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import os as _os
        # MUST set HF cache dirs before any HF imports resolve
        _os.environ["HF_HOME"] = "/cache/hf"
        _os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
        _os.environ["HF_HUB_CACHE"] = "/cache/hf/hub"
        _os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        import torch
        from voxcpm.core import VoxCPM
        from voxcpm.model.voxcpm2 import LoRAConfig

        print(f"[VoxCPMRunner] Loading {HF_MODEL_ID} with LoRA...")
        start = time.time()

        # Find LoRA weights in volume
        lora_dir = Path(f"/cache/{LORA_VOLUME}/latest")
        lora_config_path = lora_dir / "lora_config.json"
        lora_weights_path = lora_dir / "lora_weights.safetensors"

        lora_cfg = None
        lora_path = None

        if lora_weights_path.exists():
            print(f"[VoxCPMRunner] Found LoRA weights: {lora_weights_path}")
            if lora_config_path.exists():
                with open(lora_config_path) as f:
                    cfg_dict = json.load(f)
                lora_cfg = LoRAConfig(**cfg_dict.get("lora_config", {
                    "enable_lm": True,
                    "enable_dit": True,
                    "enable_proj": False,
                    "r": 32,
                    "alpha": 16,
                }))
            else:
                # Default LoRA config matching training setup
                lora_cfg = LoRAConfig(
                    enable_lm=True,
                    enable_dit=True,
                    enable_proj=False,
                    r=32,
                    alpha=16,
                )
            lora_path = str(lora_dir)
            print(f"[VoxCPMRunner] LoRA config: r={lora_cfg.r}, alpha={lora_cfg.alpha}")
        else:
            print(f"[VoxCPMRunner] No LoRA weights found at {lora_weights_path}, loading base model only")
            # Try alternate paths
            for alt in [
                Path(f"/cache/{LORA_VOLUME}/lora_weights.safetensors"),
                Path(f"/cache/{LORA_VOLUME}/checkpoints/latest/lora_weights.safetensors"),
            ]:
                if alt.exists():
                    print(f"[VoxCPMRunner] Found LoRA at alternate path: {alt}")
                    lora_path = str(alt.parent)
                    lora_cfg = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False, r=32, alpha=16)
                    break

        self.model = VoxCPM.from_pretrained(
            hf_model_id=HF_MODEL_ID,
            load_denoiser=False,
            optimize=False,  # torch.compile can be slow on first run
            lora_config=lora_cfg,
            lora_weights_path=lora_path,
        )

        elapsed = time.time() - start
        self.sample_rate = self.model.tts_model.sample_rate
        print(f"[VoxCPMRunner] Model loaded in {elapsed:.1f}s, sample_rate={self.sample_rate}")
        print(f"[VoxCPMRunner] LoRA enabled: {self.model.lora_enabled}")

    @modal.method()
    def generate(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> dict:
        """Generate speech from text."""
        import numpy as np
        import soundfile as sf

        start = time.time()

        # Generate audio
        audio_np = self.model.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=False,
            denoise=False,
        )

        duration = len(audio_np) / self.sample_rate
        elapsed = time.time() - start
        print(f"[VoxCPMRunner] Generated {duration:.1f}s audio in {elapsed:.1f}s for: {text[:80]}...")

        # Encode to requested format
        buf = io.BytesIO()
        if response_format == "wav":
            sf.write(buf, audio_np, self.sample_rate, format="WAV")
        elif response_format == "flac":
            sf.write(buf, audio_np, self.sample_rate, format="FLAC")
        elif response_format in ("mp3", "opus"):
            # Write WAV first, then convert
            sf.write(buf, audio_np, self.sample_rate, format="WAV")
            buf.seek(0)
            wav_data = buf.read()
            # Use ffmpeg for conversion
            import subprocess
            fmt = "mp3" if response_format == "mp3" else "opus"
            proc = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", fmt, "pipe:1"],
                input=wav_data,
                capture_output=True,
            )
            if proc.returncode == 0:
                audio_bytes = proc.stdout
            else:
                # Fallback to WAV
                audio_bytes = wav_data
                response_format = "wav"
        else:
            sf.write(buf, audio_np, self.sample_rate, format="WAV")

        audio_bytes = buf.getvalue() if response_format in ("wav", "flac") else audio_bytes

        return {
            "audio_bytes": audio_bytes,
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "format": response_format,
            "generation_time_seconds": round(elapsed, 2),
        }


@app.function(
    image=gpu_image,
    gpu="RTX-PRO-6000",
    timeout=600,
    scaledown_window=120,
    min_containers=0,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True),
    },
)
def generate_audio(
    text: str,
    voice: str = "default",
    response_format: str = "wav",
    speed: float = 1.0,
) -> dict:
    """Standalone GPU function for generating audio (called by CPU web app)."""
    runner = VoxCPMRunner()
    runner.load_model()
    return runner.generate(text=text, voice=voice, response_format=response_format, speed=speed)


@app.function(
    image=gpu_image,
    gpu="RTX-PRO-6000",
    timeout=600,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
    },
)
def cache_model():
    """Pre-download VoxCPM2 model to HF cache volume. Run once after deploy."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["HF_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    from pathlib import Path
    from huggingface_hub import snapshot_download

    print(f"[cache_model] Downloading {HF_MODEL_ID} to /cache/hf/hub...")
    path = snapshot_download(
        repo_id=HF_MODEL_ID,
        cache_dir="/cache/hf/hub",
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"[cache_model] Downloaded to: {path}")

    # Verify key files
    p = Path(path)
    for f in ["model.safetensors", "audiovae.pth", "config.json"]:
        fp = p / f
        if fp.exists():
            print(f"  ✅ {f}: {fp.stat().st_size / 1024 / 1024:.1f}MB")
        else:
            print(f"  ❌ {f}: MISSING")

    # Commit volume so it persists
    vol = modal.Volume.from_name(HF_CACHE_VOLUME)
    vol.commit()
    print(f"[cache_model] Volume committed. Cold starts will now use cached model!")


@app.local_entrypoint()
def main(
    text: str = "Is this real, or did the stars just blink? Did the room start breathing when I stopped to think?",
    voice: str = "default",
    cache: bool = False,
):
    """Test the endpoint locally. Pass --cache to pre-download model."""
    if cache:
        print("Pre-caching VoxCPM2 model to volume...")
        cache_model.remote()
        print("Done! Model cached. Future cold starts will be much faster.")
        return
    print(f"Generating: {text}")
    result = generate_audio.remote(text=text, voice=voice)
    out_path = Path("/tmp/mindexpander_test.wav")
    out_path.write_bytes(result["audio_bytes"])
    print(f"Saved: {out_path} ({result['duration_seconds']}s, {result['sample_rate']}Hz)")
    print(f"Generation time: {result['generation_time_seconds']}s")
