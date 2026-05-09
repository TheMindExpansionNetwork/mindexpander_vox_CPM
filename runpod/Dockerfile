FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install VoxCPM
RUN pip install --no-cache-dir \
    "torch>=2.5.0" \
    "torchaudio>=2.5.0" \
    "transformers>=4.36.2" \
    einops inflect addict wetext \
    "datasets>=3,<4" \
    huggingface-hub pydantic safetensors \
    soundfile librosa "numpy<2" accelerate \
    runpod

RUN pip install --no-cache-dir \
    "git+https://github.com/OpenBMB/VoxCPM.git"

# Copy handler
COPY runpod/handler.py /workspace/handler.py

# Set HF cache
ENV HF_HOME=/workspace/hf_cache

# RunPod entrypoint
CMD ["python", "-u", "/workspace/handler.py"]
