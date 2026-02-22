# GPU image — matches c3-comfyui base layers for cache efficiency
# Build args
ARG MAX_JOBS=8

# Base image with CUDA runtime (same as c3-comfyui)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG MAX_JOBS

# Set CUDA architectures for building without GPUs
# 8.0=A100, 8.6=RTX30xx, 8.9=RTX40xx/L40S, 9.0=H100, 12.0=Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Install Python and required packages (matches c3-comfyui order)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace directory
WORKDIR /app

# Install PyTorch with CUDA 12.8 (matches c3-comfyui for layer caching)
RUN pip install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Pre-download models at build time (default: CustomVoice + Base)
# Override: --build-arg PRELOAD_MODELS=CustomVoice:1.7B,Base:1.7B,VoiceDesign:1.7B
ARG PRELOAD_MODELS=CustomVoice:1.7B,Base:1.7B
COPY download_models.py .
RUN PRELOAD_MODELS=${PRELOAD_MODELS} python3 download_models.py

# Application code
COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
