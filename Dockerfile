# GPU image — runtime only, no compilation
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install flash-attn (official prebuilt wheel from Dao-AILab)
RUN pip install --no-cache-dir --break-system-packages \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Install TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Pre-download models at build time (default: none for ghcr, download at runtime)
# Override: --build-arg PRELOAD_MODELS=CustomVoice:1.7B,Base:1.7B,VoiceDesign:1.7B
ARG PRELOAD_MODELS=CustomVoice:1.7B,Base:1.7B
COPY download_models.py .
RUN PRELOAD_MODELS=${PRELOAD_MODELS} python3 download_models.py

# Application code
COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
