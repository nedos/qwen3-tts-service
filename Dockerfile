# GPU image — matches c3-comfyui base layers for cache efficiency
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# CUDA architectures for building without GPU
# 8.0=A100, 8.6=RTX30xx, 8.9=RTX40xx/L40S, 9.0=H100, 12.0=Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    sox libsox-fmt-all ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Pre-download models at build time
COPY download_models.py .
RUN python3 download_models.py

# Application code (no voice files — mount refs at runtime)
COPY server.py .
COPY README.md .

EXPOSE 8000

CMD ["python3", "server.py"]
