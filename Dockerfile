FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    sox libsox-fmt-all ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Pre-download models at build time
COPY download_models.py .
RUN python download_models.py

# Application code (no voice files — mount refs at runtime)
COPY server.py .
COPY README.md .

EXPOSE 8000

CMD ["python", "server.py"]
