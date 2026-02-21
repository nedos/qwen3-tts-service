FROM python:3.12-slim

WORKDIR /app

# System dependencies + NVIDIA runtime support
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsox-fmt-all ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 (bundles its own CUDA libs)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time
COPY download_models.py .
RUN python download_models.py

# Application code (no voice files — mount refs at runtime)
COPY server.py .
COPY README.md .

EXPOSE 8000

CMD ["python", "server.py"]
