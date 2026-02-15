"""Pre-download models during Docker build."""
import os

# Only download, don't load (saves build RAM)
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]

cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")

for model_id in MODELS:
    print(f"Downloading {model_id}...")
    snapshot_download(model_id, cache_dir=cache_dir)
    print(f"  Done: {model_id}")

print("All models downloaded.")
