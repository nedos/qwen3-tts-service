"""Pre-download Qwen3-TTS models.

Used at Docker build time to bake models into the image,
or at runtime to download additional models on startup.

Control via env var:
    PRELOAD_MODELS=CustomVoice:1.7B,Base:1.7B,VoiceDesign:1.7B

Default (no env): downloads CustomVoice + Base + Tokenizer (baked into image).
"""
import os
import sys
from huggingface_hub import snapshot_download

VALID_TYPES = {"CustomVoice", "Base", "VoiceDesign"}
VALID_SIZES = {"0.6B", "1.7B"}

# Always download the tokenizer
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

DEFAULT_MODELS = "CustomVoice:1.7B,Base:1.7B"


def parse_models(spec: str) -> list[tuple[str, str]]:
    """Parse 'CustomVoice:1.7B,Base:0.6B' into list of (type, size) tuples."""
    models = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(f"Warning: skipping '{entry}' (expected Type:Size)", flush=True)
            continue
        model_type, model_size = entry.split(":", 1)
        model_type = model_type.strip()
        model_size = model_size.strip()
        if model_type not in VALID_TYPES:
            print(f"Warning: unknown type '{model_type}' (valid: {VALID_TYPES})", flush=True)
            continue
        if model_size not in VALID_SIZES:
            print(f"Warning: unknown size '{model_size}' (valid: {VALID_SIZES})", flush=True)
            continue
        models.append((model_type, model_size))
    return models


def main():
    spec = os.getenv("PRELOAD_MODELS", DEFAULT_MODELS).strip()
    models = parse_models(spec)

    if not models:
        print("No models to download.", flush=True)
        return

    cache_dir = os.environ.get("HF_HOME", None)

    # Tokenizer first
    print(f"Downloading tokenizer: {TOKENIZER}", flush=True)
    snapshot_download(TOKENIZER, cache_dir=cache_dir)

    success = 0
    for model_type, model_size in models:
        repo_id = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
        print(f"Downloading: {repo_id}", flush=True)
        try:
            path = snapshot_download(repo_id, cache_dir=cache_dir)
            print(f"  -> {path}", flush=True)
            success += 1
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}", flush=True)

    print(f"Done: {success}/{len(models)} models downloaded.", flush=True)


if __name__ == "__main__":
    main()
