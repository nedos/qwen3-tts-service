# Qwen3-TTS Service

TTS service with preset voices, voice cloning, and voice design — built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

- **Preset Voices** (`/v1/tts`) — 9 built-in voices with instruction control
- **Voice Cloning** (`/v1/clone`) — Clone any voice from a short audio clip
- **Voice Design** (`/v1/design`) — Describe a voice and the model creates it
- **Auto-chunking** — Long text is split into sentence chunks automatically
- **CPU & CUDA** — Runs on CPU (~9GB RAM) or GPU (~5GB VRAM, 10x faster)
- **No baked-in voices** — Mount reference audio at runtime

## API

### `POST /v1/tts` — Preset voice synthesis

```json
{
  "text": "Your deployment is on fire.",
  "language": "English",
  "speaker": "Ryan",
  "instruct": "Speak with dry sarcasm"
}
```

**Speakers:** Ryan, Luna, Chelsie, Aidan, Serena, Daniel, Aria, Ethan, Nova

### `POST /v1/clone` — Voice cloning

```json
{
  "text": "Your deployment is on fire.",
  "language": "English",
  "ref_audio_base64": "<base64-encoded-wav>",
  "x_vector_only": true
}
```

Reference audio can be provided as:
- `ref_audio_base64` — Base64-encoded audio (with optional `data:` URI prefix), max 1MB
- `ref_audio_url` — URL to download reference audio
- `ref_audio_path` — Local file path on the server (for mounted refs)

### `POST /v1/design` — Voice design

```json
{
  "text": "Your deployment is on fire.",
  "language": "English",
  "instruct": "A deep male voice with a British accent, speaking slowly and deliberately"
}
```

Describe the voice characteristics and the model synthesizes a matching voice.

### `GET /health` — Health check

```json
{
  "status": "ok",
  "models_loaded": ["custom", "base"],
  "device": "cpu",
  "uptime": 123.4
}
```

## Docker

### CPU (default)

```bash
docker build -t qwen3-tts .
docker run -p 8000:8000 qwen3-tts
```

### CUDA

```bash
docker build -t qwen3-tts-cuda \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
docker run --gpus all -p 8000:8000 qwen3-tts-cuda
```

### With reference audio

```bash
docker run -p 8000:8000 -v /path/to/refs:/data/refs qwen3-tts
```

Then use `"ref_audio_path": "/data/refs/speaker.wav"` in clone requests.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TTS_DEVICE` | `auto` | Device: `auto`, `cpu`, or `cuda` |
| `TTS_MODEL_BASE` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model (clone + design) |
| `TTS_MODEL_CUSTOM` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Custom voice model (preset voices) |
| `TTS_MAX_SENTENCES` | `2` | Max sentences per generation chunk |
| `TTS_CHUNK_PAUSE` | `0.3` | Pause between chunks (seconds) |
| `PORT` | `8000` | Server port |

## Auto-Chunking

Long text is automatically split into sentence-sized chunks (default: 2 sentences per chunk). This prevents model quality degradation on long passages. Chunks are generated sequentially and concatenated with a configurable pause.

For very long sentences (>200 chars), additional splitting occurs at commas and semicolons.

## Performance

| Setup | Generation Speed | RAM/VRAM |
|---|---|---|
| CPU (EPYC 7502P) | ~8-10x slower than realtime | ~9GB RAM |
| GPU (L40S/A100) | ~1-2x realtime | ~5GB VRAM |

## Responses

All endpoints return `audio/wav` with headers:
- `X-Audio-Duration` — Generated audio duration in seconds
- `X-Generation-Time` — Wall-clock generation time in seconds
