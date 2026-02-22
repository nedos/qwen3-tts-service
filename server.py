"""Qwen3-TTS FastAPI Service — CPU/CUDA inference with voice cloning & design."""
import base64
import io
import os
import tempfile
import time
import asyncio
import logging
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3-tts")

# --- Config ---
MODEL_BASE = os.environ.get("TTS_MODEL_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_CUSTOM = os.environ.get("TTS_MODEL_CUSTOM", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
MODEL_DESIGN = os.environ.get("TTS_MODEL_DESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
MAX_SENTENCES_PER_CHUNK = int(os.environ.get("TTS_MAX_SENTENCES", "2"))
INTER_CHUNK_PAUSE = float(os.environ.get("TTS_CHUNK_PAUSE", "0.3"))
PORT = int(os.environ.get("PORT", "8000"))
MAX_REF_AUDIO_BYTES = 1 * 1024 * 1024  # 1MB limit for base64-decoded ref audio
DEVICE = os.environ.get("TTS_DEVICE", "auto")  # auto, cpu, cuda

# --- Global state ---
models = {}
generation_lock = asyncio.Lock()


def _resolve_device() -> str:
    """Resolve device: auto picks cuda if available."""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def load_models():
    """Load TTS models into RAM/VRAM."""
    device = _resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Loading models on {device} (dtype={dtype})")

    logger.info("Loading CustomVoice model...")
    t0 = time.time()
    from qwen_tts import Qwen3TTSModel
    models["custom"] = Qwen3TTSModel.from_pretrained(
        MODEL_CUSTOM,
        device_map=device,
        dtype=dtype,
    )
    logger.info(f"CustomVoice loaded in {time.time()-t0:.1f}s")

    logger.info("Loading Base model (for voice cloning)...")
    t0 = time.time()
    models["base"] = Qwen3TTSModel.from_pretrained(
        MODEL_BASE,
        device_map=device,
        dtype=dtype,
    )
    logger.info(f"Base model loaded in {time.time()-t0:.1f}s")

    # VoiceDesign — not baked into Docker image (saves ~4.5GB).
    # from_pretrained auto-downloads from HF if not cached.
    # On GPU instances this takes ~2-5s at 20gbit.
    logger.info(f"Loading VoiceDesign model ({MODEL_DESIGN})...")
    t0 = time.time()
    try:
        models["design"] = Qwen3TTSModel.from_pretrained(
            MODEL_DESIGN,
            device_map=device,
            dtype=dtype,
        )
        logger.info(f"VoiceDesign loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.warning(f"VoiceDesign model not available, /v1/design disabled: {e}")
        models["design"] = None

    models["_device"] = device


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    models.clear()


app = FastAPI(
    title="Qwen3-TTS Service",
    description="TTS with preset voices, voice cloning, and voice design",
    lifespan=lifespan,
)


# --- Request/Response models ---
class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "Ryan"
    instruct: Optional[str] = None
    response_format: str = "wav"


class CloneRequest(BaseModel):
    text: str
    language: str = "English"
    ref_audio_path: Optional[str] = None
    ref_audio_url: Optional[str] = None
    ref_audio_base64: Optional[str] = None
    x_vector_only: bool = True
    response_format: str = "wav"


class DesignRequest(BaseModel):
    text: str
    language: str = "English"
    instruct: str  # Voice description, e.g. "A deep male voice with a British accent"
    response_format: str = "wav"


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    device: str
    uptime: float


# --- Helpers ---
import re


def split_sentences(text: str, max_per_chunk: int = 2) -> list[str]:
    """Split text into chunks of max_per_chunk sentences.
    
    Handles sentence-ending punctuation (.!?) and also splits on
    semicolons and long clauses to keep chunks manageable.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Further split overly long sentences (>200 chars) on commas/semicolons
    expanded = []
    for s in sentences:
        if len(s) > 200:
            parts = re.split(r'(?<=[;,])\s+', s)
            expanded.extend(parts)
        else:
            expanded.append(s)
    
    chunks = []
    for i in range(0, len(expanded), max_per_chunk):
        chunk = ' '.join(expanded[i:i+max_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks if chunks else [text.strip()]


def wav_to_bytes(wav_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav_data, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


# Supported output formats
_FORMAT_MAP = {
    "wav": {"ext": "wav", "media_type": "audio/wav"},
    "mp3": {"ext": "mp3", "media_type": "audio/mpeg"},
    "opus": {"ext": "ogg", "media_type": "audio/ogg"},
    "ogg": {"ext": "ogg", "media_type": "audio/ogg"},
    "flac": {"ext": "flac", "media_type": "audio/flac"},
}


def encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convert WAV bytes to the requested format via pydub. Returns (audio_bytes, media_type)."""
    from pydub import AudioSegment

    fmt = fmt.lower()
    if fmt not in _FORMAT_MAP:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(_FORMAT_MAP.keys())}")

    info = _FORMAT_MAP[fmt]
    if fmt == "wav":
        return wav_bytes, info["media_type"]

    try:
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        out = io.BytesIO()
        export_fmt = "ogg" if fmt in ("opus", "ogg") else fmt
        export_params = []
        if fmt in ("opus", "ogg"):
            export_params = ["-c:a", "libopus", "-b:a", "96k"]
        elif fmt == "mp3":
            export_params = ["-q:a", "2"]
        audio.export(out, format=export_fmt, parameters=export_params)
        out.seek(0)
        return out.read(), info["media_type"]
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return wav_bytes, "audio/wav"


def _make_pause(sr: int, dtype) -> np.ndarray:
    """Create a silence array for inter-chunk pauses."""
    return np.zeros(int(sr * INTER_CHUNK_PAUSE), dtype=dtype)


async def generate_custom_voice(
    text: str, language: str, speaker: str, instruct: Optional[str] = None
) -> tuple[np.ndarray, int]:
    """Generate speech using CustomVoice model, chunked by sentences."""
    model = models["custom"]
    chunks = split_sentences(text, MAX_SENTENCES_PER_CHUNK)
    logger.info(f"CustomVoice: {len(chunks)} chunk(s) for {len(text)} chars")

    all_wavs = []
    for i, chunk in enumerate(chunks):
        kwargs = dict(text=chunk, language=language, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct

        wavs, sr = await asyncio.to_thread(
            model.generate_custom_voice, **kwargs
        )
        all_wavs.append(wavs[0])
        if i < len(chunks) - 1:
            all_wavs.append(_make_pause(sr, wavs[0].dtype))

    combined = np.concatenate(all_wavs)
    return combined, sr


async def generate_clone_voice(
    text: str, language: str,
    ref_audio: str, x_vector_only: bool = True
) -> tuple[np.ndarray, int]:
    """Generate speech using voice cloning, chunked by sentences."""
    model = models["base"]
    chunks = split_sentences(text, MAX_SENTENCES_PER_CHUNK)
    logger.info(f"Clone: {len(chunks)} chunk(s) for {len(text)} chars")

    # Build clone prompt once (expensive — extracts x-vector from ref audio)
    prompt_items = await asyncio.to_thread(
        model.create_voice_clone_prompt,
        ref_audio=ref_audio,
        ref_text="",
        x_vector_only_mode=x_vector_only,
    )

    all_wavs = []
    for i, chunk in enumerate(chunks):
        wavs, sr = await asyncio.to_thread(
            model.generate_voice_clone,
            text=chunk,
            language=language,
            voice_clone_prompt=prompt_items,
        )
        all_wavs.append(wavs[0])
        if i < len(chunks) - 1:
            all_wavs.append(_make_pause(sr, wavs[0].dtype))

    combined = np.concatenate(all_wavs)
    return combined, sr


async def generate_voice_design(
    text: str, language: str, instruct: str
) -> tuple[np.ndarray, int]:
    """Generate speech using voice design (description-based), chunked by sentences."""
    model = models.get("design")
    if model is None:
        raise RuntimeError("VoiceDesign model not loaded. Set TTS_MODEL_DESIGN or ensure model is available.")
    chunks = split_sentences(text, MAX_SENTENCES_PER_CHUNK)
    logger.info(f"Design: {len(chunks)} chunk(s) for {len(text)} chars")

    all_wavs = []
    for i, chunk in enumerate(chunks):
        wavs, sr = await asyncio.to_thread(
            model.generate_voice_design,
            text=chunk,
            language=language,
            instruct=instruct,
        )
        all_wavs.append(wavs[0])
        if i < len(chunks) - 1:
            all_wavs.append(_make_pause(sr, wavs[0].dtype))

    combined = np.concatenate(all_wavs)
    return combined, sr


# --- Endpoints ---
_start_time = time.time()


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=[k for k in models.keys() if not k.startswith("_")],
        device=models.get("_device", "unknown"),
        uptime=time.time() - _start_time,
    )


@app.post("/v1/tts")
async def tts(req: TTSRequest):
    """Generate speech using preset voices (CustomVoice model).
    
    Speakers: Ryan, Luna, Chelsie, Aidan, Serena, Daniel, Aria, Ethan, Nova
    """
    async with generation_lock:
        t0 = time.time()
        try:
            audio, sr = await generate_custom_voice(
                req.text, req.language, req.speaker, req.instruct
            )
        except Exception as e:
            logger.exception("TTS generation failed")
            raise HTTPException(status_code=500, detail=str(e))

        duration = len(audio) / sr
        gen_time = time.time() - t0
        logger.info(f"TTS: {duration:.1f}s audio in {gen_time:.1f}s ({gen_time/duration:.1f}x realtime)")

        wav_bytes = wav_to_bytes(audio, sr)
        out_bytes, media_type = encode_audio(wav_bytes, req.response_format)
        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type=media_type,
            headers={
                "X-Audio-Duration": f"{duration:.2f}",
                "X-Generation-Time": f"{gen_time:.2f}",
                "X-Output-Format": req.response_format,
            },
        )


def _resolve_ref_audio(req: CloneRequest) -> tuple[str, Optional[str]]:
    """Resolve ref audio to a file path. Returns (path, tmp_path_to_cleanup)."""
    if req.ref_audio_base64:
        b64 = req.ref_audio_base64
        if b64.startswith("data:"):
            _, b64 = b64.split(",", 1)
        try:
            raw = base64.b64decode(b64)
        except Exception:
            raise HTTPException(400, "Invalid base64 encoding")
        if len(raw) > MAX_REF_AUDIO_BYTES:
            raise HTTPException(400, f"ref_audio_base64 exceeds {MAX_REF_AUDIO_BYTES} byte limit")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name, tmp.name
    elif req.ref_audio_path:
        return req.ref_audio_path, None
    elif req.ref_audio_url:
        return req.ref_audio_url, None
    else:
        raise HTTPException(400, "Provide ref_audio_base64, ref_audio_path, or ref_audio_url")


@app.post("/v1/clone")
async def clone(req: CloneRequest):
    """Generate speech using voice cloning.

    Accepts reference audio as:
    - ref_audio_base64: base64-encoded audio bytes (with optional data URI prefix), max 1MB decoded
    - ref_audio_url: URL to download reference audio from
    - ref_audio_path: local file path (for server-side refs)
    """
    ref_audio_path, tmp_path = _resolve_ref_audio(req)

    try:
        async with generation_lock:
            t0 = time.time()
            try:
                audio, sr = await generate_clone_voice(
                    req.text, req.language, ref_audio_path, req.x_vector_only
                )
            except Exception as e:
                logger.exception("Clone generation failed")
                raise HTTPException(status_code=500, detail=str(e))

            duration = len(audio) / sr
            gen_time = time.time() - t0
            logger.info(f"Clone: {duration:.1f}s audio in {gen_time:.1f}s ({gen_time/duration:.1f}x realtime)")

            wav_bytes = wav_to_bytes(audio, sr)
            out_bytes, media_type = encode_audio(wav_bytes, req.response_format)
            return StreamingResponse(
                io.BytesIO(out_bytes),
                media_type=media_type,
                headers={
                    "X-Audio-Duration": f"{duration:.2f}",
                    "X-Generation-Time": f"{gen_time:.2f}",
                    "X-Output-Format": req.response_format,
                },
            )
    finally:
        if tmp_path:
            os.unlink(tmp_path)


@app.post("/v1/design")
async def design(req: DesignRequest):
    """Generate speech using voice design (description-based voice creation).

    Describe the voice you want and the model will synthesize it.
    Example instruct: "A warm female voice with a slight Southern accent, speaking slowly"
    """
    async with generation_lock:
        t0 = time.time()
        try:
            audio, sr = await generate_voice_design(
                req.text, req.language, req.instruct
            )
        except Exception as e:
            logger.exception("Design generation failed")
            raise HTTPException(status_code=500, detail=str(e))

        duration = len(audio) / sr
        gen_time = time.time() - t0
        logger.info(f"Design: {duration:.1f}s audio in {gen_time:.1f}s ({gen_time/duration:.1f}x realtime)")

        wav_bytes = wav_to_bytes(audio, sr)
        out_bytes, media_type = encode_audio(wav_bytes, req.response_format)
        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type=media_type,
            headers={
                "X-Audio-Duration": f"{duration:.2f}",
                "X-Generation-Time": f"{gen_time:.2f}",
                "X-Output-Format": req.response_format,
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
