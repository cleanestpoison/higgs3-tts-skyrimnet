"""Higgs Audio v3 TTS (audio.cpp) Gradio wrapper exposing the Zonos client API.

Native Windows port of the wrapper from `elbios/higgs3-whisper`.

The Skyrim AI mod speaks a fixed "Zonos" Gradio API. This module reproduces that
surface exactly -- 29 positional inputs, a single gr.Audio output returning a
file path -- and drives audio.cpp's `audiocpp_server` behind it.

The engine is a separate C++ process on an internal port (8084, loopback only),
and the wrapper is a thin HTTP client whose job is (a) speaking the Zonos API,
(b) normalising the reference clip, (c) bounding generation length, and (d) not
binding its port until everything downstream is genuinely warm.

Worth knowing:
  * audiocpp_server takes a *server-local path* for the reference (`voice_ref`),
    not base64 bytes, so the content-addressed refcache path is handed to the
    engine directly.
  * The engine keeps its own reference-encode cache, keyed on the *content* of
    the audio (FNV hash over samples + reference_text), so identical clips hit
    it regardless of path. Slots are raised from the default 1 via
    `higgs_audio_tts.reference_cache_slots` in server.json -- a playthrough
    alternates actor voices.
  * Cloning is ICL-style (reference codes in the prompt) but the transcript is
    OPTIONAL, and transcript-free cloning works. Default is transcript-free: no
    transcriber on the TTS path, no truncated-transcript EOS collapse possible.
    HIGGS_USE_REFERENCE_TEXT=1 re-enables transcripts via SimpleParakeet.

The port is not bound until the engine has loaded and a full dummy generation
has completed, so a racing client can never win with a cold box.

Windows changes from the container version, all marked WINDOWS below:
  * Paths default to this script's folder instead of /opt/...
  * ffmpeg is resolved via HIGGS_FFMPEG (or PATH) rather than assumed at /usr/bin.
  * ffmpeg runs without flashing a console window.
  * The /health transcriber probe derives host:port from WHISPER_API_URL instead
    of assuming 127.0.0.1:8080.
  * STT is off by default -- SimpleParakeet already serves SkyrimNet on 8210,
    and transcript-free cloning is the default path anyway.
  * Binds loopback by default rather than 0.0.0.0 (no container to escape, and
    no Windows Firewall prompt).
"""

import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from loguru import logger

# WINDOWS: everything is relative to this file so the folder can be moved.
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HIGGS_URL = os.environ.get("HIGGS_ENGINE_URL", "http://127.0.0.1:8081")
HIGGS_MODEL_ID = os.environ.get("HIGGS_MODEL_ID", "higgs")
HIGGS_WAIT_SECONDS = int(os.environ.get("HIGGS_WAIT_SECONDS", "900"))
HIGGS_REQUEST_TIMEOUT = int(os.environ.get("HIGGS_REQUEST_TIMEOUT", "600"))

# Engine sampling defaults for higgs_audio_tts are temperature=0.8, top_k=30,
# top_p=0.8 (audio.cpp's own defaults, deliberately narrower than the Python
# client's -- less prone to premature EOC). Only top_p is under the mod's
# control (Zonos input [20]); the others stay at engine defaults unless
# overridden here.
AUDIO_TEMPERATURE = os.environ.get("HIGGS_TEMPERATURE", "").strip()
AUDIO_TOP_K = os.environ.get("HIGGS_TOP_K", "").strip()
AUDIO_REPETITION_PENALTY = os.environ.get("HIGGS_REPETITION_PENALTY", "").strip()
TEXT_CHUNK_SIZE = os.environ.get("HIGGS_TEXT_CHUNK_SIZE", "").strip()
TEXT_CHUNK_MODE = os.environ.get("HIGGS_TEXT_CHUNK_MODE", "").strip()


# WINDOWS: ffmpeg is not a given on PATH the way it is in the image.
FFMPEG = os.environ.get("HIGGS_FFMPEG", "").strip() or "ffmpeg"

# WINDOWS: keep ffmpeg from flashing a console window on every new reference.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# --------------------------------------------------------------------------
# Length budget -- OFF by default (the measurements back it)
#
# The text-derived cap existed to truncate other models' runaway
# hallucinations. Higgs v3 measured clean on the short-line grid at stock
# settings, and audio.cpp's cap is a *hard error with no audio* rather than a
# graceful flush -- so an aggressive budget here could only hurt: it can never
# save a good line, and a miscalibrated one turns a slow-spoken line into a
# failed request. The engine's own per-chunk default (2048 AR steps ~ 80 s of
# audio) stays as the pathological bound.
#
# HIGGS_TOKEN_BUDGET=1 re-enables the text-derived budget (measured: ~25 AR
# tokens per audio-second, English ~15 chars/s), with a CapHit retry at the
# ceiling so even then nothing is lost -- just latency.
TOKEN_BUDGET_ENABLED = os.environ.get("HIGGS_TOKEN_BUDGET", "0") == "1"
TOKENS_PER_SECOND = float(os.environ.get("HIGGS_TOKENS_PER_SECOND", "25.0"))
CHARS_PER_SECOND = float(os.environ.get("HIGGS_CHARS_PER_SECOND", "15.0"))
MAX_TOKEN_SLACK = float(os.environ.get("HIGGS_MAX_TOKEN_SLACK", "4.0"))
MAX_TOKEN_FLOOR = int(os.environ.get("HIGGS_MAX_TOKEN_FLOOR", "150"))
MAX_TOKEN_CEIL = int(os.environ.get("HIGGS_MAX_TOKEN_CEIL", "2048"))

# Trailing near-silence trim: the model finishes the line and then may pad;
# nothing downstream wants dead air.
TRIM_SILENCE = os.environ.get("HIGGS_TRIM_SILENCE", "0").lower() not in ("0", "false", "no")
TRIM_THRESHOLD = float(os.environ.get("HIGGS_TRIM_THRESHOLD", "0.02"))  # of peak
TRIM_TAIL_PAD_S = float(os.environ.get("HIGGS_TRIM_TAIL_PAD_S", "0.15"))

# WINDOWS: default points at SimpleParakeet (already running for SkyrimNet's
# External Whisper) rather than a whisper.cpp this port does not ship.
WHISPER_API_URL = os.environ.get(
    "WHISPER_API_URL", "http://127.0.0.1:8210/v1/audio/transcriptions"
)
WHISPER_WAIT_SECONDS = int(os.environ.get("WHISPER_WAIT_SECONDS", "20"))
# WINDOWS: off by default. It is only ever used to read reference clips, and
# only when HIGGS_USE_REFERENCE_TEXT=1; transcript-free cloning is the default.
WHISPER_ENABLED = os.environ.get("WHISPER_ENABLED", "0") != "0"
# Boot-time STT state, reported in /health: "ok" | "down" | "disabled".
WHISPER_STATUS = "disabled"

# Off by default: transcript-free cloning avoids the ICL truncated-transcript
# collapse entirely. When on, the reference is transcribed once through the
# transcriber above and cached by content digest.
USE_REFERENCE_TEXT = os.environ.get("HIGGS_USE_REFERENCE_TEXT", "0") == "1"

# Content-addressed store for normalised reference clips.
REF_CACHE_DIR = Path(os.environ.get("HIGGS_REF_CACHE_DIR", str(HERE / "refcache")))
REF_CACHE_SIZE = int(os.environ.get("HIGGS_REF_CACHE_SIZE", "64"))

# The codec's sample rate; references are normalised to it so the engine's
# resampler never runs, and its content-keyed cache sees identical samples for
# identical uploads.
HIGGS_SAMPLE_RATE = int(os.environ.get("HIGGS_SAMPLE_RATE", "24000"))
REF_MAX_SECONDS = float(os.environ.get("HIGGS_REF_MAX_SECONDS", "60"))

# Local high-quality reference clips, keyed by the *name* of the file the mod
# uploads. SkyrimNet resamples every reference it sends to 16 kHz -- even the
# 44.1 kHz clips in its own voice-samples folder -- which throws away everything
# above 8 kHz before the codec (24 kHz, so ~12 kHz of headroom) ever sees it.
# A file here named after the incoming upload is used in its place, at whatever
# rate it is stored, so the clip reaches the engine unspoiled.
#
# The key is the upload's filename stem, which is the voicetype for a generic
# NPC (`femalenord.wav`) and the character for a curated one (`serana.wav`) --
# so a single file can re-voice every NPC sharing a voicetype, or one named
# character. Matching is case-insensitive; any format ffmpeg reads will do,
# since references are normalised on the way in regardless.
SAMPLES_DIR = Path(os.environ.get("HIGGS_SAMPLES_DIR", str(HERE / "samples")))
SAMPLES_ENABLED = os.environ.get("HIGGS_SAMPLES", "1") != "0"
SAMPLE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")

SERVER_PORT = int(os.environ.get("HIGGS_PORT", "7860"))
# WINDOWS: loopback by default. Set HIGGS_HOST=0.0.0.0 to reach it from the LAN.
SERVER_HOST = os.environ.get("HIGGS_HOST", "127.0.0.1")

# Reference clip used to warm every lazy code path at boot, and to answer the
# mod's health-check ping (see PING_TEXT).
WARMUP_REF = Path(os.environ.get("HIGGS_WARMUP_REF", str(HERE / "warmup_ref.wav")))

# SkyrimNet's TTSPingManager sends this exact line every 30 s to check the
# endpoint is alive. It goes out in the player voice, which has no reference
# clip, so it is the one request allowed to fall back to the warmup reference.
# Set HIGGS_PING_TEXT= (empty) to treat it as an ordinary request instead.
PING_TEXT = os.environ.get("HIGGS_PING_TEXT", "ping").strip().lower()

# What that ping is answered with. The warmup clip is already on disk and the
# check only looks for audio, so nothing is generated for it. Point this at a
# silent clip if the mod ever turns out to actually play the ping.
PING_WAV = Path(os.environ.get("HIGGS_PING_WAV", "").strip() or WARMUP_REF)

READY = False
ENGINE_INFO = {}

# --------------------------------------------------------------------------
# Content-addressed reference cache
#
# Gradio's /gradio_api/upload writes every upload to a *fresh* temp path, so
# the identical Skyrim reference clip arrives under a new name every request.
# Hashing the content gives a stable key and a stable canonical path -- which
# is what gets handed to the engine as `voice_ref`.
#
# Everything is normalised to 16-bit PCM mono at the codec's 24 kHz first:
# reference clips in circulation include MP3-in-WAV (fmt tag 85), and
# normalising also keeps the engine's own content-keyed encode cache hot (same
# samples in => same FNV hash => hit).
# --------------------------------------------------------------------------

# digest -> canonical path (LRU of what we know is on disk and normalised)
_REF_CACHE: "OrderedDict[str, str]" = OrderedDict()
# digest -> transcript (only populated when USE_REFERENCE_TEXT)
_TRANSCRIPT_CACHE: "OrderedDict[str, str]" = OrderedDict()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalise_reference(src: Path, dst: Path) -> None:
    """Rewrite any audio file as 16-bit PCM mono at HIGGS_SAMPLE_RATE."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-t", str(REF_MAX_SECONDS),
        "-ac", "1", "-ar", str(HIGGS_SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120, **_NO_WINDOW)
    except FileNotFoundError:
        # WINDOWS: the container always had ffmpeg; here its absence is the
        # single most likely setup mistake, so say so in those words.
        raise RuntimeError(
            f"ffmpeg not found (tried '{FFMPEG}'). Install it and put it on PATH, "
            f"or set HIGGS_FFMPEG to its full path in config.ps1."
        ) from None
    if not dst.exists() or dst.stat().st_size <= 44:
        raise RuntimeError(f"reference normalisation produced nothing for {src}")


def reference_path(upload_path: str) -> tuple[str, str]:
    """Return (canonical normalised path, content digest) for an upload."""
    digest = file_sha256(upload_path)

    cached = _REF_CACHE.get(digest)
    if cached is not None and os.path.exists(cached):
        _REF_CACHE.move_to_end(digest)
        return cached, digest

    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    canonical = REF_CACHE_DIR / f"{digest}.wav"
    if not canonical.exists():
        tmp = REF_CACHE_DIR / f".{digest}.partial.wav"
        _normalise_reference(Path(upload_path), tmp)
        os.replace(tmp, canonical)
        info = sf.info(str(canonical))
        logger.info(
            f"Normalised new reference {digest[:12]} -> {canonical.name} "
            f"({info.duration:.2f}s, {info.samplerate} Hz, {info.channels}ch)"
        )
        # ffmpeg's -t truncates silently, so a too-long clip would otherwise be
        # clipped with no trace. Inferred from the output length rather than
        # probing the source: sf.info is already loaded here, so this costs
        # nothing on a path that is latency-sensitive.
        if info.duration >= REF_MAX_SECONDS - 0.05:
            logger.warning(
                f"Reference {digest[:12]} hit the {REF_MAX_SECONDS:.0f}s cap and "
                f"was truncated (raise HIGGS_REF_MAX_SECONDS to keep more)"
            )

    _REF_CACHE[digest] = str(canonical)
    while len(_REF_CACHE) > REF_CACHE_SIZE:
        evicted, path = _REF_CACHE.popitem(last=False)
        try:
            os.unlink(path)
        except OSError:
            pass
        logger.info(f"Evicted reference {evicted[:12]} from the refcache")
    return str(canonical), digest


def reference_transcript(digest: str, ref_path: str) -> str | None:
    """Transcribe a reference once via the transcriber, cached by content digest.

    Only used when HIGGS_USE_REFERENCE_TEXT=1. Returns None (=> transcript-free
    request) when the transcriber is unavailable or fails -- the engine treats an
    absent transcript as its normal mode, so failure here degrades gracefully.
    """
    cached = _TRANSCRIPT_CACHE.get(digest)
    if cached is not None:
        _TRANSCRIPT_CACHE.move_to_end(digest)
        return cached
    if WHISPER_STATUS != "ok":
        return None
    try:
        with open(ref_path, "rb") as f:
            r = requests.post(
                WHISPER_API_URL,
                files={"file": ("ref.wav", f, "audio/wav")},
                data={"temperature": "0", "response_format": "json"},
                timeout=120,
            )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if not text:
            return None
        _TRANSCRIPT_CACHE[digest] = text
        while len(_TRANSCRIPT_CACHE) > REF_CACHE_SIZE:
            _TRANSCRIPT_CACHE.popitem(last=False)
        logger.info(f"Transcribed reference {digest[:12]}: '{text[:80]}'")
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Reference transcription failed ({exc}); going transcript-free")
        return None


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------

CHINESE_TO_ENGLISH_PUNCT = {
    "，": ", ", "。": ". ", "：": ": ", "；": "; ", "？": "? ", "！": "! ",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "、": ", ", "—": "-", "…": "...", "·": ".",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "「": '"', "」": '"', "『": '"', "』": '"',
}


def preprocess_text(text: str) -> str:
    for zh, en in CHINESE_TO_ENGLISH_PUNCT.items():
        text = text.replace(zh, en)
    text = text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n") if line.strip())
    text = text.strip()
    if text and text[-1] not in ".!?,;\"'":
        text += "."
    return text


_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def strip_control_tokens(text: str) -> str:
    """What the model will actually speak -- tokens removed, for length maths."""
    return re.sub(r"[ \t]+", " ", _TOKEN_RE.sub("", text)).strip()


def expected_seconds(text: str) -> float:
    return len(text) / CHARS_PER_SECOND


def token_budget(text: str) -> int | None:
    """Length bound in AR steps, derived from text length with a floor.

    Returns None (send nothing; engine default 2048 applies) unless
    HIGGS_TOKEN_BUDGET=1.
    """
    if not TOKEN_BUDGET_ENABLED:
        return None
    expected_tokens = expected_seconds(text) * TOKENS_PER_SECOND
    return int(min(MAX_TOKEN_CEIL,
                   max(MAX_TOKEN_FLOOR, expected_tokens * MAX_TOKEN_SLACK + MAX_TOKEN_FLOOR)))


# --------------------------------------------------------------------------
# Engine client
# --------------------------------------------------------------------------


class CapHitError(RuntimeError):
    """The engine hit max_tokens before EOC.

    Measured behaviour: audiocpp_server returns an HTTP error ("Higgs TTS
    generation reached max_tokens before EOC") and NO audio -- the cap is a
    hard failure, not a graceful flush. The caller retries once with the
    ceiling so a legitimately slow-spoken line still returns audio.
    """


def higgs_tts(text: str, ref_path: str | None, reference_text: str | None,
              top_p: float | None, seed: int | None,
              temperature: float | None = None,
              top_k: int | None = None,
              repetition_penalty: float | None = None,
              max_tokens: int | None = None,
              text_chunk_size: int | None = None,
              text_chunk_mode: str | None = None) -> bytes:
    """POST /v1/audio/speech and return WAV bytes."""
    payload: dict = {
        "model": HIGGS_MODEL_ID,
        "input": text,
    }
    budget = max_tokens if max_tokens is not None else token_budget(text)
    if budget is not None:
        payload["max_tokens"] = budget
    if ref_path:
        payload["voice_ref"] = ref_path
    if reference_text:
        payload["reference_text"] = reference_text
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed

    if temperature is not None:
        payload["temperature"] = temperature
    elif AUDIO_TEMPERATURE:
        payload["temperature"] = float(AUDIO_TEMPERATURE)

    if top_k is not None:
        payload["top_k"] = top_k
    elif AUDIO_TOP_K:
        payload["top_k"] = int(AUDIO_TOP_K)

    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    elif AUDIO_REPETITION_PENALTY:
        payload["repetition_penalty"] = float(AUDIO_REPETITION_PENALTY)

    if text_chunk_size is not None:
        payload["text_chunk_size"] = text_chunk_size
    elif TEXT_CHUNK_SIZE:
        payload["text_chunk_size"] = int(TEXT_CHUNK_SIZE)

    if text_chunk_mode is not None:
        payload["text_chunk_mode"] = text_chunk_mode
    elif TEXT_CHUNK_MODE:
        payload["text_chunk_mode"] = TEXT_CHUNK_MODE

    response = requests.post(f"{HIGGS_URL}/v1/audio/speech", json=payload,
                             timeout=HIGGS_REQUEST_TIMEOUT)
    if response.status_code != 200:
        body = response.text[:400]
        if "max_tokens" in body:
            raise CapHitError(body)
        raise RuntimeError(
            f"audiocpp_server returned {response.status_code}: {body}"
        )
    return response.content


def wav_duration(data: bytes) -> tuple[float, int]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate(), w.getframerate()


def trim_trailing_silence(data: bytes) -> tuple[bytes, float]:
    """Drop trailing near-silence from a 16-bit PCM mono WAV.

    Only the *tail* is trimmed, only below TRIM_THRESHOLD of the clip's own
    peak, with TRIM_TAIL_PAD_S left so nothing clips a decaying consonant.
    Returns (wav bytes, seconds removed); on any surprise the input is
    returned untouched -- a nicety, never a reason to fail a request.
    """
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return data, 0.0
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
        x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            return data, 0.0

        win = max(1, int(0.02 * rate))
        n_win = x.size // win
        if n_win < 2:
            return data, 0.0
        energies = np.sqrt((x[:n_win * win].reshape(n_win, win) ** 2).mean(axis=1))
        max_e = energies.max()
        loud = np.flatnonzero(energies > TRIM_THRESHOLD * max_e)
        if loud.size == 0:
            return data, 0.0

        last_loud_win = loud[-1]

        # Detect post-speech EOS noise pops/bursts (an isolated noise burst <= 350ms
        # occurring after a silence gap >= 80ms following main speech).
        is_quiet = energies < (0.015 * max_e)
        q_end = last_loud_win
        while q_end >= 0 and not is_quiet[q_end]:
            q_end -= 1
        if q_end >= 0:
            q_start = q_end
            while q_start >= 0 and is_quiet[q_start]:
                q_start -= 1
            quiet_len = q_end - q_start
            if quiet_len >= 4 and (last_loud_win - q_end) * win / rate <= 0.35:
                main_speech = loud[loud <= q_start]
                if main_speech.size > 0:
                    last_loud_win = main_speech[-1]

        keep = min(x.size, int((last_loud_win + 1) * win + TRIM_TAIL_PAD_S * rate))
        removed = (x.size - keep) / rate
        logger.info(f"Trim silence: input {x.size/rate:.2f}s -> kept {keep/rate:.2f}s (removed {removed:.3f}s)")
        # Apply a 10ms micro fade-out at the tail to eliminate PCM DC-offset clicks and pops
        fade_samples = min(int(0.010 * rate), keep)
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            x[keep - fade_samples : keep] *= fade

        out_pcm = (x[:keep] * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(out_pcm)
        return out.getvalue(), removed
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Silence trim skipped: {exc}")
        return data, 0.0


def truncate_runaway_audio(data: bytes, max_allowed_s: float) -> tuple[bytes, bool]:
    """Truncate runaway audio (e.g. repetition loops) and apply a 0.5s fade-out."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return data, False
            rate = w.getframerate()
            n_frames = w.getnframes()
            dur = n_frames / rate
            if dur <= max_allowed_s:
                return data, False
            frames = w.readframes(n_frames)

        x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        target_len = int(max_allowed_s * rate)
        if target_len >= x.size:
            return data, False

        # Apply a 0.5-second linear fade-out at the end
        fade_len = min(int(0.5 * rate), target_len)
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            x[target_len - fade_len:target_len] *= fade

        out_pcm = (x[:target_len] * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(out_pcm)
        return out.getvalue(), True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Runaway audio truncation skipped: {exc}")
        return data, False


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------


def wait_for_engine() -> None:
    """Block until audiocpp_server answers /health (the GGUF is loading/loaded)."""
    deadline = time.time() + HIGGS_WAIT_SECONDS
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"{HIGGS_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info(f"audiocpp_server is answering after {attempt} probe(s)")
                return
        except Exception as exc:  # noqa: BLE001 - any failure means "not up yet"
            if attempt % 15 == 1:
                logger.info(f"Waiting for audiocpp_server ({type(exc).__name__})...")
        time.sleep(2)
    raise RuntimeError(f"audiocpp_server did not come up within {HIGGS_WAIT_SECONDS}s")


def wait_for_whisper() -> bool:
    """Block until the transcriber answers a real transcription. Non-fatal.

    WINDOWS: bounded far tighter than the container's 300s. SimpleParakeet is
    either already up (it answers in ~0.2s) or it is not running at all, and
    this is not on the TTS path either way.
    """
    deadline = time.time() + WHISPER_WAIT_SECONDS
    probe = WARMUP_REF if WARMUP_REF.exists() else None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            if probe is None:
                requests.get(WHISPER_API_URL.rsplit("/", 1)[0] + "/", timeout=5)
                logger.info("Transcriber is answering")
                return True
            with open(probe, "rb") as f:
                r = requests.post(
                    WHISPER_API_URL,
                    files={"file": ("probe.wav", f, "audio/wav")},
                    data={"temperature": "0", "response_format": "json"},
                    timeout=120,
                )
            r.raise_for_status()
            logger.info(f"Transcriber ready after {attempt} probe(s)")
            return True
        except Exception as exc:  # noqa: BLE001
            if attempt % 5 == 1:
                logger.info(f"Waiting for transcriber ({type(exc).__name__})...")
            time.sleep(2)
    logger.warning(
        f"Transcriber did not answer within {WHISPER_WAIT_SECONDS}s; "
        "continuing transcript-free (TTS unaffected)"
    )
    return False


def initialize() -> None:
    """Wait for the engine and force every lazy path warm before binding the port."""
    global READY, ENGINE_INFO, WHISPER_STATUS

    wait_for_engine()
    try:
        ENGINE_INFO = requests.get(f"{HIGGS_URL}/v1/models", timeout=10).json()
        logger.info(f"Engine models: {ENGINE_INFO}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read /v1/models: {exc}")

    # The transcriber is not on the TTS path (transcript-free cloning): with
    # HIGGS_USE_REFERENCE_TEXT=1 it improves cloning, but its absence still
    # degrades gracefully to transcript-free requests.
    if WHISPER_ENABLED:
        WHISPER_STATUS = "ok" if wait_for_whisper() else "down"
    else:
        logger.info("WHISPER_ENABLED=0: reference transcription off (transcript-free cloning)")

    # A full dummy clone. The engine's /health answers as soon as the HTTP
    # layer is up; the CUDA graphs, codec encoder and codec decoder are all
    # cold until one real generation has been through the full pipeline.
    if WARMUP_REF.exists():
        t0 = time.time()
        try:
            ref, digest = reference_path(str(WARMUP_REF))
            transcript = reference_transcript(digest, ref) if USE_REFERENCE_TEXT else None
            audio = higgs_tts(
                "The Jarl will see you now, if you have business here.",
                ref, transcript, None, 1234,
            )
            duration, _ = wav_duration(audio)
            logger.info(
                f"Warmup generation: {duration:.2f}s audio in {time.time() - t0:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Warmup generation failed (continuing): {exc}")
    else:
        logger.warning(f"No warmup reference at {WARMUP_REF}; first request will be slow")

    READY = True
    logger.info("Higgs3 wrapper is READY")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


# Parameter names of generate_audio in Zonos positional order. Kept as an
# explicit tuple (rather than read off locals()) so the request log stays in
# the mod's wire order no matter how the signature is edited.
INPUT_NAMES = (
    "model", "text", "language", "speaker_audio", "prefix_audio",
    "response_tone_happiness", "response_tone_sadness", "response_tone_disgust",
    "response_tone_fear", "response_tone_surprise", "response_tone_anger",
    "response_tone_other", "response_tone_neutral",
    "vq_score", "fmax", "pitch_std", "speaking_rate", "dnsmos_overall",
    "denoise_speaker", "cfg_scale", "top_p", "min_k", "min_p", "linear",
    "confidence", "quadratic", "seed", "randomize_seed", "unconditional_keys",
)

# Longest a single logged string value may be before it is elided; the text
# field is the only one that ever gets near it.
LOG_VALUE_CHARS = int(os.environ.get("HIGGS_LOG_VALUE_CHARS", "160"))


def _describe_param(value) -> str:
    """Compact single-line description of one incoming request parameter."""
    if isinstance(value, str):
        shown = value if len(value) <= LOG_VALUE_CHARS else value[:LOG_VALUE_CHARS] + "..."
        return repr(shown)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    # Gradio file inputs: the path matters, and so does whether it is really
    # there -- a missing/zero-byte upload is a failure mode worth seeing.
    path = None
    for attr in ("name", "path"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str):
            path = candidate
            break
    if path is None and isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        path = candidate if isinstance(candidate, str) else None
    if path is not None:
        try:
            size = os.path.getsize(path)
            return f"{path!r} ({size} bytes)"
        except OSError:
            return f"{path!r} (missing)"
    return f"<{type(value).__name__}>"


def is_ping(text) -> bool:
    """True for the mod's health-check line (see PING_TEXT)."""
    return bool(PING_TEXT) and str(text or "").strip().lower() == PING_TEXT


def log_request(
    params: dict,
    temperature_val: float | None = None,
    top_p_val: float | None = None,
    top_k_val: int | None = None,
    rep_pen_val: float | None = None,
    chunk_size_val: int | None = None,
    chunk_mode_val: str | None = None,
    seed_val: int | None = None,
) -> None:
    """Log only the parameters that Higgs Audio TTS actually uses."""
    parts = []
    text = params.get("text")
    if text:
        parts.append(f"text={_describe_param(text)}")

    speaker_audio = params.get("speaker_audio")
    if speaker_audio is not None:
        parts.append(f"speaker_audio={_describe_param(speaker_audio)}")

    language = params.get("language")
    if language:
        parts.append(f"language={_describe_param(language)}")

    if temperature_val is not None:
        parts.append(f"temperature={temperature_val}")
    elif AUDIO_TEMPERATURE:
        parts.append(f"temperature={AUDIO_TEMPERATURE} (env)")

    if top_p_val is not None:
        parts.append(f"top_p={top_p_val}")

    if top_k_val is not None:
        parts.append(f"top_k={top_k_val}")
    elif AUDIO_TOP_K:
        parts.append(f"top_k={AUDIO_TOP_K} (env)")

    if rep_pen_val is not None:
        parts.append(f"repetition_penalty={rep_pen_val}")
    elif AUDIO_REPETITION_PENALTY:
        parts.append(f"repetition_penalty={AUDIO_REPETITION_PENALTY} (env)")

    if chunk_size_val is not None:
        parts.append(f"text_chunk_size={chunk_size_val}")
    elif TEXT_CHUNK_SIZE:
        parts.append(f"text_chunk_size={TEXT_CHUNK_SIZE} (env)")

    if chunk_mode_val is not None:
        parts.append(f"text_chunk_mode={chunk_mode_val}")
    elif TEXT_CHUNK_MODE:
        parts.append(f"text_chunk_mode={TEXT_CHUNK_MODE} (env)")

    if seed_val is not None:
        parts.append(f"seed={seed_val}")

    logger.info(f"Incoming request: {', '.join(parts)}")


def _resolve_upload(speaker_audio) -> str | None:
    """Gradio hands us either a NamedString/tempfile object or a plain path."""
    if speaker_audio is None:
        return None
    for attr in ("name", "path"):
        value = getattr(speaker_audio, attr, None)
        if isinstance(value, str) and os.path.exists(value):
            return value
    if isinstance(speaker_audio, dict):
        value = speaker_audio.get("path") or speaker_audio.get("name")
        if isinstance(value, str) and os.path.exists(value):
            return value
    if isinstance(speaker_audio, str) and os.path.exists(speaker_audio):
        return speaker_audio
    logger.warning(f"Could not resolve speaker audio from {type(speaker_audio)}")
    return None


# Index of SAMPLES_DIR, rebuilt when the directory's mtime moves so clips can be
# dropped in while the server runs. Maps lowercased stem -> path; a stem present
# under several extensions resolves in SAMPLE_EXTENSIONS order.
_samples_index: dict[str, Path] = {}
_samples_mtime: float | None = None


def _sample_index() -> dict[str, Path]:
    global _samples_index, _samples_mtime
    try:
        mtime = SAMPLES_DIR.stat().st_mtime
    except OSError:
        _samples_index, _samples_mtime = {}, None
        return _samples_index
    if mtime == _samples_mtime:
        return _samples_index

    index: dict[str, Path] = {}
    for entry in sorted(SAMPLES_DIR.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in SAMPLE_EXTENSIONS:
            continue
        stem = entry.stem.lower()
        current = index.get(stem)
        if current is None or SAMPLE_EXTENSIONS.index(suffix) < SAMPLE_EXTENSIONS.index(
            current.suffix.lower()
        ):
            index[stem] = entry

    _samples_index, _samples_mtime = index, mtime
    logger.info(f"Local samples: {len(index)} clip(s) in {SAMPLES_DIR}")
    return _samples_index


def _local_sample(upload_path: str) -> str | None:
    """Local override for an upload, matched on filename stem. None if absent."""
    if not SAMPLES_ENABLED:
        return None
    try:
        match = _sample_index().get(Path(upload_path).stem.lower())
    except OSError as exc:
        logger.warning(f"Local samples unreadable: {exc}")
        return None
    return str(match) if match else None


def generate_audio(
    model,                     # 0  ignored (Zonos model id)
    text,                      # 1  the line to speak
    language,                  # 2  used as an emotion-tag channel (see below)
    speaker_audio,             # 3  reference clip for voice cloning
    prefix_audio,              # 4  ignored
    response_tone_happiness,   # 5  ignored - moods accepted and discarded
    response_tone_sadness,     # 6  ignored
    response_tone_disgust,     # 7  ignored
    response_tone_fear,        # 8  ignored
    response_tone_surprise,    # 9  ignored
    response_tone_anger,       # 10 ignored
    response_tone_other,       # 11 ignored
    response_tone_neutral,     # 12 ignored
    vq_score,                  # 13 ignored
    fmax,                      # 14 ignored
    pitch_std,                 # 15 ignored
    speaking_rate,             # 16 ignored
    dnsmos_overall,            # 17 ignored
    denoise_speaker,           # 18 ignored
    cfg_scale,                 # 19 ignored
    top_p,                     # 20 used (top_p)
    min_k,                     # 21 used (top_k)
    min_p,                     # 22 used (repetition_penalty)
    linear,                    # 23 used (temperature - sent by SkyrimNet HiggsInterface)
    confidence,                # 24 used (text_chunk_size)
    quadratic,                 # 25 used (text_chunk_mode)
    seed,                      # 26 used (seed)
    randomize_seed,            # 27 ignored
    unconditional_keys,        # 28 ignored
):
    """Zonos-compatible entry point. Returns a path to a completed WAV."""
    request_start = time.time()

    if not text or not str(text).strip():
        logger.error("Empty text provided")
        return None

    engine_text = preprocess_text(str(text))
    processed_text = strip_control_tokens(engine_text)

    if not processed_text:
        logger.error("Nothing left to speak after preprocessing")
        return None

    logger.info(f"Request: '{engine_text[:120]}'")

    upload_path = _resolve_upload(speaker_audio)
    if upload_path is None and is_ping(text) and PING_WAV.exists():
        # SkyrimNet's TTSPingManager health check speaks in the *player* voice,
        # which has no reference clip -- so the speaker slot arrives empty and
        # there is nothing to clone. It only asks whether the endpoint answers
        # with audio, so it is served straight off disk rather than costing the
        # engine a generation every 30 s while the game is running.
        logger.info(f"Health-check ping answered from {PING_WAV.name} (engine not called)")
        return str(PING_WAV)
    if upload_path is None:
        logger.error("No reference audio supplied; voice cloning requires one")
        return None

    # A local clip of the same name outranks the upload (see SAMPLES_DIR).
    override = _local_sample(upload_path)
    if override is not None:
        logger.info(
            f"Reference override: {Path(upload_path).name} -> "
            f"samples\\{Path(override).name}"
        )
        upload_path = override

    try:
        ref, digest = reference_path(upload_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Reference preparation failed: {exc}")
        return None

    transcript = reference_transcript(digest, ref) if USE_REFERENCE_TEXT else None

    try:
        top_p_value = float(top_p) if top_p and 0.0 < float(top_p) <= 1.0 else None
    except (TypeError, ValueError):
        top_p_value = None

    # SkyrimNet HiggsInterface passes configured temperature in slot 23 ('linear')
    try:
        temp_val = float(linear) if linear is not None else None
        temperature_value = temp_val if temp_val is not None and temp_val > 0.0 else None
    except (TypeError, ValueError):
        temperature_value = None

    # Slot 21 ('min_k') is top_k if specified
    try:
        top_k_val = int(min_k) if min_k is not None else None
        top_k_value = top_k_val if top_k_val is not None and top_k_val >= 0 else None
    except (TypeError, ValueError):
        top_k_value = None

    # Slot 22 ('min_p') is repetition_penalty if specified
    try:
        rep_val = float(min_p) if min_p is not None else None
        repetition_penalty_value = rep_val if rep_val is not None and rep_val > 0.0 else None
    except (TypeError, ValueError):
        repetition_penalty_value = None

    # Slot 24 ('confidence') is text_chunk_size if specified
    try:
        chunk_sz = int(confidence) if confidence is not None else None
        text_chunk_size_value = chunk_sz if chunk_sz is not None and chunk_sz > 0 else None
    except (TypeError, ValueError):
        text_chunk_size_value = None

    # Slot 25 ('quadratic') is text_chunk_mode if specified
    text_chunk_mode_value = str(quadratic).strip() if quadratic and str(quadratic).strip() not in ("False", "True", "None", "") else None

    seed_value = None
    if seed:
        try:
            seed_value = int(seed) % (2 ** 32)  # engine takes uint32
            if seed_value == 0:
                seed_value = None
        except (TypeError, ValueError):
            seed_value = None

    log_request(
        locals(),
        temperature_val=temperature_value,
        top_p_val=top_p_value,
        top_k_val=top_k_value,
        rep_pen_val=repetition_penalty_value,
        chunk_size_val=text_chunk_size_value,
        chunk_mode_val=text_chunk_mode_value,
        seed_val=seed_value,
    )

    try:
        gen_start = time.time()
        budget = token_budget(processed_text)

        attempts = []

        def add_attempt(text_arg: str, seed_arg: int | None, transcript_arg: str | None, max_tokens_arg: int | None, why_arg: str) -> None:
            cand = {
                "text": text_arg,
                "seed": seed_arg,
                "transcript": transcript_arg,
                "max_tokens": max_tokens_arg,
                "why": why_arg,
            }
            for prev in attempts:
                if (prev["text"] == cand["text"] and
                    prev["seed"] == cand["seed"] and
                    prev["transcript"] == cand["transcript"] and
                    prev["max_tokens"] == cand["max_tokens"]):
                    return
            attempts.append(cand)

        # Attempt 1: stock request with calculated budget
        add_attempt(engine_text, seed_value, transcript, budget,
                    f"initial request (budget={budget})" if budget else "initial request")

        # Fallback 1: ceiling budget if initial budget was lower
        if budget is not None and budget < MAX_TOKEN_CEIL:
            add_attempt(engine_text, seed_value, transcript, MAX_TOKEN_CEIL,
                        f"budget {budget} -> {MAX_TOKEN_CEIL}")

        # Fallback 2: drop control tokens if present
        if engine_text != processed_text:
            add_attempt(processed_text, seed_value, transcript, MAX_TOKEN_CEIL,
                        "dropping control tokens")

        # Fallback 3: re-roll / clear seed
        add_attempt(processed_text, None, transcript, MAX_TOKEN_CEIL,
                    "re-rolling the seed")

        # Fallback 4: drop transcript if reference transcript was used
        if transcript is not None:
            add_attempt(processed_text, None, None, MAX_TOKEN_CEIL,
                        "dropping transcript & re-rolling seed")

        audio_bytes = None
        last_cap_hit = None

        for i, attempt in enumerate(attempts):
            if i > 0:
                logger.warning(f"max_tokens hit before EOC; retrying ({attempt['why']})")
            try:
                audio_bytes = higgs_tts(attempt["text"], ref, attempt["transcript"],
                                        top_p_value, attempt["seed"],
                                        temperature=temperature_value,
                                        top_k=top_k_value,
                                        repetition_penalty=repetition_penalty_value,
                                        text_chunk_size=text_chunk_size_value,
                                        text_chunk_mode=text_chunk_mode_value,
                                        max_tokens=attempt["max_tokens"])
                break
            except CapHitError as exc:
                last_cap_hit = exc
                continue

        if audio_bytes is None:
            if last_cap_hit:
                raise last_cap_hit
            raise RuntimeError("Audio generation yielded no bytes")

        gen_s = time.time() - gen_start
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Generation failed: {exc}")
        return None

    trimmed = 0.0
    if TRIM_SILENCE:
        audio_bytes, trimmed = trim_trailing_silence(audio_bytes)
    else:
        logger.info("Trim silence: disabled (TRIM_SILENCE=0)")

    try:
        duration, sample_rate = wav_duration(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Engine returned something that is not a WAV: {exc}")
        return None

    if duration <= 0.0:
        logger.error("Generation returned empty audio")
        return None

    expected = expected_seconds(processed_text)
    if transcript is not None and expected >= 1.0 and duration < 0.35 * expected:
        # ICL collapse: a wrong/truncated transcript makes the model emit EOS
        # almost immediately. Only possible when a transcript was sent; retry
        # once transcript-free.
        logger.warning(
            f"Suspected ICL collapse ({duration:.2f}s for {len(processed_text)} chars, "
            f"expected ~{expected:.1f}s); retrying transcript-free"
        )
        _TRANSCRIPT_CACHE.pop(digest, None)
        try:
            audio_bytes = higgs_tts(engine_text, ref, None,
                                    top_p_value, seed_value,
                                    temperature=temperature_value,
                                    top_k=top_k_value,
                                    repetition_penalty=repetition_penalty_value,
                                    text_chunk_size=text_chunk_size_value,
                                    text_chunk_mode=text_chunk_mode_value,
                                    max_tokens=budget)
            if TRIM_SILENCE:
                audio_bytes, trimmed = trim_trailing_silence(audio_bytes)
            duration, sample_rate = wav_duration(audio_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Transcript-free retry failed: {exc}")
            return None
    elif expected >= 1.0 and duration < 0.35 * expected:
        logger.warning(
            f"Short output: {duration:.2f}s for {len(processed_text)} chars "
            f"(expected ~{expected:.1f}s)"
        )

    max_allowed = max(10.0, expected * 2.2 + 3.0)
    if expected >= 0.5 and duration > max_allowed:
        logger.warning(
            f"Runaway repetition loop detected ({duration:.2f}s audio for expected ~{expected:.1f}s, cap {max_allowed:.1f}s); "
            f"truncating and fading out."
        )
        audio_bytes, truncated = truncate_runaway_audio(audio_bytes, max_allowed)
        if truncated:
            duration, sample_rate = wav_duration(audio_bytes)

    # The engine returns 16-bit PCM at 24 kHz mono -- exactly what the mod
    # already consumes -- so the bytes go straight to disk.
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.write(audio_bytes)
    temp_file.close()

    total_s = time.time() - request_start
    logger.info(
        f"Done: {duration:.2f}s audio @ {sample_rate} Hz | engine {gen_s:.2f}s | "
        f"total {total_s:.2f}s | RTF {duration / gen_s if gen_s else 0:.2f} | "
        f"budget {budget or 'off'} | trimmed {trimmed:.2f}s | "
        f"ref {digest[:12]} | {temp_path}"
    )
    return temp_path


# --------------------------------------------------------------------------
# Gradio interface -- 29 positional inputs, exactly the Zonos order
# --------------------------------------------------------------------------

api_inputs = [
    gr.Textbox(label="Model", value="higgs"),                           # 0
    gr.Textbox(label="Text"),                                           # 1
    gr.Textbox(label="Language", value="en-us"),                        # 2
    gr.File(label="Speaker Audio"),                                     # 3
    gr.File(label="Prefix Audio", visible=False),                       # 4
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Happiness", visible=False),     # 5
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Sadness", visible=False),       # 6
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Disgust", visible=False),       # 7
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Fear", visible=False),          # 8
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Surprise", visible=False),      # 9
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Anger", visible=False),         # 10
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Other", visible=False),         # 11
    gr.Slider(minimum=0, maximum=1, value=0.2, label="Neutral", visible=False),        # 12
    gr.Slider(minimum=0.5, maximum=1.0, value=0.7, label="VQ Score", visible=False),   # 13
    gr.Slider(minimum=20000, maximum=25000, value=24000, label="Fmax (Hz)", visible=False),  # 14
    gr.Slider(minimum=20, maximum=150, value=45, label="Pitch Std", visible=False),    # 15
    gr.Slider(minimum=0, maximum=50, value=14.6, label="Speaking Rate", visible=False),  # 16
    gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall", visible=False),   # 17
    gr.Checkbox(value=True, label="Denoise Speaker", visible=False),                   # 18
    gr.Slider(minimum=1, maximum=10, value=3, label="CFG Scale", visible=False),       # 19
    gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.01, label="Top P"),          # 20
    gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),                # 21 (min_k slot)
    gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.01, label="Repetition Penalty"), # 22 (min_p slot)
    gr.Slider(minimum=0.05, maximum=2.0, value=0.8, step=0.05, label="Temperature"),  # 23 (linear slot)
    gr.Number(value=1024, label="Text Chunk Size"),                     # 24 (confidence slot)
    gr.Textbox(value="default", label="Text Chunk Mode"),               # 25 (quadratic slot)
    gr.Number(value=123, label="Seed"),                                 # 26
    gr.Checkbox(value=False, label="Randomize Seed", visible=False),                   # 27
    gr.Textbox(value="[]", label="Unconditional Keys", visible=False),                 # 28
]


# --------------------------------------------------------------------------
# File-input sanitising
#
# The ping described at PING_TEXT arrives with an *empty* path in the speaker
# slot, because the player voice has no clip to upload. Gradio resolves "" to
# the process CWD and rejects the request during preprocess -- before
# generate_audio is ever reached:
#
#   InvalidPathError: Cannot move <cwd> to the gradio cache dir because it was
#   not uploaded by a user.
#
# The client only sees `event: error, data: null`, and the wrapper log gets a
# traceback with no request line to explain it. Nothing downstream can recover
# from that, so the payload is cleaned one layer up: a file input that does not
# name a real file becomes None, which every component here already handles.
# --------------------------------------------------------------------------


def patch_file_input_sanitiser() -> None:
    """Drop file inputs Gradio refuses, instead of failing the whole request.

    Gradio walks a payload of any shape looking for file objects, so rather
    than trying to predict which shapes a client can produce, its own verdict
    is used: if the upload-folder check rejects the input, that input becomes
    None -- which every component here already handles -- and the offending
    payload is logged in full.

    Only preprocessing of client-supplied payloads is touched
    (check_in_upload_folder is Gradio's own marker for exactly that); outputs
    and internally-produced files go through untouched.
    """
    from gradio import processing_utils
    from gradio.exceptions import InvalidPathError

    original = processing_utils.async_move_files_to_cache

    async def sanitising_move(data, block, *args, **kwargs):
        if not kwargs.get("check_in_upload_folder"):
            return await original(data, block, *args, **kwargs)
        try:
            return await original(data, block, *args, **kwargs)
        except InvalidPathError as exc:
            logger.warning(
                f"Dropping rejected file input "
                f"{getattr(block, 'label', None) or type(block).__name__}={data!r} ({exc})"
            )
            return await original(None, block, *args, **kwargs)

    processing_utils.async_move_files_to_cache = sanitising_move
    logger.info("File-input sanitiser installed")


def build_app():
    # .queue() is critical: the Zonos client flow goes through /gradio_api/call
    # + SSE polling, which only exists when the queue is enabled. The default
    # concurrency of 1 also matches the engine, which serialises requests per
    # model behind an internal lock.
    return gr.Interface(
        fn=generate_audio,
        inputs=api_inputs,
        outputs=gr.Audio(label="Generated Audio"),
        title="Higgs Audio v3 TTS Zonos-Compatible Wrapper",
        description="Higgs Audio v3 TTS 4B (audio.cpp, GGML) behind the Zonos Gradio API.",
        api_name="generate_audio",
    ).queue()


def attach_health(demo) -> None:
    """Expose /health on the Gradio FastAPI app.

    Inserted at the front of the router so it cannot be shadowed by Gradio's
    own catch-all routes.
    """
    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRoute

    def transcriber_live_status() -> str:
        """Boot state, downgraded by a live TCP probe so a transcriber that died
        *after* boot shows as "down". A refused connect is instant, so the probe
        adds no meaningful latency.

        WINDOWS: host and port come from WHISPER_API_URL rather than the
        container's hardcoded 127.0.0.1:8080.
        """
        if WHISPER_STATUS != "ok":
            return WHISPER_STATUS
        import socket
        parsed = urlparse(WHISPER_API_URL)
        try:
            with socket.create_connection(
                (parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.3
            ):
                return "ok"
        except OSError:
            return "down"

    async def health():
        return JSONResponse(
            {
                "status": "ok" if READY else "loading",
                "whisper": transcriber_live_status(),
                "transcriber": WHISPER_API_URL if WHISPER_STATUS == "ok" else None,
                "reference_text": USE_REFERENCE_TEXT,
                "engine": HIGGS_URL,
                "engine_info": ENGINE_INFO,
                "quant": os.environ.get("HIGGS_QUANT", "q8"),
                "cached_references": len(_REF_CACHE),
            },
            status_code=200 if READY else 503,
        )

    demo.app.router.routes.insert(0, APIRoute("/health", health, methods=["GET"]))
    logger.info("/health endpoint attached")


if __name__ == "__main__":
    logger.info("Initializing Higgs3 wrapper ...")
    boot_start = time.time()
    try:
        initialize()
    except Exception as exc:  # noqa: BLE001
        # WINDOWS: no bash supervisor to restart us, so a failed boot must exit
        # loudly rather than leave a half-dead process the mod can still reach.
        logger.error(f"Startup failed: {exc}")
        sys.exit(1)

    patch_file_input_sanitiser()
    app = build_app()
    # Only now do we bind the port -- the mod races instances and must never
    # win against a box that is still loading.
    app.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=False,
        prevent_thread_lock=True,
    )
    attach_health(app)
    logger.info(f"READY: Gradio listening on {SERVER_HOST}:{SERVER_PORT} "
                f"({time.time() - boot_start:.1f}s since wrapper start)")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down")
