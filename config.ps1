# Higgs Audio v3 TTS configuration.
#
# Every value here maps to an environment variable the original Docker image
# used, so anything documented for elbios/higgs3-whisper applies unchanged.
# Edit this file; Start_Higgs3.ps1 reads it on every launch.

# --- Networking -------------------------------------------------------------

# Port SkyrimNet points at.
# SkyrimNet TTS URL:  http://127.0.0.1:7860
$HIGGS_PORT = 7860

# Interface the wrapper binds. Loopback keeps it off the LAN and avoids a
# Windows Firewall prompt. Use "0.0.0.0" only if SkyrimNet runs on another PC.
$HIGGS_HOST = "0.0.0.0"

# The audio.cpp engine. Loopback only -- nothing but the wrapper should reach
# it. 8081 is deliberately avoided: SkyrimSE itself listens there.
$HIGGS_ENGINE_PORT = 8084

# --- Model ------------------------------------------------------------------

# The GGUF baked into the Docker image: Higgs Audio v3 TTS 4B at q8_0 (~4.7 GB,
# ~6 GB VRAM in use). The file carries its own tokenizer, config and model spec.
$HIGGS_MODEL_FILE = "models\higgs-audio-v3-tts-4b-q8_0.gguf"

# Engine CPU worker threads. The work is overwhelmingly on the GPU; this only
# covers tokenisation and audio I/O, so more is not better.
$HIGGS_ENGINE_THREADS = 4

# Distinct reference voices whose encoded form the engine keeps resident. A
# playthrough alternates actor voices, so the image raised this from the
# engine default of 1. Each slot is small; 8 covers a busy scene.
$HIGGS_REF_CACHE_SLOTS = 8

# Fail a request with 503 once the model has been busy this long, in ms.
$HIGGS_BUSY_TIMEOUT_MS = 300000

# --- Voice cloning ----------------------------------------------------------

# Higgs v3 clones transcript-free, which is the default and needs no
# transcriber at all. Setting this to 1 transcribes each reference clip once
# and feeds the text to the model as a worked example (ICL). That can improve
# similarity on some voices, but it also makes a *wrong* transcript able to
# collapse a generation to a fraction of a second -- the wrapper detects that
# and retries transcript-free, at the cost of latency.
$HIGGS_USE_REFERENCE_TEXT = 0

# Where to transcribe reference clips when the above is 1. This is the
# SimpleParakeet server already running for SkyrimNet's External Whisper.
# It is used ONLY to read reference clips -- never your microphone, and never
# in the text-to-speech path. Ignored entirely when the above is 0.
$WHISPER_API_URL = "http://127.0.0.1:8210/v1/audio/transcriptions"
$WHISPER_WAIT_SECONDS = 20

# Cap on reference-clip length, in seconds. Clips are trimmed to this when
# normalised. The image default is 60; local voice samples in speakers\ run up
# to 120 seconds. Changing this re-derives the cache rather than serving clips
# trimmed under the old value.
$HIGGS_REF_MAX_SECONDS = 60

# Local reference clips, in samples\. A clip named after the file SkyrimNet
# uploads (`femalenord.wav`, `serana.wav` -- the name is in the request log) is
# used instead of that upload. Worth having because SkyrimNet resamples every
# reference to 16 kHz before sending it, including the 44.1 kHz clips in its own
# voice-samples\ folder, which costs everything above 8 kHz; Higgs runs at
# 24 kHz and has room for far more. See samples\README.md. Set to 0 to always
# use whatever the mod sends.
$HIGGS_SAMPLES = 1

# --- Generation tunables (engine defaults unless set) ------------------------

# Leave empty to use audio.cpp's own higgs_audio_tts defaults (temperature 0.8,
# top_k 30, top_p 0.8), which are deliberately narrower than the Python
# client's and less prone to cutting a line short. top_p is also settable
# per-request by SkyrimNet.
$HIGGS_TEMPERATURE = ""
$HIGGS_TOP_K = ""

# Text-derived length cap, off by default. audio.cpp treats hitting the cap as
# a hard error with no audio rather than a graceful flush, so a miscalibrated
# budget can only turn a slow-spoken line into a failed request. The engine's
# own 2048-step ceiling (~80 s of audio) still applies as the pathological
# bound. Set to 1 only if you see runaway generations.
$HIGGS_TOKEN_BUDGET = 0

# Trim trailing near-silence from generated lines. Off by default, as in the
# image.
$HIGGS_TRIM_SILENCE = 0

# --- Tools ------------------------------------------------------------------

# Full path to ffmpeg.exe, used to normalise reference clips to 24 kHz mono
# PCM. Leave empty to find it on PATH.
$HIGGS_FFMPEG = ""
