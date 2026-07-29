# Installing

Two ways to do this. **Automatic** is one command and handles everything.
**Manual** is the same steps done by hand, for when a download fails, a proxy
blocks the script, or you already have the files somewhere.

See [README.md](README.md) for what this actually is and how to tune it.

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA, compute capability **7.5+** (RTX 20/30/40/50 series), driver **580+** |
| **VRAM** | ~6 GB free while running |
| **OS** | Windows 10 1803+ / Windows 11 |
| **Python** | 3.12, installed from [python.org](https://www.python.org/downloads/) with the `py` launcher |
| **ffmpeg** | `ffmpeg.exe` on PATH, or `$HIGGS_FFMPEG` set in `config.ps1` |
| **Disk** | ~7 GB (5.5 GB payload + ~150 MB venv + downloaded archives) |

Pascal and Volta cards will not work — the prebuilt engine targets 7.5 and newer.

---

## Automatic

```powershell
.\Setup.ps1
```

That's it. It downloads the engine and the model, creates the venv, installs the
Python dependencies, and verifies the whole chain. Roughly 5.5 GB the first time.

Safe to re-run. Every step is skipped if already satisfied, and interrupted
downloads **resume** rather than starting over, so a dropped connection costs you
only the remainder.

Pick a different CUDA build if the default misbehaves:

```powershell
.\Setup.ps1 -EngineProfile portable   # if 'balance' crashes on startup
```

`balance` (default), `fast` and `portable` differ **only** in which CPU
instruction sets their kernels use — AVX2, native, and none respectively. The
GPU requirement is identical for all three.

Rebuild the venv from scratch without re-downloading the 4.7 GB model:

```powershell
.\Setup.ps1 -Force
```

Then skip to [Starting it](#starting-it).

---

## Manual

Do these four steps, then run `.\Setup.ps1` — it will find everything in place,
skip the downloads, and just build the venv.

### 1. Engine binaries

Download
**[audiocpp-windows-cuda-balance-27d87ba.zip](https://github.com/0xShug0/audio.cpp/releases/download/release-0.4.2/audiocpp-windows-cuda-balance-27d87ba.zip)**
(247 MB) from the audio.cpp
[release-0.4.2](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.4.2)
page and extract it into `engine\`.

Substitute `fast` or `portable` for `balance` in the filename if you want a
different profile.

It contains exactly:

```
audiocpp_server.exe      142,652,928 bytes
audiocpp_cli.exe         142,758,400 bytes
MSVCP140.dll                 557,728
VCRUNTIME140.dll             124,544
VCRUNTIME140_1.dll            49,792
VCOMP140.DLL                 193,152
README.md                              (not needed)
```

### 2. CUDA runtime libraries

Download
**[audiocpp-windows-cuda-runtime.zip](https://github.com/0xShug0/audio.cpp/releases/download/release-0.4.2/audiocpp-windows-cuda-runtime.zip)**
(575 MB) from the same release and extract it into `engine\` as well.

This one is shared by all three profiles, and contains:

```
cublasLt64_13.dll        460,301,424 bytes
cufft64_12.dll           256,092,784
cublas64_13.dll           51,870,320
README.md                              (not needed)
```

> Both zips are flat — no wrapper folder — so "extract here" into `engine\` is
> correct. If your unzip tool creates `engine\audiocpp-windows-cuda-balance\`,
> move the contents up one level.

### 3. Model specs

Nothing to do — `model_specs\` is already in this repo and `Setup.ps1` copies it
into `engine\model_specs\`.

Worth knowing, because it is the one thing that trips people up: these JSON files
are **not** in the release zips. They ship only in the audio.cpp *source* tree,
and the engine reads them from its working directory at runtime. If you assemble
`engine\` purely from the binary archives, the engine will not start. See
[`model_specs/NOTICE.md`](model_specs/NOTICE.md).

### 4. Model weights

Download
**[higgs-audio-v3-tts-4b-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf)**
(4.7 GB) from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Higgs-Audio-v3-TTS-4B-GGUF)
and save it to `models\higgs-audio-v3-tts-4b-q8_0.gguf`.

The `bf16` file in that folder is the same model unquantised — 8.5 GB and no
better for this purpose. Take the q8_0.

### 5. Finish

```powershell
.\Setup.ps1
```

It verifies every file above, then creates the venv and installs the wrapper's
dependencies (~150 MB).

### Expected layout when done

```
engine\
  audiocpp_server.exe        audiocpp_cli.exe
  cublas64_13.dll            cublasLt64_13.dll       cufft64_12.dll
  MSVCP140.dll               VCRUNTIME140.dll        VCRUNTIME140_1.dll
  VCOMP140.DLL
  model_specs\               35 .json files
models\
  higgs-audio-v3-tts-4b-q8_0.gguf     ~4.7 GB
venv\
```

---

## Starting it

```powershell
.\Start_Higgs3.ps1
```

or double-click **`Start.bat`**.

Startup takes about 40 seconds: the GGUF maps and the engine binds in ~35 s, then
a warmup generation runs before the port opens. That ordering is deliberate — the
port staying shut until the model is genuinely ready is what stops SkyrimNet
connecting to a half-loaded server and failing its first line.

Options:

```powershell
.\Start_Higgs3.ps1 -Port 7864        # if 7863 is taken
.\Start_Higgs3.ps1 -ReferenceText    # ICL cloning via transcripts
```

Both halves run in the one window, and closing it stops the engine too.

### Point SkyrimNet at it

Set SkyrimNet's TTS URL to:

```
http://127.0.0.1:7863
```

Port 7863 is chosen so OmniTTS (7860), Qwen3-TTS (7861) and pocket-tts (7862) can
all coexist — switch between them by changing that URL. Each holds its own VRAM
while running, though; this one wants ~6 GB.

### Enable the emotion / prosody tags

Optional, but it is most of what makes this worth running. Three steps, all
required:

1. **Set the TTS system to Chatterbox** in SkyrimNet's settings. The wrapper
   serves the Zonos API, but the tag pipeline rides on SkyrimNet's Chatterbox
   integration — that is the backend with an audio-tags feature and a
   configurable allowed-tag list.

2. **Enable audio tags** in the Chatterbox settings, then under **Advanced
   settings** paste in every tag from
   [`skyrimnet-allowed-audiotags-chatterbox.md`](skyrimnet-allowed-audiotags-chatterbox.md),
   one per line exactly as written. SkyrimNet discards anything not on this list
   before it reaches the TTS.

3. **Copy [`0650_audio_tags.prompt`](0650_audio_tags.prompt)** to
   `submodules/user_final_instructions/0650_audio_tags.prompt`, overwriting the
   original.

The included tags are a deliberate subset — some of Higgs v3's tags worked
noticeably worse than others and were left out. The full catalogue is on the
model card: [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b).

Details and the rewrite rules are in
[README.md](README.md#wiring-the-tags-up-in-skyrimnet).

### Check it is alive

```powershell
curl http://127.0.0.1:7863/health
```

Reports the loaded model and whether you are in transcript-free or ICL mode.

---

## When setup fails

**`nvidia-smi not found`** — no NVIDIA driver on PATH. Install/update the GPU
driver; you need 580 or newer.

**`Python 3.12 not found`** — installed without the `py` launcher, or a different
version. `py -3.12 --version` must work.

**`ffmpeg not found`** — put `ffmpeg.exe` on PATH or set `$HIGGS_FFMPEG` in
`config.ps1`. Not optional: every reference clip is normalised through it,
because clips in circulation include MP3-in-WAV that most decoders mis-report.

**`audiocpp_server.exe could not start`** — a CUDA runtime DLL is missing or the
driver is too old. Confirm all three `cu*.dll` files are in `engine\` at the sizes
listed above, then check the driver version.

**Download keeps failing** — just re-run `.\Setup.ps1`; it resumes. If the network
is the problem entirely, use the [manual steps](#manual).

**`Model ... is only N GB; expected ~4.7 GB`** — truncated download. Delete
`models\higgs-audio-v3-tts-4b-q8_0.gguf` and re-run.

**Engine exits immediately at startup** — the launcher prints the tail of
`logs\engine-*.err.log`. If it mentions an illegal instruction, try
`.\Setup.ps1 -EngineProfile portable`.

**Port already in use** — the launcher names the process holding it. Stop that or
use `-Port`. Both 7863 and the engine port are checked before the model loads.

More in [README.md](README.md#troubleshooting).

## Reclaiming disk

`dl\` keeps the downloaded archives (~820 MB) so a re-run does not fetch them
again. Delete it once you are running.
