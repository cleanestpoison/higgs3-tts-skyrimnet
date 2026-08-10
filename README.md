# Higgs Audio v3 TTS for SkyrimNet

Native Windows port of the `elbios/higgs3-whisper` Docker image. No Docker, no WSL.

Higgs Audio v3 voice cloning behind the Zonos Gradio API that SkyrimNet already
speaks, so the mod talks to it without knowing anything changed.

| | |
|---|---|
| **Endpoint** | `http://127.0.0.1:7863` |
| **Health** | `http://127.0.0.1:7863/health` |
| **Model** | `higgs-audio-v3-tts-4b` q8_0 GGUF (~4.7 GB, ~6 GB VRAM) |
| **Engine** | [audio.cpp](https://github.com/0xShug0/audio.cpp) `audiocpp_server.exe`, CUDA |
| **Requires** | NVIDIA GPU, Python 3.12, ffmpeg on PATH |

## Setup

```powershell
.\Setup.ps1
```

Downloads the engine and the weights, creates the venv, and verifies the GPU and
ffmpeg. About **5.5 GB** the first time. Safe to re-run: finished steps are
skipped and interrupted downloads resume, so a failed run costs you nothing.

> **The 4.7 GB model is usually faster to fetch by hand.** Hugging Face often
> serves the script's download at a crawl while a browser gets full speed. If
> `Setup.ps1` looks stuck on the model, cancel it and grab
> **[higgs-audio-v3-tts-4b-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/blob/main/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf)**
> yourself, save it as `models\higgs-audio-v3-tts-4b-q8_0.gguf`, then re-run
> `.\Setup.ps1` — it will find the file, skip the download and carry on.

Needs an NVIDIA GPU with **compute capability 7.5+** (RTX 20/30/40/50) on
**driver 580+**, plus Python 3.12 and ffmpeg on PATH.

**→ [INSTALL.md](INSTALL.md)** for the full walkthrough, the CUDA build profiles,
manual download instructions, and what to do when setup fails.

## Running

```powershell
.\Start_Higgs3.ps1
```

or double-click **`Start.bat`**. Then point SkyrimNet's TTS at
`http://127.0.0.1:7863`, with the TTS system set to **Chatterbox** — see
[Wiring the tags up in SkyrimNet](#wiring-the-tags-up-in-skyrimnet).

The port does not open until the model is loaded and a warmup generation has
completed, so SkyrimNet cannot connect to a half-ready server and get a failed
first line.

Options:

```powershell
.\Start_Higgs3.ps1 -Port 7864        # different port
.\Start_Higgs3.ps1 -ReferenceText    # ICL cloning with reference transcripts
```

Settings live in `config.ps1`.

## Higher-quality reference clips

SkyrimNet resamples every reference clip to **16 kHz** before sending it — even
the 44.1 kHz files in its own `voice-samples\` folder. Higgs runs at 24 kHz, so
everything above 8 kHz is discarded before the codec ever sees it.

Drop a clip into `samples\` named after the file the mod uploads and it is used
instead, at full rate. The name is in the request log:

    Incoming request: ... speaker_audio='...\gradio\<hash>\femalenord.wav', ...

so `samples\femalenord.wav` re-voices every NPC with that voicetype, and
`samples\serana.wav` covers that one character. Matching ignores case and
extension; anything ffmpeg reads works. Clips can be added while the server
runs. Measured on one 8.87 s clip, this lifts the 8–12 kHz band from 0.01 % to
1.05 % of total energy — the sibilance and air the 16 kHz path removes.

See `samples\README.md` for what makes a good clip. Set `$HIGGS_SAMPLES = 0` to
always use whatever the mod sends. 

## Control tags

Higgs v3 takes control tokens shaped `<|category:value|>` — 43 of them, across emotion (21), prosody (10), style (3) and sound effects (9). SkyrimNet produces and consumes tags as square bracket syntax `[category-value]` and transforms it to higgs syntax internally:

| written by the LLM | sent to the engine |
| --- | --- |
| `[EMOTION-FEAR]` | `<\|emotion:fear\|>` |
| `[STYLE-WHISPERING]` | `<\|style:whispering\|>` |
| `[PROSODY-SPEED_SLOW]` | `<\|prosody:speed_slow\|>` |
| `[PROSODY-PAUSE]` | `<\|prosody:pause\|>` |
| `[SFX-LAUGHTER]` | `<\|sfx:laughter\|>Haha,` |

### Wiring the tags up in SkyrimNet

You pretty much need jsut to select Higgs as your tts and you are done.
One setting you might find optional is **Propagate Sentence Tags** it is enabled by default. When LLM returns multiple sentence for single dialogue SkyrimNet will divide it into segments. Usually higgs sentence level tags are present at the very beginning of first sentence. When it splits into multiple segments 2nd+ segment will loose those tags. This config allows to carry them over to subsequent segments.


**1. Select Higgs as the TTS system.**
In SkyrimNet's settings, set the TTS system to **Higgs**. This wrapper serves the Zonos API, but the tag pipeline now rides on SkyrimNet's Higgs integration — that is the backend with an audio-tags feature and a configurable allowed-tag list.

**2. Enable audio tags and register the allowed list.**

Higgs config has tags enabled by default. All [higgs supported tags](https://huggingface.co/bosonai/higgs-tts-3-4b#control-tokens) are included by default. Some of them distort voice too much, so if you see major changes in cloned voice go and check what tags were used and remove them from allowed tags config.

## Credit

Ported from [`elbios/higgs3-whisper`](https://hub.docker.com/r/elbios/higgs3-whisper)
by Elbios. Engine is [audio.cpp](https://github.com/0xShug0/audio.cpp) by ShugoAI
(Windows CUDA prebuilt, `release-0.4.2`); model is
[Higgs Audio v3](https://github.com/boson-ai/higgs-audio) by Boson AI, quantised
to q8_0 GGUF in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Higgs-Audio-v3-TTS-4B-GGUF).

`Setup.ps1` downloads both from those upstreams. The only upstream files carried
in this repo are `model_specs\*.json`, vendored unmodified from audio.cpp
(Apache-2.0, Copyright ShugoAI LLC) because the engine cannot start without them
and they are absent from the release archives.

## Licence

[Apache-2.0](LICENSE), Copyright 2026 cleanestpoison. This covers the launcher
scripts, the wrapper and the documentation — the parts written here.

It does **not** cover what `Setup.ps1` downloads. The audio.cpp engine is
Apache-2.0 (Copyright ShugoAI LLC) and the Higgs Audio v3 weights carry Boson
AI's own terms; both stay with their upstreams and neither is redistributed
here. `model_specs\*.json` is vendored unmodified from audio.cpp under the same
Apache-2.0 licence — see [`model_specs/NOTICE.md`](model_specs/NOTICE.md).
