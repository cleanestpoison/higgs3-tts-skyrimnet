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

## Control tags

Higgs v3 takes control tokens shaped `<|category:value|>` — 43 of them, across
emotion (21), prosody (10), style (3) and sound effects (9). SkyrimNet strips
almost every special character before a line reaches the TTS, so angle brackets
and pipes never survive the trip. The LLM is therefore prompted to write
**ALL-CAPS tags in square brackets** mirroring the same structure, and the
wrapper rewrites them:

| written by the LLM | sent to the engine |
| --- | --- |
| `[EMOTION-FEAR]` | `<|emotion:fear|>` |
| `[STYLE-WHISPERING]` | `<|style:whispering|>` |
| `[PROSODY-SPEED_SLOW]` | `<|prosody:speed_slow|>` |
| `[PROSODY-PAUSE]` | `<|prosody:pause|>` |
| `[SFX-LAUGHTER]` | `<|sfx:laughter|>Haha,` |

### Wiring the tags up in SkyrimNet

Two files in this repo do this, and all three steps are needed — tags will not
work with any one of them missing.

**1. Select Chatterbox as the TTS system.**

In SkyrimNet's settings, set the TTS system to **Chatterbox**. This wrapper
serves the Zonos API, but the tag pipeline rides on SkyrimNet's Chatterbox
integration — that is the backend with an audio-tags feature and a configurable
allowed-tag list, which is what makes the rest of this possible. It is also why
[`0650_audio_tags.prompt`](0650_audio_tags.prompt) branches on
`get_actor_tts(npc.UUID) == "chatterbox"`.

**2. Enable audio tags and register the allowed list.**

In the Chatterbox settings, turn **audio tags on**. Then under **Advanced
settings**, paste in the tags from
[`skyrimnet-allowed-audiotags-chatterbox.md`](skyrimnet-allowed-audiotags-chatterbox.md)
— one per line, exactly as they appear in that file. SkyrimNet drops anything not
on this list before it ever reaches the TTS, so a tag missing here simply never
arrives.

**3. Install the prompt.**

Copy [`0650_audio_tags.prompt`](0650_audio_tags.prompt) to:

```
submodules/user_final_instructions/0650_audio_tags.prompt
```

overwriting the original file. This is what instructs the LLM to write the tags
in the first place.

> **Not every supported tag is included.** The lists here are a deliberate
> subset — some of Higgs v3's tags performed noticeably worse than others in
> practice and were left out rather than shipped as unreliable. The full
> catalogue is on the model card:
> [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b).
> Add any of them back to both files if they work better for you than they did
> here.

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
