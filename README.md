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

## Two processes, one window

The launcher starts both halves and stops the engine when it exits:

| Port | Process | Role |
|------|---------|------|
| 7863 | `python higgs3_zonos_wrapper.py` | Zonos API SkyrimNet talks to |
| 8081 | `audiocpp_server.exe` | the model, loopback only |

If the engine fails to load, the launcher says so and prints its log rather than
leaving the wrapper waiting 15 minutes on a process that is already dead.

## Voice cloning

Higgs v3 clones **transcript-free** by default — hand it a reference clip and it
copies the voice, no transcript needed. That is the default here and it needs no
transcriber at all.

`-ReferenceText` instead transcribes each reference clip once (via SimpleParakeet
on port 8210) and feeds the text to the model as a worked example. It can improve
similarity on some voices, but a *wrong* transcript can collapse a generation to
a fraction of a second. The wrapper detects that and retries transcript-free, so
the cost is latency rather than a bad line. Off unless you ask for it.

Clips are normalised to 24 kHz mono PCM and cached in `refcache\` by content
hash, so the same voice sample is only ever converted once.

Check which mode you actually got:

```powershell
curl http://127.0.0.1:7863/health
```

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

The brackets leave with the tag rather than being left behind as markup the
engine would read aloud. The separator inside is optional and may be anything the
mod leaves behind — `EMOTION-FEAR`, `EMOTION_FEAR`, `EMOTION FEAR` and
`EMOTIONFEAR` all resolve, bracketed or not. Matching is uppercase-only on
purpose: "emotion" and "style" are ordinary English words, and the caps
requirement is what stops dialogue being read as markup.

Three rules are enforced here rather than left to the LLM — two from the model
card, one measured against this engine:

* **Emotion, style and speed/pitch prosody are sentence-level.** A tag written
  mid-sentence is moved to the front of that sentence. At most one emotion and
  one style survive per sentence.
* **Sound effects must be immediately followed by onomatopoeia, no space** — a
  bare `<|sfx:laughter|>` does nothing. The onomatopoeia is injected, using the
  spellings the model card documents for all nine effects (`ONOMATOPOEIA` in the
  wrapper). Several have a documented alternative — `Hehe` for laughter, `Sob`
  for crying, `Uh` for sigh — noted inline and worth swapping if an effect comes
  out wrong.
* **Pause tags must sit between two words.** A `PROSODY-LONG_PAUSE` that lands at
  the start or the end of a line is dropped. It would be inaudible there anyway,
  and measured against audio.cpp it is worse than useless: the decoder never
  emits EOC, runs to its 2048-step cap and the request fails outright with
  "reached max_tokens before EOC", returning no audio at all.

Anything shaped like a tag but not in the catalogue is **deleted, never passed
through** — unrecognised markup gets read aloud, and an NPC saying "emotion fear"
out loud is the failure worth engineering against. Every drop is logged at
WARNING. Tokens are kept out of the length heuristics, so they cannot skew the
budget or the collapse detection. `HIGGS_CONTROL_TAGS=0` disables the rewrite and
strips tokens instead.

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

### The language field

The Zonos `language` field is dead weight here (Higgs takes no language input),
so the wrapper reuses it as a **sentence-level tag channel**: set it to
`<|emotion:anger|>` or `EMOTION-ANGER` and that tag is prepended to every line.
Anything else — including a plain `en-us` — is ignored.

## Running alongside your other TTS servers

Port 7863 was chosen so OmniTTS keeps 7860, Qwen3-TTS keeps 7861 and
skyrimnet-pocket-tts keeps 7862, and all four can run at once — switch SkyrimNet
between them by changing its TTS URL.
Nothing here touches the Chatterbox, OmniTTS, Qwen3TTS, Pocket-TTS or Parakeet
folders.

Note that each server holds its own VRAM while running. This one wants ~6 GB.

## Performance

Measured on an RTX 5090, q8_0 weights, transcript-free cloning, driven through
the same Zonos Gradio API SkyrimNet uses. "Cold" is the first ever use of a
voice (normalise the clip + encode the reference); "warm" is any later line in
that voice. Latency is the median of 5 runs.

| Case | Audio | Latency | RTF |
|---|---|---|---|
| Cold voice (91-char line) | 5.4–5.9 s | 2.05–2.21 s | ~2.6× |
| Warm, short (12 chars) | 1.3 s | 0.80 s | 1.7× |
| Warm, medium (91 chars) | 5.2 s | 1.20 s | 4.4× |
| Warm, long (232 chars) | 18.2 s | 2.64 s | 6.9× |
| Warm, alternating voices | 5.6 s | 1.85 s | 3.0× |

Every line finishes far faster than it plays, so the model is never the
bottleneck in conversation — even a 232-character speech is ready in 2.6 s.

**Where the time actually goes.** The engine itself runs at **6.9× realtime**
and the Python wrapper adds **14 ms**. Everything else is Gradio transport.

Two things that look like they should matter, and do not:

* **Reference file size is irrelevant.** A 0.3 MB clip and a 9.7 MB clip both
  land at ~1.3 s — the upload is trivial over loopback, and the clip is
  normalised and encoded only once, then cached.
* **Cold voices only cost ~0.9 s extra**, once. The refcache and the engine's
  reference-encode cache mean you pay it a single time per voice, ever.

One thing that *does* matter:

* **Switching voices costs about +0.57 s per line** versus repeating the same
  one (1.29 s → 1.85 s, measured back-to-back on two already-cached voices).
  This is prompt prefill, not encoding, so `$HIGGS_REF_CACHE_SLOTS` does not
  remove it. It is the normal cost of a scene alternating actors, and it is
  still comfortably faster than playback.

Startup is about 40 seconds: the GGUF maps and the engine binds in ~35 s, then
the warmup generation takes ~0.7 s before the port opens.

## Tuning

`config.ps1` exposes every setting the Docker image had. The ones worth knowing:

**`$HIGGS_REF_MAX_SECONDS`** (default `60`) caps how much of a reference clip is
used. Unlike the Qwen3 port, Higgs encodes the reference once and caches it in
the engine, so long clips cost far less here — the image's own 60 s default is
kept.

**`$HIGGS_TOKEN_BUDGET`** (default `0`) enables a text-derived length cap. Leave
it off. audio.cpp treats hitting the cap as a **hard error with no audio** rather
than a graceful flush, so a miscalibrated budget can only turn a slow-spoken line
into a failed request; it can never rescue a good one. The engine's own 2048-step
ceiling (~80 s of audio) still bounds pathological cases.

**`$HIGGS_TEMPERATURE` / `$HIGGS_TOP_K`** (default empty) leave audio.cpp's own
`higgs_audio_tts` defaults in place (0.8 / 30 / top_p 0.8). These are deliberately
narrower than the Python client's and less prone to cutting a line short.

**`$HIGGS_REF_CACHE_SLOTS`** (default `8`) is how many distinct voices the engine
keeps encoded in memory. A busy scene alternates actors; the engine's own default
of 1 would re-encode on every switch.

## Troubleshooting

**Port already in use** — the launcher names the process holding it. Stop that,
or use `-Port`. Both 7863 and 8081 are checked before the model loads.

**CUDA out of memory** — the q8 model wants ~6 GB. Close Skyrim, or stop the
other TTS servers, and retry.

**`ffmpeg not found`** — every reference clip is normalised through ffmpeg
(clips in circulation include MP3-in-WAV, which most decoders mis-report). Put
`ffmpeg.exe` on PATH or set `$HIGGS_FFMPEG` in `config.ps1`.

**Engine exits at startup** — the launcher prints the tail of
`logs\engine-*.err.log`. A missing CUDA runtime DLL and an out-of-date driver
both surface here.

**A line comes out much shorter than expected** — logged as `Short output:`. With
transcript-free cloning this is the model, not a bad transcript; try a different
seed or a cleaner reference clip.

Full logs land in `logs\`, one pair of files per run (wrapper + engine).

Generated WAVs are written to `%TEMP%` and left there for SkyrimNet to read, the
same as the Docker image did. Windows Disk Cleanup clears them.

## Layout

```
INSTALL.md                   install guide, incl. manual downloads
Setup.ps1                    one-time installer / verifier
Start.bat                    double-click launcher
Start_Higgs3.ps1             launcher (engine + wrapper)
config.ps1                   all settings
higgs3_zonos_wrapper.py      the Zonos-API server
0650_audio_tags.prompt       SkyrimNet prompt -- goes in the mod, not here
skyrimnet-allowed-audiotags-chatterbox.md
                             the allowed-tag list to paste into SkyrimNet
warmup_ref.wav               boot warmup clip
model_specs\                 vendored from audio.cpp; copied into engine\
engine\                      audiocpp_server.exe, CUDA DLLs                 (downloaded)
engine\server.json           generated on every launch from config.ps1
models\                      higgs-audio-v3-tts-4b-q8_0.gguf                (downloaded)
dl\                          the downloaded archives; deletable
refcache\                    normalised reference clips, by content hash
logs\                        one wrapper log + one engine log per run
```

Everything marked *(downloaded)* is fetched by `Setup.ps1` and is not in the
repo — the GGUF alone is 4.7 GB, and the engine and weights belong to their
respective upstreams.

`model_specs\` is the exception: the engine needs it at runtime but it ships only
in the audio.cpp *source* tree, never in the prebuilt release archives, so
assembling `engine\` from the binaries alone leaves the engine unable to start.
It is 140 KB of JSON, so it is carried here instead. See
[`model_specs/NOTICE.md`](model_specs/NOTICE.md).

## Differences from the Docker image

* **No whisper.cpp** — the image ran `whisper-server` on 8080 with
  `large-v3-turbo` for SkyrimNet's STT. SimpleParakeet already serves that on
  8210 here, and Higgs v3 clones transcript-free, so the whole STT half is gone.
  Saves a 1.5 GB download, a process, and ~2.2 GB of VRAM. `-ReferenceText`
  points at Parakeet if you want ICL transcripts.
* **No vast.ai idle watchdog** — it self-stopped a *rented cloud instance* after
  30 minutes of quiet. On your own machine that is just a server that dies
  during a long conversation.
* **No bash supervisor** — a crash is printed rather than silently restarted.
* **Binds loopback, not `0.0.0.0`** — there is no container to escape, and this
  avoids a Windows Firewall prompt. Set `$HIGGS_HOST` if SkyrimNet runs on
  another PC.
* **The engine is watched during startup.** The wrapper alone would wait its full
  900-second timeout for an engine process that had already died; the launcher
  notices immediately and prints the engine's own error.
* **ffmpeg is configurable and its absence is reported in those words** — the
  image could assume `/usr/bin/ffmpeg`; on Windows a missing ffmpeg is the most
  likely setup mistake, and it would otherwise surface as a bare `FileNotFoundError`.
* **The `/health` transcriber probe follows `WHISPER_API_URL`** rather than the
  image's hardcoded `127.0.0.1:8080`, which would have reported a healthy
  Parakeet as down.
* **ffmpeg runs without flashing a console window** on each new reference clip.

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
