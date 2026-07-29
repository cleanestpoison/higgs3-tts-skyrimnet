# Local reference clips

Drop a clip in here named after the reference SkyrimNet uploads, and it is used
instead of that upload.

## Why

SkyrimNet resamples every reference clip to **16 kHz** before sending it —
including the 44.1 kHz files in its own `voice-samples\` folder. That discards
everything above 8 kHz. Higgs runs at 24 kHz, so it has roughly 12 kHz of
headroom the reference never fills, and a band-limited reference gives the clone
a duller, more telephone-like timbre than it needs to have.

Clips in this folder bypass that: the wrapper reads them straight off disk at
whatever rate they are stored.

## Naming

The name to match is the **filename of the upload**, which the log records on
every request:

    Incoming request: ... speaker_audio='...\gradio\<hash>\femalenord.wav', ...

So `femalenord.wav` here overrides that upload. In practice:

| Upload name | Comes from | A file here affects |
|---|---|---|
| `femalenord.wav`, `malecommoner.wav` | vanilla voicetype | every NPC with that voicetype |
| `serana.wav`, `aylin.wav` | a named character | that character |

Matching is case-insensitive and ignores the extension — `FemaleNord.flac`
matches an upload called `femalenord.wav`. Accepted formats: `.wav`, `.flac`,
`.mp3`, `.ogg`, `.m4a`, `.opus` (anything ffmpeg reads, since every reference is
normalised to mono 24 kHz on the way in regardless).

When a file matches, the log says so:

    Reference override: femalenord.wav -> samples\femalenord.wav

Files can be added or replaced while the server is running; the folder is
re-indexed when its contents change.

## What to put here

Quality of the *voice* matters more than the sample rate. A random 44.1 kHz clip
under `femalenord.wav` makes every female Nord in Skyrim sound like that person.

To keep vanilla identity while gaining bandwidth, re-extract the voice lines
from Skyrim's BSAs — the source audio is 44.1 kHz mono inside `.fuz`/`.xwm`, so
the 16 kHz is purely SkyrimNet's downsampling, not a limit of the game files.
Same voice, ~2.75x the bandwidth.

Clips are trimmed to `HIGGS_REF_MAX_SECONDS` (default 60 s) when normalised.
Clean speech, no music or combat noise, is worth more than length.

## Turning it off

Set `$HIGGS_SAMPLES = 0` in `config.ps1`. `$HIGGS_SAMPLES_DIR` points the lookup
at a different folder.
