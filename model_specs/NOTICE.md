# model_specs

These JSON files are vendored from [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp)
(`release-0.4.2`), Copyright ShugoAI LLC, licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**They are unmodified.** `Setup.ps1` copies them into `engine\model_specs\`,
where the engine reads them from its working directory at runtime.

They live here rather than being downloaded because they ship in the audio.cpp
*source* tree and not in the prebuilt Windows release archives — so fetching
only the binaries leaves the engine without them. They total ~140 KB, so
vendoring costs nothing and removes a network dependency from setup.

Only `higgs_audio_tts.json` is used by this project; the rest are kept as the
upstream set so the folder can be refreshed wholesale from a newer audio.cpp
tag without picking through it.
