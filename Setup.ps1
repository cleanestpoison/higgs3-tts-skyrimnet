<#
.SYNOPSIS
    One-time setup for Higgs Audio v3 TTS. Creates the venv and verifies the
    engine, model and tools are all present and working.

.DESCRIPTION
    Native Windows equivalent of building elbios/higgs3-whisper. Safe to re-run:
    each step is skipped if already satisfied.

    Unlike the Qwen3-TTS port there is no PyTorch here at all -- the engine is
    audio.cpp, a native CUDA binary, so the venv is only the HTTP wrapper and
    its dependencies (~150 MB rather than ~3 GB).

.PARAMETER Force
    Delete the existing venv and reinstall from scratch.

.PARAMETER EngineProfile
    Which audio.cpp CUDA build to fetch: balance (default), fast or portable.
    These differ only in the CPU instruction sets their kernels use -- the GPU
    requirement is identical. 'fast' assumes a recent x86-64 CPU; 'portable'
    avoids AVX entirely and is the fallback if 'balance' crashes on startup.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [ValidateSet("balance", "fast", "portable")]
    [string]$EngineProfile = "balance"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

. (Join-Path $Here "config.ps1")

$VenvDir = Join-Path $Here "venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$EngineDir = Join-Path $Here "engine"
$EngineExe = Join-Path $EngineDir "audiocpp_server.exe"
$ModelFile = Join-Path $Here $HIGGS_MODEL_FILE
$DlDir = Join-Path $Here "dl"

# --- Upstream payload -------------------------------------------------------
# The engine and the weights are other people's redistributables and far too
# large for git, so they are fetched here instead of being committed. Both are
# pinned: audio.cpp is not API-stable across releases, and the wrapper is
# written against this engine's higgs_audio_tts behaviour.

$AudioCppTag = "release-0.4.2"
$AudioCppBuild = "27d87ba"
$AudioCppBase = "https://github.com/0xShug0/audio.cpp/releases/download/$AudioCppTag"
$ModelRepo = "audio-cpp/audio.cpp-gguf"
$ModelRepoFile = "Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf"

function Write-Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }

# Run a command and capture the result without tripping $ErrorActionPreference.
# Windows PowerShell turns *any* native stderr output into a terminating
# NativeCommandError when EAP is Stop -- which would abort setup on probes whose
# whole purpose is to fail (e.g. "is gradio installed yet?").
function Invoke-Probe {
    param([string]$Exe, [string[]]$ProbeArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe @ProbeArgs 2>&1
        return [pscustomobject]@{
            Ok     = ($LASTEXITCODE -eq 0)
            Output = ($out | Out-String).Trim()
        }
    } finally { $ErrorActionPreference = $prev }
}

# Fetch to a .part file and rename only on success, so an interrupted download
# is never mistaken for a complete one by the next run. curl.exe ships with
# Windows 10 1803+ and both resumes (-C -) and prints a usable progress bar;
# Invoke-WebRequest buffers the whole response in memory, which is untenable at
# 4.7 GB, so it is only the last resort.
function Get-RemoteFile {
    param([string]$Uri, [string]$OutFile, [string]$Label)

    if (Test-Path $OutFile) { Write-Ok "$Label already present"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
    $part = "$OutFile.part"
    Write-Host "  downloading $Label" -ForegroundColor DarkGray

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $curl.Source -L --fail --retry 3 --retry-delay 5 -C - -o $part $Uri
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $prev }
        if ($code -ne 0) { throw "download failed ($Label). Re-run to resume from where it stopped." }
    } else {
        $prev = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try { Invoke-WebRequest -Uri $Uri -OutFile $part -UseBasicParsing }
        finally { $ProgressPreference = $prev }
    }

    Move-Item -Force $part $OutFile
    Write-Ok "$Label downloaded"
}

# Model downloads require the Hugging Face CLI plus hf-xet for high-speed
# transfers. Install both into Python 3.12's per-user site when the command is
# missing, then expose that Python Scripts directory to this setup process.
function Ensure-HuggingFaceCli {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        Write-Ok "Hugging Face CLI available ($($hf.Source))"
        return
    }

    Write-Warn "Hugging Face CLI ('hf') is not installed or not on PATH."
    $answer = Read-Host "  Install hf and hf-xet now? [Y/n]"
    if ($answer -and $answer -notmatch '^[Yy]') {
        throw "Hugging Face CLI is required to download the model. Re-run setup and approve its installation."
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyLauncher) {
        throw "Python launcher ('py') not found. Install Python 3.12, then re-run setup."
    }

    $pythonProbe = Invoke-Probe $pyLauncher.Source @("-3.12", "-c", "import sys; print(sys.executable)")
    if (-not $pythonProbe.Ok) {
        throw "Python 3.12 not found. Install it from python.org, then re-run setup."
    }

    Write-Host "  installing Hugging Face CLI and fast-transfer support" -ForegroundColor DarkGray
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $pyLauncher.Source -3.12 -m pip install --user --upgrade huggingface_hub hf_xet
        $installCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousEap
    }
    if ($installCode -ne 0) { throw "Failed to install the Hugging Face CLI and hf-xet." }

    $scriptsProbe = Invoke-Probe $pyLauncher.Source @(
        "-3.12", "-c", "import sysconfig; print(sysconfig.get_path('scripts')); print(sysconfig.get_path('scripts', scheme='nt_user'))"
    )
    if (-not $scriptsProbe.Ok) { throw "Could not locate Python's Scripts directories after installing hf." }
    $scriptDirs = @($scriptsProbe.Output -split "`r?`n" | Where-Object { $_ } | Select-Object -Unique)
    foreach ($scriptDir in $scriptDirs) {
        if (($env:Path -split ';') -notcontains $scriptDir) {
            $env:Path = "$scriptDir;$env:Path"
        }
    }

    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $hf) {
        throw "hf was installed, but its command could not be found in Python's Scripts directories. Open a new terminal and re-run setup."
    }
    Write-Ok "Hugging Face CLI installed ($($hf.Source))"
}

# Download a single Hub file with the Hugging Face CLI. Modern versions use
# hf-xet automatically; high-performance mode increases concurrency and uses
# as much available network and disk bandwidth as possible. --local-dir keeps
# partial-download metadata so an interrupted setup can resume efficiently.
function Get-HuggingFaceFile {
    param([string]$Repo, [string]$RepoFile, [string]$OutFile, [string]$Label)

    if (Test-Path $OutFile) { Write-Ok "$Label already present"; return }

    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $hf) {
        throw "Hugging Face CLI ('hf') is unavailable even though the prerequisite check passed. Re-run setup."
    }

    $outDir = Split-Path -Parent $OutFile
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "  downloading $Label with hf (Xet high-performance mode)" -ForegroundColor DarkGray

    $previousXetMode = $env:HF_XET_HIGH_PERFORMANCE
    $previousEap = $ErrorActionPreference
    $env:HF_XET_HIGH_PERFORMANCE = "1"
    $ErrorActionPreference = "Continue"
    try {
        & $hf.Source download $Repo $RepoFile --local-dir $outDir
        if ($LASTEXITCODE -ne 0) { throw "hf download failed ($Label). Re-run to resume." }
    } finally {
        $ErrorActionPreference = $previousEap
        $env:HF_XET_HIGH_PERFORMANCE = $previousXetMode
    }

    $downloadedFile = Join-Path $outDir ($RepoFile -replace '/', '\')
    if (-not (Test-Path $downloadedFile)) {
        throw "hf reported success, but the downloaded model was not found at $downloadedFile."
    }
    Move-Item -Force $downloadedFile $OutFile
    Write-Ok "$Label downloaded"
}

# Unpack a release zip over engine\. The zips have varied between a flat layout
# and a single wrapper folder, so descend through any lone top-level directory
# rather than assuming either shape.
function Expand-IntoEngine {
    param([string]$Zip)

    $stage = Join-Path $DlDir ("stage-" + [IO.Path]::GetFileNameWithoutExtension($Zip))
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    Expand-Archive -Path $Zip -DestinationPath $stage -Force

    $root = $stage
    while ($true) {
        $items = @(Get-ChildItem -Force $root)
        if ($items.Count -eq 1 -and $items[0].PSIsContainer) { $root = $items[0].FullName } else { break }
    }

    New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
    Copy-Item -Path (Join-Path $root '*') -Destination $EngineDir -Recurse -Force
    Remove-Item -Recurse -Force $stage
}

Write-Host "Higgs Audio v3 TTS setup" -ForegroundColor White
Write-Host "Installing into: $Here"

Write-Step "Checking Hugging Face CLI"
Ensure-HuggingFaceCli

# --- 1. GPU check -----------------------------------------------------------
# The engine is built CUDA-only in this package. It can fall back to
# --backend cpu, but a 4B autoregressive model on CPU is far too slow to play
# with, so treat a missing GPU as a failure rather than a quiet downgrade.

Write-Step "Checking NVIDIA GPU"
try {
    $gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Write-Ok $gpu
} catch {
    throw "nvidia-smi not found. Higgs3 requires an NVIDIA GPU with recent drivers."
}

# --- 2. Engine binaries -----------------------------------------------------

Write-Step "Checking audio.cpp engine"

# Two zips: one shared CUDA runtime, plus one profile package holding the
# binaries. Roughly 820 MB combined, downloaded once.
if (-not (Test-Path $EngineExe)) {
    $zip = Join-Path $DlDir "audiocpp-windows-cuda-$EngineProfile-$AudioCppBuild.zip"
    Get-RemoteFile "$AudioCppBase/audiocpp-windows-cuda-$EngineProfile-$AudioCppBuild.zip" $zip "audio.cpp $AudioCppTag ($EngineProfile)"
    Expand-IntoEngine $zip
}
if (-not (Test-Path $EngineExe)) {
    throw "audiocpp_server.exe still missing after extraction. Delete dl\ and re-run."
}

# model_specs\ is NOT in the binary zips -- it ships in the audio.cpp *source*
# tree, and the engine reads it from its working directory at runtime (see
# Start_Higgs3.ps1). It is vendored into this repo instead of downloaded: ~140 KB
# of JSON, so carrying it removes a network dependency and a way for setup to
# half-succeed. See model_specs\NOTICE.md.
$SpecsSrc = Join-Path $Here "model_specs"
$SpecsDir = Join-Path $EngineDir "model_specs"
if (-not (Test-Path (Join-Path $SpecsSrc "higgs_audio_tts.json"))) {
    throw "model_specs\higgs_audio_tts.json is missing from this checkout. Re-clone, or copy model_specs\ from audio.cpp $AudioCppTag."
}
New-Item -ItemType Directory -Force -Path $SpecsDir | Out-Null
Copy-Item -Path (Join-Path $SpecsSrc "*.json") -Destination $SpecsDir -Force
Write-Ok "model_specs ($((Get-ChildItem $SpecsDir -Filter *.json).Count) specs)"

$missingDlls = @('cublas64_13.dll', 'cublasLt64_13.dll', 'cufft64_12.dll', 'MSVCP140.dll') |
    Where-Object { -not (Test-Path (Join-Path $EngineDir $_)) }
if ($missingDlls) {
    $zip = Join-Path $DlDir "audiocpp-windows-cuda-runtime.zip"
    Get-RemoteFile "$AudioCppBase/audiocpp-windows-cuda-runtime.zip" $zip "CUDA runtime libraries"
    Expand-IntoEngine $zip
    foreach ($dll in $missingDlls) {
        if (-not (Test-Path (Join-Path $EngineDir $dll))) {
            throw "engine\$dll still missing after extracting the CUDA runtime. Delete dl\ and re-run."
        }
    }
}
# --help exercises DLL resolution: a missing CUDA runtime fails here with a
# loader error rather than 60 seconds into a model load.
$probe = Invoke-Probe $EngineExe @("--help")
if (-not ($probe.Output -match "audiocpp_server")) {
    Write-Host $probe.Output -ForegroundColor DarkGray
    throw "audiocpp_server.exe could not start. A CUDA runtime DLL is probably missing."
}
Write-Ok "audiocpp_server.exe runs and resolves its DLLs"

# --- 3. Model ---------------------------------------------------------------
# The GGUF is self-contained: it embeds the model spec, tokenizer and config,
# which is why there is no models\higgs\ directory of loose JSON here.

Write-Step "Checking model weights"
if (-not (Test-Path $ModelFile)) {
    Write-Warn "4.7 GB download via hf with fast Xet transfers; it resumes if interrupted."
    Get-HuggingFaceFile $ModelRepo $ModelRepoFile $ModelFile "higgs-audio-v3-tts-4b-q8_0.gguf"
}
$sizeGB = [math]::Round((Get-Item $ModelFile).Length / 1GB, 2)
if ($sizeGB -lt 4.0) {
    throw "Model at $ModelFile is only $sizeGB GB; expected ~4.7 GB. The download was truncated -- delete it and re-run."
}
Write-Ok "higgs-audio-v3-tts-4b-q8_0.gguf ($sizeGB GB)"

# --- 4. ffmpeg --------------------------------------------------------------
# Used to normalise every reference clip to 24 kHz mono PCM. Reference clips in
# circulation include MP3-in-WAV, so this is not optional.

Write-Step "Checking ffmpeg"
$ffmpeg = if ($HIGGS_FFMPEG) { $HIGGS_FFMPEG } else { "ffmpeg" }
$ff = Invoke-Probe $ffmpeg @("-version")
if ($ff.Ok -and $ff.Output -match "ffmpeg version") {
    Write-Ok (($ff.Output -split "`n")[0]).Trim()
} else {
    throw "ffmpeg not found (tried '$ffmpeg'). Install it and put it on PATH, or set `$HIGGS_FFMPEG in config.ps1."
}

# --- 5. Python 3.12 ---------------------------------------------------------
# Matching the container's interpreter. Nothing here is version-sensitive
# beyond Gradio's own support window, but 3.12 is what the image shipped.

Write-Step "Locating Python 3.12"
$SysPy = $null
try {
    $out = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { $SysPy = $out.Trim() }
} catch { }
if (-not $SysPy) {
    throw "Python 3.12 not found. Install it from python.org, then re-run this script."
}
Write-Ok $SysPy

# --- 6. Virtual environment -------------------------------------------------

if ($Force -and (Test-Path $VenvDir)) {
    Write-Step "Removing existing venv (-Force)"
    Remove-Item -Recurse -Force $VenvDir
}

if (Test-Path $VenvPy) {
    Write-Step "Virtual environment already exists"
    Write-Ok $VenvDir
} else {
    Write-Step "Creating virtual environment"
    & $SysPy -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    Write-Ok $VenvDir
}

Write-Step "Upgrading pip"
& $VenvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
Write-Ok "pip current"

# --- 7. Wrapper dependencies ------------------------------------------------
# Versions pinned exactly as the image had them. Gradio in particular is pinned
# because the mod's client talks to /gradio_api/call + SSE, which is sensitive
# to the Gradio major version.

Write-Step "Installing wrapper dependencies"
$gradioProbe = Invoke-Probe $VenvPy @("-c", "import gradio; print(gradio.__version__)")
if ($gradioProbe.Ok -and $gradioProbe.Output -eq "5.38.2") {
    Write-Ok "dependencies already installed (gradio $($gradioProbe.Output))"
} else {
    & $VenvPy -m pip install `
        "numpy<2.3" `
        soundfile loguru requests `
        "gradio==5.38.2" "gradio-client==1.11.0" `
        "fastapi==0.115.14" "starlette==0.45.3" `
        "pydantic>=2.0,<3"
    if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
    Write-Ok "dependencies installed"
}

# --- 8. Import check --------------------------------------------------------
# pip succeeding does not mean the wrapper can import: soundfile in particular
# carries a native libsndfile that can fail to load independently of the wheel.

Write-Step "Verifying imports"
$imports = Invoke-Probe $VenvPy @(
    "-c",
    "import gradio, numpy, soundfile, requests, loguru, fastapi; print('IMPORTS_OK')"
)
if (-not ($imports.Ok -and $imports.Output -match "IMPORTS_OK")) {
    Write-Host $imports.Output -ForegroundColor DarkGray
    throw "the wrapper's dependencies do not import. Re-run with -Force."
}
Write-Ok "all imports resolve"

# --- 9. Warmup reference ----------------------------------------------------

Write-Step "Checking warmup reference"
$warmup = Join-Path $Here "warmup_ref.wav"
if (Test-Path $warmup) {
    Write-Ok "warmup_ref.wav present"
} else {
    Write-Warn "warmup_ref.wav is missing; the first request of a session will be slow."
}

New-Item -ItemType Directory -Force -Path (Join-Path $Here "refcache"), (Join-Path $Here "logs") | Out-Null

if (Test-Path $DlDir) {
    Write-Host "`n  dl\ holds the downloaded archives (~820 MB) purely so a re-run" -ForegroundColor DarkGray
    Write-Host "  does not fetch them again. Safe to delete." -ForegroundColor DarkGray
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Start the server with:  .\Start_Higgs3.ps1" -ForegroundColor White
Write-Host "Then point SkyrimNet's TTS at:  http://127.0.0.1:$HIGGS_PORT" -ForegroundColor White
