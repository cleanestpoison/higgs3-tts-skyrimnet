<#
.SYNOPSIS
    Launches the Higgs Audio v3 TTS server for SkyrimNet.

.DESCRIPTION
    Native Windows replacement for the container's docker-entrypoint.sh. Starts
    two processes: the audio.cpp engine (loopback only) and the Zonos-compatible
    wrapper SkyrimNet talks to. The engine is stopped when the wrapper exits.

    Differences from the container, all deliberate:
      * No whisper.cpp process -- SimpleParakeet already serves SkyrimNet's STT
        on 8210, and Higgs v3 clones transcript-free, so nothing here needs it.
      * No vast.ai idle watchdog -- that self-stopped a rented cloud instance
        after 30 minutes of quiet, which on your own machine is just a server
        that dies mid-conversation.
      * No bash supervisor -- a crash is printed rather than silently restarted.

    The wrapper does not bind its port until the engine is loaded and a full
    warmup generation has completed, so SkyrimNet can never win a race against
    a still-loading server.

.PARAMETER Port
    Override the wrapper port from config.ps1.

.PARAMETER ReferenceText
    Transcribe reference clips via SimpleParakeet and use them as ICL examples,
    instead of the default transcript-free cloning.
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [switch]$ReferenceText
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

. (Join-Path $Here "config.ps1")

if ($Port -gt 0) { $HIGGS_PORT = $Port }
if ($ReferenceText) { $HIGGS_USE_REFERENCE_TEXT = 1 }

$VenvPy = Join-Path $Here "venv\Scripts\python.exe"
$Wrapper = Join-Path $Here "higgs3_zonos_wrapper.py"
$EngineDir = Join-Path $Here "engine"
$EngineExe = Join-Path $EngineDir "audiocpp_server.exe"
$ServerJson = Join-Path $EngineDir "server.json"
$ModelFile = Join-Path $Here $HIGGS_MODEL_FILE
$LogDir = Join-Path $Here "logs"

if (-not (Test-Path $VenvPy)) {
    Write-Host "Not set up yet. Run .\Setup.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ModelFile)) {
    Write-Host "Model not found at $ModelFile" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Here "refcache") | Out-Null

# --- Port conflict checks ---------------------------------------------------
# Both are checked up front. Gradio would otherwise fail deep in startup after
# the model is already loaded, wasting a minute on a knowable error.

function Test-PortFree($portNumber, $label) {
    $inUse = Get-NetTCPConnection -LocalPort $portNumber -State Listen -ErrorAction SilentlyContinue
    if ($inUse) {
        $owner = (Get-Process -Id $inUse[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
        Write-Host "$label port $portNumber is already in use by '$owner' (PID $($inUse[0].OwningProcess))." -ForegroundColor Red
        return $false
    }
    return $true
}

if (-not (Test-PortFree $HIGGS_PORT "Wrapper")) {
    Write-Host "Stop it, or pick another port:  .\Start_Higgs3.ps1 -Port 7863" -ForegroundColor Yellow
    exit 1
}

# The engine port needs one extra case. Closing this window with the X button
# kills PowerShell outright, so the cleanup below never runs and the engine is
# left orphaned holding ~6 GB of VRAM. Telling the user to "pick another port"
# would be actively wrong advice there: the fix is to reclaim our own process.
# Only ever a process running *this folder's* exe is touched.
$engineInUse = Get-NetTCPConnection -LocalPort $HIGGS_ENGINE_PORT -State Listen -ErrorAction SilentlyContinue
if ($engineInUse) {
    $holder = Get-Process -Id $engineInUse[0].OwningProcess -ErrorAction SilentlyContinue
    if ($holder -and $holder.Path -eq $EngineExe) {
        Write-Host "Reclaiming an orphaned engine from a previous run (PID $($holder.Id))..." -ForegroundColor Yellow
        try { Stop-Process -Id $holder.Id -Force -ErrorAction Stop } catch {
            Write-Host "Could not stop it: $_" -ForegroundColor Red
            exit 1
        }
        # The socket lingers briefly after the process dies.
        $waitUntil = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $waitUntil -and
               (Get-NetTCPConnection -LocalPort $HIGGS_ENGINE_PORT -State Listen -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 250
        }
    } else {
        $name = if ($holder) { $holder.ProcessName } else { "unknown" }
        Write-Host "Engine port $HIGGS_ENGINE_PORT is already in use by '$name' (PID $($engineInUse[0].OwningProcess))." -ForegroundColor Red
        Write-Host "Change `$HIGGS_ENGINE_PORT in config.ps1." -ForegroundColor Yellow
        exit 1
    }
}

# --- VRAM advisory ----------------------------------------------------------
# q8_0 of a 4B model plus the codec and KV cache lands around 6 GB. This is a
# warning, not a gate: free VRAM fluctuates and the driver can page, so a low
# reading is not a certain failure.

try {
    $freeMiB = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if ($freeMiB -lt 7000) {
        Write-Host "Warning: only ${freeMiB} MiB VRAM free; the q8 4B model wants ~6000 MiB." -ForegroundColor Yellow
        Write-Host "         Close Skyrim or other GPU applications if loading fails." -ForegroundColor Yellow
    }
} catch { }

# --- Engine configuration ---------------------------------------------------
# Generated on every launch rather than kept as a checked-in file, so config.ps1
# stays the single place anything is edited. This is the same JSON the
# container's entrypoint wrote, with Windows paths.
#
# lazy_load=false: the GGUF maps at boot, and the wrapper's warmup generation
# then warms the graphs before the mod-facing port is ever bound.

$engineConfig = [ordered]@{
    host            = "127.0.0.1"
    port            = $HIGGS_ENGINE_PORT
    backend         = "cuda"
    device          = 0
    threads         = $HIGGS_ENGINE_THREADS
    lazy_load       = $false
    busy_timeout_ms = $HIGGS_BUSY_TIMEOUT_MS
    models          = @(
        [ordered]@{
            id              = "higgs"
            family          = "higgs_audio_tts"
            path            = $ModelFile.Replace('\', '/')
            task            = "tts"
            mode            = "offline"
            session_options = [ordered]@{
                "higgs_audio_tts.reference_cache_slots" = $HIGGS_REF_CACHE_SLOTS
            }
        }
    )
}
$engineConfig | ConvertTo-Json -Depth 6 | Set-Content -Path $ServerJson -Encoding UTF8

# --- Environment ------------------------------------------------------------
# These are the exact variable names the container used, so the wrapper is
# unmodified in how it reads its configuration.

$env:HIGGS_PORT = $HIGGS_PORT
$env:HIGGS_HOST = $HIGGS_HOST
$env:HIGGS_ENGINE_URL = "http://127.0.0.1:$HIGGS_ENGINE_PORT"
$env:HIGGS_MODEL_ID = "higgs"
$env:HIGGS_QUANT = "q8"
$env:HIGGS_REF_CACHE_DIR = Join-Path $Here "refcache"
$env:HIGGS_WARMUP_REF = Join-Path $Here "warmup_ref.wav"
$env:HIGGS_REF_MAX_SECONDS = $HIGGS_REF_MAX_SECONDS
$env:HIGGS_SAMPLES = $HIGGS_SAMPLES
$env:HIGGS_SAMPLES_DIR = Join-Path $Here "samples"
$env:HIGGS_TOKEN_BUDGET = $HIGGS_TOKEN_BUDGET
$env:HIGGS_TRIM_SILENCE = $HIGGS_TRIM_SILENCE
$env:HIGGS_TEMPERATURE = $HIGGS_TEMPERATURE
$env:HIGGS_TOP_K = $HIGGS_TOP_K
$env:HIGGS_FFMPEG = $HIGGS_FFMPEG
$env:HIGGS_USE_REFERENCE_TEXT = $HIGGS_USE_REFERENCE_TEXT
# The transcriber is only ever contacted when reference text is enabled, so the
# wrapper's whole STT half stays off otherwise.
$env:WHISPER_ENABLED = $HIGGS_USE_REFERENCE_TEXT
$env:WHISPER_API_URL = $WHISPER_API_URL
$env:WHISPER_WAIT_SECONDS = $WHISPER_WAIT_SECONDS
$env:PYTHONUNBUFFERED = "1"

# --- Cloning-mode advisory --------------------------------------------------

$mode = "transcript-free (default)"
if ($HIGGS_USE_REFERENCE_TEXT -eq 1) {
    $reachable = $false
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $uri = [Uri]$WHISPER_API_URL
        $reachable = $tcp.ConnectAsync($uri.Host, $uri.Port).Wait(1500)
        $tcp.Close()
    } catch { }
    if ($reachable) {
        $mode = "ICL with reference transcripts"
    } else {
        $mode = "transcript-free (transcriber not running)"
        Write-Host "Note: transcriber at $WHISPER_API_URL is not responding." -ForegroundColor Yellow
        Write-Host "      Falling back to transcript-free cloning. Start SimpleParakeet for ICL." -ForegroundColor Yellow
    }
}

$stamp = "{0:yyyy-MM-dd_HH-mm-ss}" -f (Get-Date)
$engineLog = Join-Path $LogDir "engine-$stamp.log"
$engineErrLog = Join-Path $LogDir "engine-$stamp.err.log"
$wrapperLog = Join-Path $LogDir "higgs3-$stamp.log"

Write-Host ""
Write-Host "  Higgs Audio v3 TTS for SkyrimNet" -ForegroundColor Cyan
Write-Host "  ----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  Model      : higgs-audio-v3-tts-4b q8_0 (audio.cpp, CUDA)"
Write-Host "  Cloning    : $mode"
Write-Host "  Endpoint   : http://127.0.0.1:$HIGGS_PORT" -ForegroundColor Green
Write-Host "  Health     : http://127.0.0.1:$HIGGS_PORT/health"
Write-Host "  ----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  Loading the model and warming the CUDA graphs. The port opens" -ForegroundColor DarkGray
Write-Host "  only once generation is warm, so SkyrimNet cannot connect early." -ForegroundColor DarkGray
Write-Host ""

# --- Engine -----------------------------------------------------------------
# Started first and given its own log: it is chatty at load, and mixing that
# into the wrapper's log makes both harder to read after a failure.
#
# WorkingDirectory is the engine folder so model_specs\ resolves the way it did
# from the container's WORKDIR. (The GGUF embeds its own spec, so this is a
# belt-and-braces measure rather than a requirement.)

Write-Host "Starting audio.cpp engine on 127.0.0.1:$HIGGS_ENGINE_PORT ..." -ForegroundColor DarkGray
$engine = Start-Process -FilePath $EngineExe `
    -ArgumentList @("--config", "`"$ServerJson`"") `
    -WorkingDirectory $EngineDir `
    -RedirectStandardOutput $engineLog `
    -RedirectStandardError $engineErrLog `
    -NoNewWindow -PassThru

function Stop-Engine {
    if ($script:engine -and -not $script:engine.HasExited) {
        Write-Host "Stopping engine (PID $($script:engine.Id))..." -ForegroundColor DarkGray
        try { Stop-Process -Id $script:engine.Id -Force -ErrorAction SilentlyContinue } catch { }
    }
}

try {
    # Wait for the engine to answer before starting the wrapper. The wrapper
    # would wait too (up to HIGGS_WAIT_SECONDS), but it cannot see that the
    # engine process has *died* -- it would just sit there for 15 minutes. This
    # loop notices immediately and reports the engine's own error.
    $deadline = (Get-Date).AddSeconds(300)
    $engineUp = $false
    while ((Get-Date) -lt $deadline) {
        if ($engine.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$HIGGS_ENGINE_PORT/health" `
                -TimeoutSec 3 -ErrorAction Stop
            if ($r.StatusCode -eq 200) { $engineUp = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }

    if (-not $engineUp) {
        Write-Host ""
        if ($engine.HasExited) {
            Write-Host "The engine exited during startup (code $($engine.ExitCode))." -ForegroundColor Red
        } else {
            Write-Host "The engine did not answer within 300s." -ForegroundColor Red
        }
        foreach ($f in @($engineErrLog, $engineLog)) {
            if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
                Write-Host "--- $(Split-Path -Leaf $f) ---" -ForegroundColor DarkGray
                Get-Content $f -Tail 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
            }
        }
        Stop-Engine
        exit 1
    }

    Write-Host "Engine ready. Starting wrapper ..." -ForegroundColor DarkGray
    Write-Host ""

    # loguru writes every line to stderr. Merging it into stdout for the log
    # means PowerShell would, under EAP=Stop, treat the first log line as a
    # fatal NativeCommandError and kill the server at startup -- so drop to
    # Continue for the run itself. Real failures are still caught by exit code.
    $ErrorActionPreference = "Continue"

    # Tee so the console stays live while a full transcript lands on disk.
    #
    # The first ForEach-Object flattens ErrorRecords to plain strings. Without
    # it, 2>&1 hands PowerShell's formatter an ErrorRecord for the first stderr
    # line and it renders a red NativeCommandError banner around an ordinary
    # INFO log line, which reads as a crash when nothing is wrong.
    #
    # Tee-Object writes UTF-16 on Windows PowerShell, which makes the log
    # unreadable to grep and most editors -- so write UTF-8 explicitly.
    & $VenvPy -u $Wrapper 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
    } | ForEach-Object {
        Write-Host $_
        Out-File -FilePath $wrapperLog -InputObject $_ -Append -Encoding utf8
    }

    $code = $LASTEXITCODE
}
finally {
    # Runs on normal exit and on Ctrl+C, so the engine never outlives the
    # wrapper and leaves 6 GB of VRAM held by an orphan.
    Stop-Engine
}

Write-Host "`nHiggs3 exited (code $code)." -ForegroundColor Yellow
Write-Host "Logs: $wrapperLog" -ForegroundColor DarkGray
Write-Host "      $engineLog" -ForegroundColor DarkGray
exit $code
