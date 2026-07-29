@echo off
setlocal
title Higgs Audio v3 TTS for SkyrimNet
rem cd to this file's folder so double-clicking works from anywhere.
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\python.exe" (
  echo Higgs3 is not set up yet.
  echo Run Setup.ps1 first ^(right-click, Run with PowerShell^).
  echo.
  pause
  exit /b 1
)

rem Prefer PowerShell 7, fall back to Windows PowerShell. Args pass straight
rem through, e.g.:  Start.bat -Port 7863
where pwsh >nul 2>&1
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Higgs3.ps1" %*
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Higgs3.ps1" %*
)

rem Keep the window open if it exited with an error, so the reason is readable.
if errorlevel 1 pause
