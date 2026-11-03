@echo off
rem Image cleanup setup: installs Nunchaku for this venv and downloads FLUX.1 [dev] (about 17GB).
rem Run once on the server after run.bat has created .venv. FLUX.1 [dev] is gated: accept its license
rem on Hugging Face and log in first (the script explains how). Keep this file ASCII.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run run.bat once first so .venv exists.
  pause
  exit /b 1
)
.venv\Scripts\python -m server.cleanup_setup
if errorlevel 1 (
  pause
  exit /b 1
)
pause
