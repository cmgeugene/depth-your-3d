@echo off
rem Downloads the upscale engines (Real-ESRGAN, RIFE) from their official GitHub releases into .\engines.
rem Run once on the server; already installed engines are skipped. Keep this file ASCII.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-upscale-engines.ps1"
if errorlevel 1 (
  echo Engine setup failed.
  pause
  exit /b 1
)
pause
