@echo off
rem Depth Studio - Windows launcher, this PC only (http://localhost:8000).
rem To share on the internal network use run-lan.bat. First run creates .venv and installs dependencies;
rem later runs install again only when requirements.txt has changed. Keep this file ASCII.
cd /d "%~dp0"
if not defined DS_HOST set DS_HOST=127.0.0.1
if not exist .venv\Scripts\python.exe (
  python -m venv .venv || goto :error
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :error
)
if exist .venv\requirements.installed fc /b requirements.txt .venv\requirements.installed >nul && goto :run
.venv\Scripts\python -m pip install -r requirements.txt || goto :error
copy /y requirements.txt .venv\requirements.installed >nul
:run
.venv\Scripts\python -m server
goto :eof
:error
echo Setup failed.
pause
