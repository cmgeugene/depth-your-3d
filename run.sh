#!/usr/bin/env bash
# Depth Studio - Linux launcher. First run creates .venv and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m server
