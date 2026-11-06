#!/usr/bin/env bash
# One-command start for Git Bash / Linux / macOS:   bash start.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo ">> Creating virtual environment (.venv)"
  python -m venv .venv
fi
# Windows (Git Bash) puts activate in Scripts/, Linux/macOS in bin/
if [ -f ".venv/Scripts/activate" ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi

if ! python -c "import fastapi, sentence_transformers" 2>/dev/null; then
  echo ">> Installing dependencies (first time only, takes a few minutes)"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-faiss.txt || echo ">> FAISS not available for this Python; using NumPy index instead (fine)."
fi

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ">> Created .env - open it, paste your GEMINI_API_KEY, then run this script again."
  exit 0
fi

echo ">> Starting server on http://localhost:8000  (Ctrl+C to stop)"
python -m uvicorn semcache.app:app --host 127.0.0.1 --port 8000
