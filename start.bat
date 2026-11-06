@echo off
REM One-command start for Windows (cmd / PowerShell):  start.bat
cd /d "%~dp0"

if not exist ".venv" (
  echo ^>^> Creating virtual environment .venv
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -c "import fastapi, sentence_transformers" 2>nul
if errorlevel 1 (
  echo ^>^> Installing dependencies - first time only, takes a few minutes
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-faiss.txt
  if errorlevel 1 echo ^>^> FAISS not available for this Python; using NumPy index instead - fine.
)

if not exist ".env" (
  copy .env.example .env >nul
  echo ^>^> Created .env - open it, paste your GEMINI_API_KEY, then run start.bat again.
  exit /b 0
)

echo ^>^> Starting server on http://localhost:8000  - Ctrl+C to stop
python -m uvicorn semcache.app:app --host 127.0.0.1 --port 8000
