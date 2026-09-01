@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found - creating one and installing dependencies...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -q -r requirements-deploy.txt
)

echo Starting churnwatch API on http://127.0.0.1:8000 ...
echo Your browser will open automatically in a few seconds.
echo Close this window to stop the server.
echo.

start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:8000/docs'"

".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
