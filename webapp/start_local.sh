#!/bin/zsh
set -euo pipefail

WEBAPP_DIR="${0:A:h}"
PROJECT_DIR="${WEBAPP_DIR:h}"
RUNTIME_DIR="$WEBAPP_DIR/.runtime"
mkdir -p "$RUNTIME_DIR"

if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  cd "$PROJECT_DIR"
  nohup "$PROJECT_DIR/.venv/bin/python" -m uvicorn webapp.backend.main:app \
    --host 127.0.0.1 --port 8000 >"$RUNTIME_DIR/backend.log" 2>&1 &
  echo $! >"$RUNTIME_DIR/backend.pid"
fi

if ! lsof -nP -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1; then
  cd "$WEBAPP_DIR/frontend"
  nohup npm run dev >"$RUNTIME_DIR/frontend.log" 2>&1 &
  echo $! >"$RUNTIME_DIR/frontend.pid"
fi

echo "Website: http://localhost:3000"
echo "API health: http://localhost:8000/api/health"
if [[ "${COQUI_TOS_AGREED:-}" != "1" && ! -f "$RUNTIME_DIR/xtts_license_accepted" ]]; then
  echo "Open the website and accept the voice-model licence once to enable dubbing."
fi
