#!/bin/zsh
set -euo pipefail

WEBAPP_DIR="${0:A:h}"
PROJECT_DIR="${WEBAPP_DIR:h}"
RUNTIME_DIR="$WEBAPP_DIR/.runtime"
mkdir -p "$RUNTIME_DIR"

# uvicorn runs without --reload, while the Next.js dev server hot-reloads. After a pull, the page
# can therefore call routes the running backend does not have yet (HTTP 405 "Method Not
# Allowed"). backend.pid is written when the backend starts, so any Python source newer than it
# means the backend we started is running stale code: restart it.
pid_file="$RUNTIME_DIR/backend.pid"
if [[ -f "$pid_file" ]]; then
  pid="$(<"$pid_file")"
  if [[ "$pid" == <-> && "$(ps -o command= -p "$pid" 2>/dev/null)" == *uvicorn* ]] \
    && [[ -n "$(find "$WEBAPP_DIR/backend" "$PROJECT_DIR/src/bilingual_voice" -name '*.py' \
      -newer "$pid_file" -print -quit)" ]]; then
    echo "Backend code changed since the backend started; restarting it."
    kill "$pid"
    for attempt in {1..50}; do
      lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 0.2
    done
    if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "The old backend (pid $pid) is still running; stop it, then run this again." >&2
      exit 1
    fi
  fi
fi

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
