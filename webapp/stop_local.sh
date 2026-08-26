#!/bin/zsh
set -euo pipefail

RUNTIME_DIR="${0:A:h}/.runtime"
for service in frontend backend; do
  pid_file="$RUNTIME_DIR/$service.pid"
  if [[ ! -f "$pid_file" ]]; then
    continue
  fi
  pid="$(<"$pid_file")"
  if [[ "$pid" == <-> ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
  fi
  rm -f "$pid_file"
done

echo "Local website services stopped."
