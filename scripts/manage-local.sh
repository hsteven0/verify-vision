#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
FRONTEND_ROOT="$REPOSITORY_ROOT/frontend"
VITE_SCRIPT="$FRONTEND_ROOT/node_modules/vite/bin/vite.js"
WORKER_PATH="$REPOSITORY_ROOT/.vendor/Eagle/Embodied/locateanything_worker.py"
RUNTIME_ROOT="$REPOSITORY_ROOT/.tmp/verifyvision-runtime"
BACKEND_PID_FILE="$RUNTIME_ROOT/backend.pid"
FRONTEND_PID_FILE="$RUNTIME_ROOT/frontend.pid"
BACKEND_PORT=8000
FRONTEND_PORT=5173

export HF_HOME="${HF_HOME:-$REPOSITORY_ROOT/model-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPOSITORY_ROOT/model-cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPOSITORY_ROOT/.cache}"
export npm_config_cache="${npm_config_cache:-$REPOSITORY_ROOT/.cache/npm}"
export VERIFYVISION_DATA_DIR="${VERIFYVISION_DATA_DIR:-$REPOSITORY_ROOT/backend/data/projects}"
export VERIFYVISION_LOCATEANYTHING_MODEL="${VERIFYVISION_LOCATEANYTHING_MODEL:-nvidia/LocateAnything-3B}"
export VERIFYVISION_LOCATEANYTHING_DEVICE="${VERIFYVISION_LOCATEANYTHING_DEVICE:-auto}"
export VERIFYVISION_LOCATEANYTHING_DTYPE="${VERIFYVISION_LOCATEANYTHING_DTYPE:-auto}"
export VERIFYVISION_LOCATEANYTHING_WORKER_PATH="${VERIFYVISION_LOCATEANYTHING_WORKER_PATH:-$WORKER_PATH}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$npm_config_cache" \
  "$VERIFYVISION_DATA_DIR" "$REPOSITORY_ROOT/.tmp" "$RUNTIME_ROOT"

setup_files_ready() {
  [[ -x "$PYTHON" && -f "$VITE_SCRIPT" && -f "$WORKER_PATH" ]] || return 1
  command -v node >/dev/null || return 1
}

setup_ready() {
  setup_files_ready || return 1
  "$PYTHON" -c 'import fastapi, torch, transformers, ultralytics, uvicorn' >/dev/null 2>&1
}

port_in_use() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

read_pid() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(tr -dc '0-9' <"$file")"
  [[ -n "$pid" ]] || return 1
  printf '%s' "$pid"
}

tracked_process() {
  local file="$1"
  local signature="$2"
  local pid
  pid="$(read_pid "$file")" || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$command_line" == *"$REPOSITORY_ROOT"* && "$command_line" == *"$signature"* ]]
}

cleanup_stale_state() {
  tracked_process "$BACKEND_PID_FILE" "uvicorn" || rm -f "$BACKEND_PID_FILE"
  tracked_process "$FRONTEND_PID_FILE" "vite" || rm -f "$FRONTEND_PID_FILE"
}

stop_tracked() {
  local file="$1"
  local signature="$2"
  if ! tracked_process "$file" "$signature"; then
    rm -f "$file"
    return 1
  fi
  local pid
  pid="$(read_pid "$file")"
  kill "$pid" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
  rm -f "$file"
}

wait_for_url() {
  local pid="$1"
  local url="$2"
  local label="$3"
  for _ in {1..60}; do
    kill -0 "$pid" 2>/dev/null || {
      echo "$label exited during startup. Check $RUNTIME_ROOT/${label,,}.err.log." >&2
      return 1
    }
    if "$PYTHON" -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1)' "$url" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "$label did not become ready at $url." >&2
  return 1
}

start_services() {
  setup_files_ready || {
    echo "VerifyVision is not set up yet. Run ./verifyvision.sh setup or ./verifyvision.sh all." >&2
    exit 1
  }
  cleanup_stale_state
  if tracked_process "$BACKEND_PID_FILE" "uvicorn" && tracked_process "$FRONTEND_PID_FILE" "vite"; then
    echo "VerifyVision is already running."
    echo "Frontend: http://localhost:$FRONTEND_PORT"
    echo "Backend:  http://localhost:$BACKEND_PORT"
    return
  fi
  stop_tracked "$FRONTEND_PID_FILE" "vite" >/dev/null 2>&1 || true
  stop_tracked "$BACKEND_PID_FILE" "uvicorn" >/dev/null 2>&1 || true
  port_in_use "$BACKEND_PORT" && {
    echo "Port $BACKEND_PORT is already in use by a process not managed by VerifyVision." >&2
    exit 1
  }
  port_in_use "$FRONTEND_PORT" && {
    echo "Port $FRONTEND_PORT is already in use by a process not managed by VerifyVision." >&2
    exit 1
  }

  nohup "$PYTHON" -m uvicorn app.main:app --app-dir "$REPOSITORY_ROOT/backend" \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"$RUNTIME_ROOT/backend.out.log" 2>"$RUNTIME_ROOT/backend.err.log" &
  local backend_pid=$!
  printf '%s\n' "$backend_pid" >"$BACKEND_PID_FILE"

  nohup node "$VITE_SCRIPT" --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort \
    >"$RUNTIME_ROOT/frontend.out.log" 2>"$RUNTIME_ROOT/frontend.err.log" &
  local frontend_pid=$!
  printf '%s\n' "$frontend_pid" >"$FRONTEND_PID_FILE"

  if ! wait_for_url "$backend_pid" "http://127.0.0.1:$BACKEND_PORT/api/health" "Backend"; then
    stop_tracked "$FRONTEND_PID_FILE" "vite" >/dev/null 2>&1 || true
    stop_tracked "$BACKEND_PID_FILE" "uvicorn" >/dev/null 2>&1 || true
    exit 1
  fi

  local runtime
  if ! runtime="$("$PYTHON" - "$BACKEND_PORT" <<'PY'
import json
import sys
from urllib.request import urlopen

with urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/api/health", timeout=1) as response:
    health = json.load(response)
if health.get("inference_provider") != "locateanything" or not health.get("inference_available"):
    raise SystemExit(
        "LocateAnything CUDA runtime unavailable: "
        + str(health.get("inference_detail") or "unknown error")
    )
print(f'{health.get("inference_gpu")} · {health.get("inference_selected_dtype")}')
PY
)"; then
    stop_tracked "$FRONTEND_PID_FILE" "vite" >/dev/null 2>&1 || true
    stop_tracked "$BACKEND_PID_FILE" "uvicorn" >/dev/null 2>&1 || true
    exit 1
  fi

  if ! wait_for_url "$frontend_pid" "http://127.0.0.1:$FRONTEND_PORT" "Frontend"; then
    stop_tracked "$FRONTEND_PID_FILE" "vite" >/dev/null 2>&1 || true
    stop_tracked "$BACKEND_PID_FILE" "uvicorn" >/dev/null 2>&1 || true
    exit 1
  fi

  echo
  echo "VerifyVision"
  echo
  echo "Frontend: http://localhost:$FRONTEND_PORT"
  echo "Backend:  http://localhost:$BACKEND_PORT"
  echo
  echo "LocateAnything-3B"
  echo "CUDA: $runtime"
}

stop_services() {
  cleanup_stale_state
  local stopped=0
  if stop_tracked "$FRONTEND_PID_FILE" "vite"; then stopped=1; fi
  if stop_tracked "$BACKEND_PID_FILE" "uvicorn"; then stopped=1; fi
  if [[ "$stopped" -eq 1 ]]; then echo "VerifyVision stopped."; else echo "VerifyVision is already stopped."; fi
}

show_status() {
  cleanup_stale_state
  local frontend_status="Stopped" backend_status="Stopped"
  if tracked_process "$FRONTEND_PID_FILE" "vite"; then frontend_status="Running"
  elif port_in_use "$FRONTEND_PORT"; then frontend_status="Occupied"; fi
  if tracked_process "$BACKEND_PID_FILE" "uvicorn"; then backend_status="Running"
  elif port_in_use "$BACKEND_PORT"; then backend_status="Occupied"; fi
  local cuda="Unavailable" gpu=""
  if command -v nvidia-smi >/dev/null; then
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    [[ -n "$gpu" ]] && cuda="Available"
  fi
  echo "VerifyVision Status"
  echo
  printf 'Frontend   %-10s http://localhost:%s\n' "$frontend_status" "$FRONTEND_PORT"
  printf 'Backend    %-10s http://localhost:%s\n' "$backend_status" "$BACKEND_PORT"
  echo "CUDA       $cuda"
  [[ -n "$gpu" ]] && echo "GPU        $gpu"
  echo "Model      LocateAnything-3B"
}

case "$COMMAND" in
  ready) setup_ready ;;
  start) start_services ;;
  stop) stop_services ;;
  status) show_status ;;
  diagnose)
    setup_ready || { echo "VerifyVision is not set up yet. Run ./verifyvision.sh setup." >&2; exit 1; }
    "$PYTHON" "$REPOSITORY_ROOT/scripts/diagnose_locateanything.py"
    echo
    show_status
    ;;
  *) echo "Unknown internal command: $COMMAND" >&2; exit 2 ;;
esac
