#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-help}"
MANAGER="$REPOSITORY_ROOT/scripts/manage-local.sh"
SETUP_SCRIPT="$REPOSITORY_ROOT/scripts/setup-local.sh"
VERSION_FILE="$REPOSITORY_ROOT/backend/app/VERSION"

show_help() {
  cat <<'EOF'
VerifyVision

Usage:
  ./verifyvision.sh <command>

Commands:
  all        Set up and start VerifyVision
  setup      Install pinned dependencies
  start      Start the frontend and backend
  stop       Stop services started by this launcher
  status     Show service and CUDA status
  diagnose   Check the environment and CUDA
  update     Check for dependency updates
  version    Show the VerifyVision version
  help       Show this help
EOF
}

dispatch_managed() {
  local command="$1"
  if [[ "${VERIFYVISION_LAUNCHER_TEST_MODE:-}" == "1" ]]; then
    echo "dispatch:$command:scripts/manage-local.sh"
    return
  fi
  "$MANAGER" "$command"
}

case "${COMMAND,,}" in
  ""|help) show_help ;;
  version) echo "VerifyVision v$(tr -d '\r\n' <"$VERSION_FILE")" ;;
  setup)
    if [[ "${VERIFYVISION_LAUNCHER_TEST_MODE:-}" == "1" ]]; then
      echo "dispatch:setup:scripts/setup-local.sh"
    else
      "$SETUP_SCRIPT"
    fi
    ;;
  start|stop|status|diagnose) dispatch_managed "${COMMAND,,}" ;;
  update)
    if [[ "${VERIFYVISION_LAUNCHER_TEST_MODE:-}" == "1" ]]; then
      echo "dispatch:update:scripts/check_dependency_updates.py"
    elif [[ ! -x "$REPOSITORY_ROOT/.venv/bin/python" ]]; then
      echo "VerifyVision is not set up yet. Run ./verifyvision.sh setup." >&2
      exit 1
    else
      "$REPOSITORY_ROOT/.venv/bin/python" "$REPOSITORY_ROOT/scripts/check_dependency_updates.py"
    fi
    ;;
  all)
    if [[ "${VERIFYVISION_LAUNCHER_TEST_MODE:-}" == "1" ]]; then
      echo "dispatch:all:setup-if-needed,start"
    else
      if ! "$MANAGER" ready; then
        echo "Preparing VerifyVision for first use..."
        "$SETUP_SCRIPT"
      fi
      "$MANAGER" start
    fi
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    show_help >&2
    exit 2
    ;;
esac
