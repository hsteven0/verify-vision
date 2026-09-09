#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
EAGLE_REVISION="783f656d127ee498137b5ff52603ce36c292d317"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPOSITORY_ROOT/.cache/pip}"
export npm_config_cache="${npm_config_cache:-$REPOSITORY_ROOT/.cache/npm}"
export HF_HOME="${HF_HOME:-$REPOSITORY_ROOT/model-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPOSITORY_ROOT/model-cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPOSITORY_ROOT/.cache}"
export VERIFYVISION_DATA_DIR="${VERIFYVISION_DATA_DIR:-$REPOSITORY_ROOT/backend/data/projects}"
mkdir -p "$PIP_CACHE_DIR" "$npm_config_cache" "$HF_HOME" "$TORCH_HOME" "$VERIFYVISION_DATA_DIR" "$REPOSITORY_ROOT/.tmp"

command -v "$PYTHON_BIN" >/dev/null || { echo "Python 3.12 is required." >&2; exit 1; }
command -v npm >/dev/null || { echo "Node.js 20+ and npm are required." >&2; exit 1; }
command -v git >/dev/null || { echo "Git is required." >&2; exit 1; }

"$PYTHON_BIN" -m venv "$REPOSITORY_ROOT/.venv"
"$REPOSITORY_ROOT/.venv/bin/python" -m pip install torch==2.12.1 torchvision==0.27.1 --index-url "$TORCH_INDEX_URL"
"$REPOSITORY_ROOT/.venv/bin/python" -m pip install -r "$REPOSITORY_ROOT/backend/requirements-local.lock"
"$REPOSITORY_ROOT/.venv/bin/python" -m pip install --no-deps -e "$REPOSITORY_ROOT/backend"
npm --prefix "$REPOSITORY_ROOT/frontend" ci

EAGLE_ROOT="$REPOSITORY_ROOT/.vendor/Eagle"
if [[ ! -d "$EAGLE_ROOT/.git" ]]; then
  mkdir -p "$EAGLE_ROOT"
  git -C "$EAGLE_ROOT" init
  git -C "$EAGLE_ROOT" remote add origin https://github.com/NVlabs/Eagle.git
fi
git -C "$EAGLE_ROOT" fetch --depth 1 origin "$EAGLE_REVISION"
git -C "$EAGLE_ROOT" checkout --detach "$EAGLE_REVISION"
export VERIFYVISION_LOCATEANYTHING_WORKER_PATH="$EAGLE_ROOT/Embodied/locateanything_worker.py"
"$REPOSITORY_ROOT/.venv/bin/python" "$REPOSITORY_ROOT/scripts/diagnose_locateanything.py"
echo "VerifyVision is ready. Start it with ./verifyvision.sh start"
