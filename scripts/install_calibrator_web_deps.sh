#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
BACKEND_REQUIREMENTS_FILE="${BACKEND_REQUIREMENTS_FILE:-$ROOT_DIR/backend/requirements-web.txt}"
FRONTEND_DIR="${FRONTEND_DIR:-$ROOT_DIR/frontend/calibrator-ui}"
NPM_BIN="${NPM_BIN:-npm}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v "$NPM_BIN" >/dev/null 2>&1; then
  echo "Missing npm command: $NPM_BIN" >&2
  exit 1
fi

if [[ ! -f "$BACKEND_REQUIREMENTS_FILE" ]]; then
  echo "Missing backend requirements file: $BACKEND_REQUIREMENTS_FILE" >&2
  exit 1
fi

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "Missing frontend package.json in: $FRONTEND_DIR" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[install-calibrator] Creating Python virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ ! -x "$VENV_PYTHON" || ! -x "$VENV_PIP" ]]; then
  echo "Python virtual environment is not valid: $VENV_DIR" >&2
  exit 1
fi

echo "[install-calibrator] Installing backend Python dependencies"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PIP" install -r "$BACKEND_REQUIREMENTS_FILE"

echo "[install-calibrator] Installing frontend npm dependencies"
pushd "$FRONTEND_DIR" >/dev/null
if [[ -f package-lock.json ]]; then
  "$NPM_BIN" ci
else
  "$NPM_BIN" install
fi
popd >/dev/null

echo "[install-calibrator] Dependency installation complete"
echo "[install-calibrator] Start the app with: ./scripts/run_calibrator_web.sh"
