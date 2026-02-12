#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
BACKEND_REQUIREMENTS_FILE="${BACKEND_REQUIREMENTS_FILE:-$ROOT_DIR/backend/requirements-web.txt}"
FRONTEND_DIR="${FRONTEND_DIR:-$ROOT_DIR/frontend/calibrator-ui}"
NPM_BIN="${NPM_BIN:-npm}"
NODE_BIN="${NODE_BIN:-node}"
AUTO_INSTALL_SYSTEM_DEPS="${AUTO_INSTALL_SYSTEM_DEPS:-0}"
PROMPT_INSTALL_SYSTEM_DEPS="${PROMPT_INSTALL_SYSTEM_DEPS:-1}"
SYSTEM_PACKAGE_MANAGER="${SYSTEM_PACKAGE_MANAGER:-auto}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9
MIN_NODE_MAJOR=18
MIN_NODE_MINOR=0

RESOLVED_PYTHON_BIN=""
RESOLVED_NODE_BIN=""
RESOLVED_NPM_BIN=""
MISSING_PYTHON=0
MISSING_NODE=0
MISSING_NPM=0
MISSING_LLAMA=0

is_truthy() {
  case "${1:-0}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

version_at_least() {
  local raw_version="${1#v}"
  local required_major="$2"
  local required_minor="$3"
  if [[ "$raw_version" =~ ^([0-9]+)(\.([0-9]+))? ]]; then
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[3]:-0}"
    if (( major > required_major )); then
      return 0
    fi
    if (( major == required_major && minor >= required_minor )); then
      return 0
    fi
  fi
  return 1
}

python_version_short() {
  local python_cmd="$1"
  "$python_cmd" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || true
}

node_version_full() {
  local node_cmd="$1"
  "$node_cmd" -p "process.versions.node" 2>/dev/null || true
}

resolve_python_bin() {
  local candidate
  for candidate in "$PYTHON_BIN" python3 python; do
    [[ -z "$candidate" ]] && continue

    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi

    local version_short
    version_short="$(python_version_short "$candidate")"
    if [[ -z "$version_short" ]]; then
      continue
    fi

    if version_at_least "$version_short" "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR"; then
      RESOLVED_PYTHON_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

resolve_node_bin() {
  local candidate
  for candidate in "$NODE_BIN" node; do
    [[ -z "$candidate" ]] && continue

    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi

    local version_full
    version_full="$(node_version_full "$candidate")"
    if [[ -z "$version_full" ]]; then
      continue
    fi

    if version_at_least "$version_full" "$MIN_NODE_MAJOR" "$MIN_NODE_MINOR"; then
      RESOLVED_NODE_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

resolve_npm_bin() {
  local candidate
  for candidate in "$NPM_BIN" npm; do
    [[ -z "$candidate" ]] && continue

    if command -v "$candidate" >/dev/null 2>&1; then
      RESOLVED_NPM_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

detect_package_manager() {
  local requested="$SYSTEM_PACKAGE_MANAGER"
  if [[ "$requested" != "auto" ]]; then
    echo "$requested"
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    echo "brew"
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "apt"
    return 0
  fi

  echo ""
}

is_interactive_terminal() {
  [[ -t 0 && -t 1 ]]
}

prompt_yes_no() {
  local question="$1"
  local default_yes="${2:-1}"
  local prompt_suffix="[y/N]"
  if (( default_yes == 1 )); then
    prompt_suffix="[Y/n]"
  fi

  while true; do
    printf "%s %s " "$question" "$prompt_suffix"
    local reply=""
    if ! read -r reply; then
      return 1
    fi
    reply="${reply// /}"
    if [[ -z "$reply" ]]; then
      if (( default_yes == 1 )); then
        return 0
      fi
      return 1
    fi
    case "$reply" in
      y | Y | yes | YES) return 0 ;;
      n | N | no | NO) return 1 ;;
    esac
    echo "Please answer 'y' or 'n'."
  done
}

install_system_requirements() {
  local need_python="$1"
  local need_node_or_npm="$2"
  local need_llama="$3"
  local manager
  manager="$(detect_package_manager)"

  if [[ -z "$manager" ]]; then
    echo "[install-calibrator] Auto-install requested, but no supported package manager was detected." >&2
    echo "[install-calibrator] Install missing requirements manually, then re-run this script." >&2
    return 1
  fi

  echo "[install-calibrator] Attempting to install missing system dependencies via: $manager"
  case "$manager" in
    brew)
      if (( need_python == 1 )); then
        brew install python
      fi
      if (( need_node_or_npm == 1 )); then
        brew install node
      fi
      if (( need_llama == 1 )); then
        brew install llama.cpp
      fi
      ;;
    apt)
      if (( need_python == 1 || need_node_or_npm == 1 )); then
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip nodejs npm
      fi
      if (( need_llama == 1 )); then
        echo "[install-calibrator] llama.cpp auto-install is not supported for apt in this script." >&2
        echo "[install-calibrator] Install llama.cpp manually and ensure 'llama-server' is in PATH." >&2
        return 1
      fi
      ;;
    *)
      echo "[install-calibrator] Unsupported package manager: $manager" >&2
      echo "[install-calibrator] Supported values are: auto, brew, apt" >&2
      return 1
      ;;
  esac

  return 0
}

refresh_requirement_status() {
  RESOLVED_PYTHON_BIN=""
  RESOLVED_NODE_BIN=""
  RESOLVED_NPM_BIN=""
  MISSING_PYTHON=0
  MISSING_NODE=0
  MISSING_NPM=0
  MISSING_LLAMA=0

  if ! resolve_python_bin; then
    MISSING_PYTHON=1
  fi
  if ! resolve_node_bin; then
    MISSING_NODE=1
  fi
  if ! resolve_npm_bin; then
    MISSING_NPM=1
  fi
  if ! command -v llama-server >/dev/null 2>&1; then
    MISSING_LLAMA=1
  fi
}

describe_missing_requirements() {
  local missing_python="$1"
  local missing_node="$2"
  local missing_npm="$3"
  local missing_llama="$4"

  if (( missing_python == 1 )); then
    echo "- Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} (checked: ${PYTHON_BIN}, python3, python)"
  fi
  if (( missing_node == 1 )); then
    echo "- Node.js >= ${MIN_NODE_MAJOR} (checked: ${NODE_BIN}, node)"
  fi
  if (( missing_npm == 1 )); then
    echo "- npm (checked: ${NPM_BIN}, npm)"
  fi
  if (( missing_llama == 1 )); then
    echo "- llama.cpp 'llama-server' command in PATH"
  fi
}

AUTO_INSTALL_SYSTEM_DEPS_FLAG=0
if is_truthy "$AUTO_INSTALL_SYSTEM_DEPS"; then
  AUTO_INSTALL_SYSTEM_DEPS_FLAG=1
fi
PROMPT_INSTALL_SYSTEM_DEPS_FLAG=0
if is_truthy "$PROMPT_INSTALL_SYSTEM_DEPS"; then
  PROMPT_INSTALL_SYSTEM_DEPS_FLAG=1
fi

refresh_requirement_status
missing_python="$MISSING_PYTHON"
missing_node="$MISSING_NODE"
missing_npm="$MISSING_NPM"
missing_llama="$MISSING_LLAMA"
need_node_or_npm=0
if (( missing_node == 1 || missing_npm == 1 )); then
  need_node_or_npm=1
fi
need_llama_install=0
if (( missing_llama == 1 )); then
  need_llama_install=1
fi

if (( missing_python == 1 || missing_node == 1 || missing_npm == 1 || need_llama_install == 1 )); then
  if (( AUTO_INSTALL_SYSTEM_DEPS_FLAG != 1 && PROMPT_INSTALL_SYSTEM_DEPS_FLAG == 1 )); then
    if is_interactive_terminal; then
      echo "Missing system dependencies detected:"
      describe_missing_requirements "$missing_python" "$missing_node" "$missing_npm" "$missing_llama"
      if prompt_yes_no "Install missing system dependencies now?" 1; then
        AUTO_INSTALL_SYSTEM_DEPS_FLAG=1
      fi
    fi
  fi
fi

if (( missing_python == 1 || missing_node == 1 || missing_npm == 1 || need_llama_install == 1 )); then
  if (( AUTO_INSTALL_SYSTEM_DEPS_FLAG == 1 )); then
    install_system_requirements "$missing_python" "$need_node_or_npm" "$need_llama_install"
    refresh_requirement_status
    missing_python="$MISSING_PYTHON"
    missing_node="$MISSING_NODE"
    missing_npm="$MISSING_NPM"
    missing_llama="$MISSING_LLAMA"
    need_llama_install=0
    if (( missing_llama == 1 )); then
      need_llama_install=1
    fi
  fi
fi

if (( missing_python == 1 || missing_node == 1 || missing_npm == 1 || missing_llama == 1 )); then
  echo "Missing required system dependencies:" >&2
  describe_missing_requirements "$missing_python" "$missing_node" "$missing_npm" "$missing_llama" >&2
  if (( AUTO_INSTALL_SYSTEM_DEPS_FLAG != 1 )); then
    echo "Tip: set AUTO_INSTALL_SYSTEM_DEPS=1 to auto-install supported dependencies." >&2
  fi
  if (( missing_llama == 1 )); then
    echo "llama.cpp is required for this app, and you also need at least one .gguf model." >&2
  fi
  exit 1
fi

if [[ -z "$RESOLVED_PYTHON_BIN" ]]; then
  echo "Missing Python interpreter after preflight checks." >&2
  exit 1
fi
if [[ "$RESOLVED_PYTHON_BIN" != "$PYTHON_BIN" ]]; then
  echo "[install-calibrator] Using Python interpreter: $RESOLVED_PYTHON_BIN"
fi
PYTHON_BIN="$RESOLVED_PYTHON_BIN"

if [[ -z "$RESOLVED_NPM_BIN" ]]; then
  echo "Missing npm command after preflight checks." >&2
  exit 1
fi
if [[ "$RESOLVED_NPM_BIN" != "$NPM_BIN" ]]; then
  echo "[install-calibrator] Using npm command: $RESOLVED_NPM_BIN"
fi
NPM_BIN="$RESOLVED_NPM_BIN"

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
