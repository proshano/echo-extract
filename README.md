# data-extractor calibrator web app

This repository includes a local web app for calibrating clinical data extraction prompts before running large-scale extraction jobs. The app combines a FastAPI backend with a React frontend. You load report data, define extraction features, test prompts, and iterate quickly in one interface.

## Quick start

Run these commands from a fresh clone:

```bash
git clone <your-repo-url>
cd data-extractor
./scripts/install_calibrator_web_deps.sh
./scripts/run_calibrator_web.sh
```

After startup, open `http://127.0.0.1:5173` in your browser. The backend API runs at `http://127.0.0.1:8000`.

## What gets installed

The install script sets up everything required to start the web app:

1. Creates a Python virtual environment at `.venv` if it does not exist.
2. Installs backend packages from `backend/requirements-web.txt`.
3. Installs frontend npm packages in `frontend/calibrator-ui`.

## Requirements

Install these tools on your machine before running setup:

- Python 3.9 or newer
- Node.js 18 or newer
- npm
- `llama.cpp` with `llama-server` available in your `PATH`
- At least one `.gguf` model

This repository does not bundle `llama.cpp` binaries.

## Optional: auto-install missing host tools

When run interactively, the install script checks for missing system tools and prompts to install them automatically.

For non-interactive runs (CI, scripts), you can force host-level install:

```bash
AUTO_INSTALL_SYSTEM_DEPS=1 ./scripts/install_calibrator_web_deps.sh
```

Supported package managers for auto-install in this script:

- `brew`
- `apt-get` (does not auto-install `llama.cpp`)

## Running services manually

Use manual startup when you want separate terminals for backend and frontend.

Backend:

```bash
source .venv/bin/activate
python -m uvicorn backend.scripts.calibrator_api.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend/calibrator-ui
VITE_API_BASE=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1 --port 5173
```

## Helpful environment variables

`scripts/run_calibrator_web.sh` supports:

- `BACKEND_HOST` and `BACKEND_PORT`
- `FRONTEND_HOST` and `FRONTEND_PORT`
- `ENABLE_BACKEND_RELOAD=1` for backend hot reload
- `SKIP_DEPENDENCY_INSTALL=1` to skip dependency checks during startup

`scripts/install_calibrator_web_deps.sh` supports:

- `PYTHON_BIN` (default: `python3`)
- `NODE_BIN` (default: `node`)
- `VENV_DIR` (default: `.venv` in repo root)
- `BACKEND_REQUIREMENTS_FILE`
- `FRONTEND_DIR`
- `NPM_BIN` (default: `npm`)
- `AUTO_INSTALL_SYSTEM_DEPS=1` to auto-install missing Python/Node/npm and `llama.cpp` when supported
- `PROMPT_INSTALL_SYSTEM_DEPS=0` to disable the interactive install prompt
- `SYSTEM_PACKAGE_MANAGER` (`auto`, `brew`, or `apt`)

## Core paths

- Backend API: `backend/scripts/calibrator_api/app.py`
- Frontend app: `frontend/calibrator-ui/src/App.tsx`
- One-command launcher: `scripts/run_calibrator_web.sh`
- Dependency installer: `scripts/install_calibrator_web_deps.sh`

## Extraction API (Large CSV)

Use the v2 off-disk extraction path for production and large CSV files:

- `POST /api/extract/v2/run`
- `GET /api/extract/v2/jobs/{job_id}`
- `POST /api/extract/v2/jobs/{job_id}/cancel`
- `GET /api/extract/v2/jobs/{job_id}/download`

Key v2 behavior:

- Input is disk-path only (`input_csv_path`), no upload payload.
- Output is always scoped to managed export root (`~/.data_prompt_calibrator/exports`, or `CALIBRATOR_EXPORT_ROOT`).
- Resume is enabled by default with checkpoint sidecar files.
- Progress starts immediately without pre-counting full CSV rows.

Legacy endpoint `POST /api/extract/run` is still available for compatibility, but new production runs should use v2.
