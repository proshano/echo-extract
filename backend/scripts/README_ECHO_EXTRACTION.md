# Echo Extraction Pipeline (Standalone)

This folder contains a resume-safe CLI pipeline for extracting structured features from echocardiography free-text reports using a local `llama.cpp` server (`/v1/chat/completions` OpenAI-compatible API).

It also includes a simple desktop calibration UI so non-engineers can tune per-feature prompts before running large-scale extraction.

## 1) Prerequisites

- Python 3.9+
- Installed Python packages:
  - `requests`
  - `pandas` (optional for downstream analysis, not required by extraction runtime)
- A local `llama-server` binary from `llama.cpp`
- A downloaded `.gguf` model

## 2) Download a GGUF model (Hugging Face)

Example models (choose one):
- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
- `bartowski/Llama-3.2-3B-Instruct-GGUF`
- `Qwen/Qwen2.5-7B-Instruct-GGUF`

Example with `huggingface-cli`:

```bash
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ~/models
```

## 3) Start llama-server manually

Example:

```bash
llama-server -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8080 --ctx-size 8192
```

Health check:

```bash
curl http://127.0.0.1:8080/v1/models
```

## 4) Run extraction

Schema example is provided at:
- `backend/scripts/echo_feature_schema.example.json`

Example extraction command:

```bash
python backend/scripts/echo_extract.py \
  --input-csv data/echo_reports.csv \
  --output-csv data/echo_extracted.csv \
  --schema-file backend/scripts/echo_feature_schema.example.json \
  --llamacpp-base-url http://127.0.0.1:8080 \
  --temperature 0.0 \
  --max-retries 5 \
  --model-label llama-local-8b
```

Default column mapping:
- ID column: `StudyID`
- Report text column: `Report`

If needed, you can override with `--id-column` and `--report-column`.
Optional server model targeting: `--llamacpp-model <model_id>`.

## 4b) Simple calibration UI (recommended before full run)

Launch a local desktop UI (no frontend build):

```bash
python backend/scripts/launch_echo_calibrator.py
```

Optional preload:

```bash
python backend/scripts/launch_echo_calibrator.py \
  --input-csv data/echo_reports.csv \
  --schema-file backend/scripts/echo_feature_schema.example.json
```

In the UI, you can:
- Select a CSV and it auto-loads immediately
- Define each feature and its exact prompt text
- Test one feature or all features on a sample row
- Preview the combined extraction prompt
- Save a calibrated schema JSON for production extraction
- Discover local GGUF models and select one
- Refresh server models from `/v1/models` and choose one for calibration calls
- Use setup helpers to install `llama.cpp` (macOS/Homebrew), download a recommended model, and start/stop `llama-server`

## 5) Resume an interrupted run

Resume is enabled by default. If `--output-csv` already exists, previously processed IDs are skipped.

```bash
python backend/scripts/echo_extract.py \
  --input-csv data/echo_reports.csv \
  --output-csv data/echo_extracted.csv \
  --schema-file backend/scripts/echo_feature_schema.example.json \
  --resume
```

To force restart from scratch:

```bash
python backend/scripts/echo_extract.py \
  --input-csv data/echo_reports.csv \
  --output-csv data/echo_extracted.csv \
  --schema-file backend/scripts/echo_feature_schema.example.json \
  --no-resume
```

## 6) Evaluate against a gold standard

```bash
python backend/scripts/evaluate_echo_extract.py \
  --pred-csv data/echo_extracted.csv \
  --gold-csv data/echo_gold.csv \
  --output-summary-csv data/echo_eval_summary.csv \
  --schema-file backend/scripts/echo_feature_schema.example.json
```

Metrics:
- Per-feature exact-match accuracy
- Row-level all-features-correct rate

## 7) Troubleshooting

- **Connection refused / timeout**
  - Verify `llama-server` is running.
  - Confirm URL and port (`--llamacpp-base-url`).
- **Slow inference**
  - Use a smaller GGUF quantization/model.
  - Lower context size if memory is constrained.
- **Malformed model output**
  - Keep temperature low (`0.0`).
  - Ensure server supports `response_format={"type":"json_object"}`.
- **UI will not open**
  - Ensure your Python installation includes `tkinter` (standard on most macOS/Linux Python builds).
- **Resume mismatch**
  - If output schema changed between runs, start with `--no-resume`.
- **Column validation errors**
  - Confirm input contains the selected `--id-column` and `--report-column`.
