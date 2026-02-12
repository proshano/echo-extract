#!/usr/bin/env python3
"""
Simple desktop UI to calibrate per-feature data extraction prompts.

No frontend build required. This script uses tkinter (stdlib) so end users can:
1) Load an input CSV with free-text reports
2) Define/edit schema features and exact per-feature prompts
3) Test prompts on sample reports against local llama.cpp
4) Save schema for use with extract_pipeline.py
5) Manage local llama.cpp/model setup for easier onboarding
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

import extract_pipeline


DEFAULT_MODEL_DIRS = [
    Path.home() / "models",
    Path.home() / "llama-models",
    Path.home() / ".local" / "share" / "llama.cpp" / "models",
]
SLOW_MODEL_DIRS = [Path.home() / ".cache" / "huggingface" / "hub"]
MODEL_SCAN_TIME_BUDGET_SECONDS = 2.0
STATE_DIR = Path.home() / ".data_prompt_calibrator"
STATE_FILE = STATE_DIR / "calibrator_state.json"

RECOMMENDED_MODELS = [
    (
        "Llama 3.2 3B (Q4_K_M)",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Llama 3.1 8B (Q4_K_M)",
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Qwen 2.5 7B (Q4_K_M)",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
    ),
]

ID_COLUMN_CANDIDATES = ["StudyID", "study_id", "id", "patient_id"]
REPORT_COLUMN_CANDIDATES = ["Report", "report_text", "report", "report_body", "narrative"]


class PromptCalibratorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Data Prompt Calibrator")
        self.root.geometry("1350x900")

        self.schema_features: List[Dict[str, Any]] = []
        self.input_rows: List[Dict[str, str]] = []
        self.input_columns: List[str] = []
        self.local_model_paths: List[str] = []
        self.server_models: List[str] = []
        self.server_process: Optional[subprocess.Popen[str]] = None

        self.csv_path_var = tk.StringVar()
        self.schema_path_var = tk.StringVar()
        self.llama_url_var = tk.StringVar(value="http://127.0.0.1:8080")
        self.temperature_var = tk.StringVar(value="0.0")
        self.max_retries_var = tk.StringVar(value="5")
        self.id_column_var = tk.StringVar(value="StudyID")
        self.report_column_var = tk.StringVar(value="Report")
        self.sample_index_var = tk.StringVar(value="1")
        self.local_model_var = tk.StringVar()
        self.server_model_var = tk.StringVar()
        self.recommended_model_var = tk.StringVar(value=RECOMMENDED_MODELS[0][0])
        self.server_port_var = tk.StringVar(value="8080")
        self.csv_status_var = tk.StringVar(value="CSV: not loaded")
        self.sample_status_var = tk.StringVar(value="Sample: none")
        self.server_status_var = tk.StringVar(value="Server: unknown")
        self.model_status_var = tk.StringVar(value="Models: not checked")
        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.state_save_after_id: Optional[str] = None

        self._build_ui()
        self.root.after(60, self._drain_output_queue)
        self._load_persistent_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.append_output(
            "Quick start: 1) Select CSV 2) Refresh local models 3) Start server 4) Refresh server models 5) Test feature."
        )
        self.root.after(250, self.check_setup)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Input CSV").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.csv_path_var, width=75).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse CSV (auto-load)", command=self.browse_csv).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Reload CSV", command=self.load_csv).grid(row=0, column=3, padx=2)

        ttk.Label(top, text="Schema JSON").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.schema_path_var, width=75).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_schema).grid(row=1, column=2, padx=2)
        ttk.Button(top, text="Load Schema", command=self.load_schema).grid(row=1, column=3, padx=2)
        ttk.Button(top, text="Save Schema", command=self.save_schema).grid(row=1, column=4, padx=2)

        ttk.Label(top, text="Llama URL").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.llama_url_var, width=35).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(top, text="Temp").grid(row=2, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.temperature_var, width=8).grid(row=2, column=3, sticky="w", padx=2)
        ttk.Label(top, text="Max retries").grid(row=2, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.max_retries_var, width=8).grid(row=2, column=5, sticky="w", padx=2)

        ttk.Label(top, text="ID column").grid(row=3, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.id_column_var, width=20).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(top, text="Report column").grid(row=3, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.report_column_var, width=20).grid(row=3, column=3, sticky="w", padx=2)
        ttk.Label(top, text="Sample row #").grid(row=3, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.sample_index_var, width=8).grid(row=3, column=5, sticky="w", padx=2)
        ttk.Button(top, text="Show Sample", command=self.show_sample_report).grid(row=3, column=6, padx=2)
        ttk.Label(top, textvariable=self.csv_status_var).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.sample_status_var).grid(row=4, column=4, columnspan=3, sticky="w", pady=(4, 0))

        model_frame = ttk.LabelFrame(self.root, text="llama.cpp / Model Setup (Step-by-step)", padding=8)
        model_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(
            model_frame,
            text="Server models only appear after llama-server is started.",
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))

        ttk.Label(model_frame, text="Step 1 - Local GGUF model").grid(row=1, column=0, sticky="w")
        self.local_model_combo = ttk.Combobox(model_frame, textvariable=self.local_model_var, width=95)
        self.local_model_combo.grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(model_frame, text="Refresh Local Models", command=self.refresh_local_models).grid(row=1, column=2, padx=2)
        ttk.Button(model_frame, text="Deep Scan Models", command=self.deep_scan_local_models).grid(row=1, column=3, padx=2)
        ttk.Button(model_frame, text="Browse Model File", command=self.browse_model_file).grid(row=1, column=4, padx=2)

        ttk.Label(model_frame, text="Step 3 - Server model").grid(row=2, column=0, sticky="w")
        self.server_model_combo = ttk.Combobox(model_frame, textvariable=self.server_model_var, width=50)
        self.server_model_combo.grid(row=2, column=1, sticky="w", padx=4)
        ttk.Button(model_frame, text="Refresh Server Models", command=self.refresh_server_models).grid(row=2, column=2, padx=2)
        ttk.Label(model_frame, text="Port").grid(row=2, column=3, sticky="e")
        ttk.Entry(model_frame, textvariable=self.server_port_var, width=8).grid(row=2, column=4, sticky="w", padx=4)

        ttk.Button(model_frame, text="Check Setup", command=self.check_setup).grid(row=3, column=0, padx=2, pady=2, sticky="w")
        ttk.Button(model_frame, text="Install llama.cpp (macOS/Homebrew)", command=self.install_llamacpp).grid(row=3, column=1, padx=2, pady=2, sticky="w")
        ttk.Button(model_frame, text="Step 2 - Start llama-server", command=self.start_llama_server).grid(row=3, column=2, padx=2, pady=2, sticky="w")
        ttk.Button(model_frame, text="Stop llama-server", command=self.stop_llama_server).grid(row=3, column=3, padx=2, pady=2, sticky="w")

        ttk.Label(model_frame, text="Recommended model").grid(row=4, column=0, sticky="w")
        self.recommended_combo = ttk.Combobox(
            model_frame,
            textvariable=self.recommended_model_var,
            values=[label for (label, _, _) in RECOMMENDED_MODELS],
            width=50,
        )
        self.recommended_combo.grid(row=4, column=1, sticky="w", padx=4)
        ttk.Button(model_frame, text="Download Selected Model", command=self.download_recommended_model).grid(row=4, column=2, padx=2, pady=2, sticky="w")
        ttk.Label(model_frame, textvariable=self.server_status_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(model_frame, textvariable=self.model_status_var).grid(row=5, column=2, columnspan=3, sticky="w", pady=(4, 0))

        model_frame.columnconfigure(1, weight=1)
        top.columnconfigure(1, weight=1)

        middle = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(middle, padding=4)
        right = ttk.Frame(middle, padding=4)
        middle.add(left, weight=1)
        middle.add(right, weight=2)

        ttk.Label(left, text="Features").pack(anchor="w")
        self.feature_listbox = tk.Listbox(left, height=20, exportselection=False)
        self.feature_listbox.pack(fill=tk.BOTH, expand=True)
        self.feature_listbox.bind("<<ListboxSelect>>", lambda _: self.on_feature_selected())

        feature_buttons = ttk.Frame(left)
        feature_buttons.pack(fill=tk.X, pady=4)
        ttk.Button(feature_buttons, text="Add Feature", command=self.add_feature).pack(side=tk.LEFT, padx=2)
        ttk.Button(feature_buttons, text="Delete Feature", command=self.delete_feature).pack(side=tk.LEFT, padx=2)

        form = ttk.LabelFrame(right, text="Selected Feature", padding=6)
        form.pack(fill=tk.X)

        self.feature_name_var = tk.StringVar()
        self.feature_missing_var = tk.StringVar(value="NA")
        self.feature_allowed_var = tk.StringVar()
        self.feature_type_hint_var = tk.StringVar()

        ttk.Label(form, text="Feature name").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.feature_name_var, width=35).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(form, text="Missing value").grid(row=0, column=2, sticky="e")
        ttk.Entry(form, textvariable=self.feature_missing_var, width=12).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(form, text="Allowed values (comma-separated)").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.feature_allowed_var, width=50).grid(row=1, column=1, columnspan=3, sticky="we", padx=4)
        ttk.Label(form, text="Type hint (if no allowed values)").grid(row=2, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.feature_type_hint_var, width=35).grid(row=2, column=1, sticky="we", padx=4)

        ttk.Label(form, text="Description").grid(row=3, column=0, sticky="nw")
        self.description_text = tk.Text(form, height=4, wrap=tk.WORD)
        self.description_text.grid(row=3, column=1, columnspan=3, sticky="we", padx=4, pady=2)

        ttk.Label(form, text="Exact prompt for this feature").grid(row=4, column=0, sticky="nw")
        self.prompt_text = tk.Text(form, height=7, wrap=tk.WORD)
        self.prompt_text.grid(row=4, column=1, columnspan=3, sticky="we", padx=4, pady=2)

        ttk.Button(form, text="Save Feature Edits", command=self.save_feature_edits).grid(row=5, column=3, sticky="e", pady=4)
        form.columnconfigure(1, weight=1)

        sample_frame = ttk.LabelFrame(right, text="Sample Report", padding=6)
        sample_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.sample_report_text = tk.Text(sample_frame, height=10, wrap=tk.WORD)
        self.sample_report_text.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Test Selected Feature", command=self.test_selected_feature).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Test All Features", command=self.test_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Show Full Combined Prompt", command=self.show_combined_prompt).pack(side=tk.LEFT, padx=2)

        output_frame = ttk.LabelFrame(right, text="Calibration Output", padding=6)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def append_output(self, text: str) -> None:
        self.output_queue.put(text)

    def _drain_output_queue(self) -> None:
        lines: List[str] = []
        for _ in range(200):
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        if lines:
            self.output_text.insert(tk.END, "\n".join(lines) + "\n")
            self.output_text.see(tk.END)
        self.root.after(60, self._drain_output_queue)

    def _on_ui(self, callback: Callable[[], None]) -> None:
        self.root.after(0, callback)

    def _build_state_payload(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "csv_path": self.csv_path_var.get().strip(),
            "schema_path": self.schema_path_var.get().strip(),
            "llama_url": self.llama_url_var.get().strip(),
            "temperature": self.temperature_var.get().strip(),
            "max_retries": self.max_retries_var.get().strip(),
            "id_column": self.id_column_var.get().strip(),
            "report_column": self.report_column_var.get().strip(),
            "sample_index": self.sample_index_var.get().strip(),
            "server_port": self.server_port_var.get().strip(),
            "local_model": self.local_model_var.get().strip(),
            "server_model": self.server_model_var.get().strip(),
            "recommended_model": self.recommended_model_var.get().strip(),
            "features": self.schema_features,
        }

    def _save_persistent_state_now(self, announce: bool = False) -> None:
        try:
            payload = self._build_state_payload()
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = STATE_FILE.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(STATE_FILE)
            if announce:
                self.append_output(f"Saved workspace state: {STATE_FILE}")
        except Exception as exc:
            self.append_output(f"Warning: failed to save workspace state: {exc}")

    def _save_persistent_state(self, announce: bool = False) -> None:
        if self.state_save_after_id is not None:
            try:
                self.root.after_cancel(self.state_save_after_id)
            except Exception:
                pass

        def flush_save() -> None:
            self.state_save_after_id = None
            thread = threading.Thread(
                target=self._save_persistent_state_now,
                kwargs={"announce": announce},
                daemon=True,
            )
            thread.start()

        self.state_save_after_id = self.root.after(350, flush_save)

    def _load_persistent_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self.csv_path_var.set(str(payload.get("csv_path", "")))
            self.schema_path_var.set(str(payload.get("schema_path", "")))
            self.llama_url_var.set(str(payload.get("llama_url", self.llama_url_var.get())))
            self.temperature_var.set(str(payload.get("temperature", self.temperature_var.get())))
            self.max_retries_var.set(str(payload.get("max_retries", self.max_retries_var.get())))
            self.id_column_var.set(str(payload.get("id_column", self.id_column_var.get())))
            self.report_column_var.set(str(payload.get("report_column", self.report_column_var.get())))
            self.sample_index_var.set(str(payload.get("sample_index", self.sample_index_var.get())))
            self.server_port_var.set(str(payload.get("server_port", self.server_port_var.get())))
            self.local_model_var.set(str(payload.get("local_model", "")))
            self.server_model_var.set(str(payload.get("server_model", "")))
            self.recommended_model_var.set(
                str(payload.get("recommended_model", self.recommended_model_var.get()))
            )
            features = payload.get("features", [])
            if isinstance(features, list):
                self.schema_features = [item for item in features if isinstance(item, dict)]
            if self.schema_features:
                self.refresh_feature_list()
            if self.csv_path_var.get().strip():
                self.csv_status_var.set(f"CSV path restored: {Path(self.csv_path_var.get()).name}")
            self.append_output(
                f"Loaded saved workspace state with {len(self.schema_features)} feature(s)."
            )
        except Exception as exc:
            self.append_output(f"Warning: could not load saved workspace state: {exc}")

    def _on_close(self) -> None:
        self.save_feature_edits(silent=True)
        self._save_persistent_state_now(announce=False)
        self.stop_llama_server(announce_when_missing=False)
        self.root.destroy()

    def _run_in_thread(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def _normalize_column_name(self, value: str) -> str:
        return str(value).replace("\ufeff", "").strip()

    def _resolve_column_name(self, requested: str) -> str:
        requested_clean = self._normalize_column_name(requested)
        if not requested_clean:
            return ""
        if requested_clean in self.input_columns:
            return requested_clean
        lowered = requested_clean.lower()
        for column in self.input_columns:
            if column.lower() == lowered:
                return column
        return requested_clean

    def _pick_existing_column(self, candidates: List[str]) -> str:
        lowered_to_actual = {column.lower(): column for column in self.input_columns}
        for candidate in candidates:
            existing = lowered_to_actual.get(candidate.lower())
            if existing:
                return existing
        return ""

    def _read_csv_data_sync(self, csv_path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
        encodings_to_try = ["utf-8-sig", "utf-8", "latin-1"]
        last_error = "unknown CSV read error"
        for encoding in encodings_to_try:
            try:
                with csv_path.open("r", encoding=encoding, newline="") as sample_handle:
                    sample = sample_handle.read(16384)
                    sample_handle.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    reader = csv.DictReader(sample_handle, dialect=dialect)
                    raw_headers = list(reader.fieldnames or [])
                    headers: List[str] = [self._normalize_column_name(h) for h in raw_headers if h is not None]
                    if not headers:
                        raise ValueError("No header columns detected.")

                    rows: List[Dict[str, str]] = []
                    for raw_row in reader:
                        row: Dict[str, str] = {}
                        for raw_key, raw_value in raw_row.items():
                            key = self._normalize_column_name(raw_key or "")
                            if not key:
                                continue
                            row[key] = str(raw_value or "")
                        if row:
                            rows.append(row)
                return headers, rows, encoding
            except Exception as exc:
                last_error = str(exc)
                continue
        raise ValueError(f"Failed to parse CSV with supported encodings. Last error: {last_error}")

    def browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)
            self._save_persistent_state(announce=False)
            self.load_csv()

    def browse_schema(self) -> None:
        path = filedialog.askopenfilename(
            title="Select schema JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.schema_path_var.set(path)
            self._save_persistent_state(announce=False)

    def browse_model_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select GGUF model file",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.local_model_var.set(path)
            if path not in self.local_model_paths:
                self.local_model_paths.append(path)
                self.local_model_combo["values"] = self.local_model_paths
            self._save_persistent_state(announce=False)

    def load_csv(self) -> None:
        csv_path = Path(self.csv_path_var.get().strip())
        if not csv_path.exists():
            messagebox.showerror("Error", f"CSV file not found: {csv_path}")
            self.csv_status_var.set("CSV: load failed (file not found)")
            return
        self.csv_status_var.set(f"CSV: loading {csv_path.name} ...")

        def worker() -> None:
            try:
                columns, rows, used_encoding = self._read_csv_data_sync(csv_path)
            except Exception as exc:
                self.append_output(f"CSV load failed: {exc}")
                self._on_ui(lambda: messagebox.showerror("Error", f"Failed to read CSV: {exc}"))
                self._on_ui(lambda: self.csv_status_var.set("CSV: load failed"))
                return

            def apply_loaded_csv() -> None:
                self.input_columns = columns
                self.input_rows = rows

                requested_id_col = self._resolve_column_name(self.id_column_var.get().strip())
                if requested_id_col in self.input_columns:
                    self.id_column_var.set(requested_id_col)
                else:
                    fallback_id = self._pick_existing_column(ID_COLUMN_CANDIDATES)
                    if fallback_id:
                        self.id_column_var.set(fallback_id)

                requested_report_col = self._resolve_column_name(self.report_column_var.get().strip())
                if requested_report_col in self.input_columns:
                    self.report_column_var.set(requested_report_col)
                else:
                    fallback_report = self._pick_existing_column(REPORT_COLUMN_CANDIDATES)
                    if fallback_report:
                        self.report_column_var.set(fallback_report)

                preview_columns = ", ".join(self.input_columns[:6])
                if len(self.input_columns) > 6:
                    preview_columns += ", ..."
                self.append_output(
                    f"Loaded CSV: {csv_path} | rows={len(self.input_rows)} | columns={self.input_columns} | encoding={used_encoding}"
                )
                self.csv_status_var.set(
                    f"CSV loaded: {csv_path.name} | rows={len(self.input_rows)} | columns={len(self.input_columns)} | [{preview_columns}]"
                )
                self.show_sample_report()
                self._save_persistent_state(announce=False)

            self._on_ui(apply_loaded_csv)

        self._run_in_thread(worker)

    def load_schema(self) -> None:
        schema_path = Path(self.schema_path_var.get().strip())
        if not schema_path.exists():
            messagebox.showerror("Error", f"Schema file not found: {schema_path}")
            return
        try:
            self.schema_features = extract_pipeline.load_schema(schema_path)
            self.refresh_feature_list()
            self.append_output(f"Loaded schema: {schema_path} | features={len(self.schema_features)}")
            self._save_persistent_state(announce=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load schema: {exc}")

    def save_schema(self) -> None:
        self.save_feature_edits(silent=True)
        output_path = self.schema_path_var.get().strip()
        if not output_path:
            output_path = filedialog.asksaveasfilename(
                title="Save schema JSON",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not output_path:
                return
            self.schema_path_var.set(output_path)

        payload = {
            "schema_name": "data_extraction_calibrated",
            "missing_default": "NA",
            "features": self.schema_features,
        }
        try:
            Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.append_output(f"Saved schema to: {output_path}")
            self._save_persistent_state(announce=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save schema: {exc}")

    def refresh_feature_list(self) -> None:
        self.feature_listbox.delete(0, tk.END)
        for idx, feature in enumerate(self.schema_features, start=1):
            name = str(feature.get("name", "")).strip() or f"feature_{idx}"
            self.feature_listbox.insert(tk.END, name)
        if self.schema_features:
            self.feature_listbox.selection_clear(0, tk.END)
            self.feature_listbox.selection_set(0)
            self.on_feature_selected()

    def add_feature(self) -> None:
        self.schema_features.append(
            {
                "name": f"feature_{len(self.schema_features) + 1}",
                "description": "",
                "type_hint": "string",
                "missing_value_rule": "NA",
                "prompt": "",
            }
        )
        self.refresh_feature_list()
        self.feature_listbox.selection_clear(0, tk.END)
        self.feature_listbox.selection_set(tk.END)
        self.on_feature_selected()
        self._save_persistent_state(announce=False)

    def delete_feature(self) -> None:
        index = self.get_selected_feature_index()
        if index is None:
            return
        deleted_name = self.schema_features[index].get("name", "")
        del self.schema_features[index]
        self.refresh_feature_list()
        self.append_output(f"Deleted feature: {deleted_name}")
        self._save_persistent_state(announce=False)

    def get_selected_feature_index(self) -> Optional[int]:
        sel = self.feature_listbox.curselection()
        if not sel:
            return None
        return int(sel[0])

    def on_feature_selected(self) -> None:
        idx = self.get_selected_feature_index()
        if idx is None or idx >= len(self.schema_features):
            return
        feature = self.schema_features[idx]
        self.feature_name_var.set(str(feature.get("name", "")))
        self.feature_missing_var.set(str(feature.get("missing_value_rule", "NA")))
        allowed = feature.get("allowed_values")
        self.feature_allowed_var.set(", ".join(allowed) if isinstance(allowed, list) else "")
        self.feature_type_hint_var.set(str(feature.get("type_hint", "")))
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert("1.0", str(feature.get("description", "")))
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", str(feature.get("prompt", "")))

    def save_feature_edits(self, silent: bool = False) -> None:
        idx = self.get_selected_feature_index()
        if idx is None or idx >= len(self.schema_features):
            return

        feature: Dict[str, Any] = {}
        feature["name"] = self.feature_name_var.get().strip()
        feature["description"] = self.description_text.get("1.0", tk.END).strip()
        feature["missing_value_rule"] = self.feature_missing_var.get().strip() or "NA"
        feature["prompt"] = self.prompt_text.get("1.0", tk.END).strip()

        allowed_raw = self.feature_allowed_var.get().strip()
        type_hint = self.feature_type_hint_var.get().strip()
        if allowed_raw:
            feature["allowed_values"] = [p.strip() for p in allowed_raw.split(",") if p.strip()]
        elif type_hint:
            feature["type_hint"] = type_hint
        else:
            feature["type_hint"] = "string"

        self.schema_features[idx] = feature
        self.refresh_feature_list()
        self.feature_listbox.selection_clear(0, tk.END)
        self.feature_listbox.selection_set(idx)
        if not silent:
            self.append_output(f"Saved edits for feature: {feature['name']}")
        self._save_persistent_state(announce=False)

    def get_sample_row(self) -> Optional[Dict[str, str]]:
        if not self.input_rows:
            return None
        try:
            idx = int(self.sample_index_var.get()) - 1
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(self.input_rows) - 1))
        return self.input_rows[idx]

    def show_sample_report(self) -> None:
        row = self.get_sample_row()
        self.sample_report_text.delete("1.0", tk.END)
        if row is None:
            self.sample_report_text.insert("1.0", "Load an input CSV first.")
            self.sample_status_var.set("Sample: none (load CSV first)")
            return

        report_col = self._resolve_column_name(self.report_column_var.get().strip())
        report_text = str(row.get(report_col, ""))
        used_column = report_col
        fallback_used = False
        if not report_text.strip():
            for candidate in REPORT_COLUMN_CANDIDATES:
                candidate_key = self._resolve_column_name(candidate)
                candidate_text = str(row.get(candidate_key, "")).strip()
                if candidate_text:
                    report_text = candidate_text
                    used_column = candidate_key
                    fallback_used = candidate_key != report_col
                    break
        if not report_text.strip():
            for col_name in self.input_columns:
                candidate_text = str(row.get(col_name, "")).strip()
                if candidate_text:
                    report_text = candidate_text
                    used_column = col_name
                    fallback_used = col_name != report_col
                    break
        if fallback_used:
            self.append_output(
                f"Show Sample: selected report column '{report_col}' was empty/missing for this row; showing '{used_column}' instead."
            )
        self.sample_report_text.insert("1.0", report_text)
        sample_row = self.get_sample_row() or {}
        id_col = self._resolve_column_name(self.id_column_var.get().strip())
        sample_id = str(sample_row.get(id_col, "")).strip() or "N/A"
        sample_number = self.sample_index_var.get().strip() or "1"
        preview_len = len(report_text.strip())
        self.sample_status_var.set(
            f"Sample row {sample_number} | {id_col}={sample_id} | source_col={used_column} | chars={preview_len}"
        )

    def build_single_feature_prompt(self, feature: Dict[str, Any], report_text: str) -> str:
        name = str(feature.get("name", "")).strip()
        description = str(feature.get("description", "")).strip()
        missing_value = str(feature.get("missing_value_rule", "NA")).strip() or "NA"
        feature_prompt = str(feature.get("prompt", "")).strip()

        if "allowed_values" in feature:
            allowed = json.dumps(feature.get("allowed_values", []), ensure_ascii=True)
            value_rule = f"Allowed values: {allowed}."
        else:
            value_rule = f"Type hint: {json.dumps(feature.get('type_hint', 'string'))}."

        return (
            "Extract one structured feature from the report.\n"
            "Return EXACTLY one JSON object with one key: \"value\".\n"
            f"Feature name: {name}\n"
            f"Feature description: {description}\n"
            f"{value_rule}\n"
            f"If not documented, return \"{missing_value}\".\n"
            f"Feature-specific prompt guidance: {feature_prompt}\n\n"
            "Report text:\n"
            f"{report_text}\n"
        )

    def _call_llm(
        self,
        prompt: str,
        base_url: str,
        temperature: float,
        max_retries: int,
        selected_server_model: str,
        has_selected_local_model: bool,
    ) -> Dict[str, Any]:
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            return {"success": False, "content": "", "error": "Llama URL is empty.", "model": ""}

        # Fast preflight to avoid long retry waits when server is unavailable.
        try:
            health = requests.get(f"{base_url}/v1/models", timeout=3)
            health.raise_for_status()
        except Exception as exc:
            return {
                "success": False,
                "content": "",
                "error": f"llama.cpp server is not reachable at {base_url}: {exc}",
                "model": "",
            }

        result = extract_pipeline.call_llamacpp_with_retries(
            base_url=base_url,
            prompt=prompt,
            temperature=float(temperature),
            max_retries=int(max_retries),
            model=selected_server_model,
        )
        if not result.get("success"):
            raw_error = str(result.get("error", "") or "")
            lowered = raw_error.lower()
            if "connection refused" in lowered or "failed to establish a new connection" in lowered:
                guidance = (
                    f"Cannot connect to llama.cpp at {base_url}. "
                    "Start llama-server and verify the port."
                )
                if has_selected_local_model:
                    guidance += " You already selected a local model, so click 'Start llama-server'."
                else:
                    guidance += " Select a local GGUF model first, then click 'Start llama-server'."
                result["error"] = guidance
        return result

    def _get_inference_readiness_error(
        self,
        base_url: str,
        selected_local_model: str,
        selected_server_model: str,
    ) -> Optional[str]:
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            return "Llama URL is empty."

        if not selected_local_model and not selected_server_model:
            return (
                "No model selected. Pick a local GGUF model (or refresh server models), "
                "then start llama-server."
            )

        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            response.raise_for_status()
        except Exception as exc:
            if selected_local_model:
                return (
                    f"llama.cpp is not reachable at {base_url}: {exc}. "
                    "Click 'Start llama-server'."
                )
            return (
                f"llama.cpp is not reachable at {base_url}: {exc}. "
                "Select a local model and click 'Start llama-server'."
            )

        return None

    def test_selected_feature(self) -> None:
        self.save_feature_edits(silent=True)
        row = self.get_sample_row()
        idx = self.get_selected_feature_index()
        if row is None:
            messagebox.showerror("Error", "Load an input CSV first.")
            return
        if idx is None:
            messagebox.showerror("Error", "Select a feature first.")
            return

        report_col = self._resolve_column_name(self.report_column_var.get().strip())
        report_text = str(row.get(report_col, "")).strip()
        feature = self.schema_features[idx]
        prompt = self.build_single_feature_prompt(feature, report_text)
        base_url = self.llama_url_var.get().strip()
        selected_local_model = self.local_model_var.get().strip()
        selected_server_model = self.server_model_var.get().strip()
        try:
            temperature = float(self.temperature_var.get().strip() or "0.0")
        except ValueError:
            temperature = 0.0
        try:
            max_retries = int(self.max_retries_var.get().strip() or "5")
        except ValueError:
            max_retries = 5

        self.append_output(f"Starting selected feature test: {feature.get('name', '')}")

        def worker() -> None:
            readiness_error = self._get_inference_readiness_error(
                base_url=base_url,
                selected_local_model=selected_local_model,
                selected_server_model=selected_server_model,
            )
            if readiness_error:
                self.append_output(f"LLM setup error: {readiness_error}")
                self._on_ui(lambda: messagebox.showerror("LLM setup required", readiness_error))
                return
            result = self._call_llm(
                prompt=prompt,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries,
                selected_server_model=selected_server_model,
                has_selected_local_model=bool(selected_local_model),
            )
            raw = str(result.get("content", "") or "")
            if not result.get("success"):
                self.append_output(f"LLM error: {result.get('error', '')}")
                return
            parsed, parse_error = extract_pipeline.parse_llm_json_response(raw)
            if parse_error:
                self.append_output(f"Parse error: {parse_error}")
                self.append_output(f"Raw: {raw}")
                return
            value = parsed.get("value", feature.get("missing_value_rule", "NA"))
            self.append_output(f"Value => {feature.get('name')}: {value}")
            self.append_output(f"Raw => {raw}")

        self._run_in_thread(worker)

    def test_all_features(self) -> None:
        self.save_feature_edits(silent=True)
        row = self.get_sample_row()
        if row is None:
            messagebox.showerror("Error", "Load an input CSV first.")
            return
        report_col = self._resolve_column_name(self.report_column_var.get().strip())
        report_text = str(row.get(report_col, "")).strip()
        features = list(self.schema_features)
        if not features:
            messagebox.showerror("Error", "Define at least one feature.")
            return
        base_url = self.llama_url_var.get().strip()
        selected_local_model = self.local_model_var.get().strip()
        selected_server_model = self.server_model_var.get().strip()
        try:
            temperature = float(self.temperature_var.get().strip() or "0.0")
        except ValueError:
            temperature = 0.0
        try:
            max_retries = int(self.max_retries_var.get().strip() or "5")
        except ValueError:
            max_retries = 5

        self.append_output(f"Starting all-feature test for {len(features)} feature(s).")

        def worker() -> None:
            readiness_error = self._get_inference_readiness_error(
                base_url=base_url,
                selected_local_model=selected_local_model,
                selected_server_model=selected_server_model,
            )
            if readiness_error:
                self.append_output(f"LLM setup error: {readiness_error}")
                self._on_ui(lambda: messagebox.showerror("LLM setup required", readiness_error))
                return
            self.append_output("Running all-feature calibration on sample row...")
            for feature in features:
                prompt = self.build_single_feature_prompt(feature, report_text)
                result = self._call_llm(
                    prompt=prompt,
                    base_url=base_url,
                    temperature=temperature,
                    max_retries=max_retries,
                    selected_server_model=selected_server_model,
                    has_selected_local_model=bool(selected_local_model),
                )
                name = str(feature.get("name", ""))
                if not result.get("success"):
                    self.append_output(f"{name}: LLM error => {result.get('error', '')}")
                    continue
                raw = str(result.get("content", "") or "")
                parsed, parse_error = extract_pipeline.parse_llm_json_response(raw)
                if parse_error:
                    self.append_output(f"{name}: parse_error => {parse_error}")
                    continue
                value = parsed.get("value", feature.get("missing_value_rule", "NA"))
                self.append_output(f"{name}: {value}")
            self.append_output("All-feature calibration pass complete.")

        self._run_in_thread(worker)

    def show_combined_prompt(self) -> None:
        self.save_feature_edits(silent=True)
        row = self.get_sample_row()
        if row is None:
            messagebox.showerror("Error", "Load an input CSV first.")
            return
        report_col = self._resolve_column_name(self.report_column_var.get().strip())
        report_text = str(row.get(report_col, "")).strip()
        combined_prompt = extract_pipeline.build_prompt(report_text=report_text, schema_features=self.schema_features)
        self.append_output("----- Combined prompt preview start -----")
        self.append_output(combined_prompt)
        self.append_output("----- Combined prompt preview end -----")

    def _get_llama_server_binary(self) -> str:
        return shutil.which("llama-server") or ""

    def _find_local_gguf_models(self, include_slow_dirs: bool = False) -> tuple[List[str], bool]:
        search_dirs = list(DEFAULT_MODEL_DIRS)
        if include_slow_dirs:
            search_dirs.extend(SLOW_MODEL_DIRS)

        env_value = os.environ.get("LLM_MODELS_DIR", "").strip()
        if env_value:
            search_dirs.insert(0, Path(env_value))

        found: List[str] = []
        seen = set()
        timed_out = False
        started = time.perf_counter()

        for directory in search_dirs:
            if not directory.exists():
                continue
            try:
                for root_dir, _, file_names in os.walk(directory):
                    if time.perf_counter() - started > MODEL_SCAN_TIME_BUDGET_SECONDS:
                        timed_out = True
                        break
                    for file_name in file_names:
                        if not file_name.lower().endswith(".gguf"):
                            continue
                        model_path = Path(root_dir) / file_name
                        model_str = str(model_path.resolve())
                        if model_str in seen:
                            continue
                        seen.add(model_str)
                        found.append(model_str)
                if timed_out:
                    break
            except Exception:
                continue

        found.sort()
        return found, timed_out

    def _refresh_local_models_sync(self, include_slow_dirs: bool = False) -> None:
        models, timed_out = self._find_local_gguf_models(include_slow_dirs=include_slow_dirs)
        self._on_ui(lambda: self._apply_local_model_refresh(models, timed_out))

    def _apply_local_model_refresh(self, models: List[str], timed_out: bool) -> None:
        self.local_model_paths = models
        self.local_model_combo["values"] = models
        if models and not self.local_model_var.get().strip():
            self.local_model_var.set(models[0])
        self.append_output(f"Found {len(models)} local GGUF model(s).")
        if models:
            selected_name = Path(models[0]).name
            self.model_status_var.set(f"Models: local found ({len(models)}), selected={selected_name}")
        else:
            self.model_status_var.set("Models: no local GGUF found yet")
        if timed_out:
            self.append_output("Model scan timed out for responsiveness. Use 'Deep Scan Models' for a broader scan.")
        if not models:
            self.append_output("No GGUF models found. Use 'Download Selected Model' to fetch one.")
        self._save_persistent_state(announce=False)

    def refresh_local_models(self) -> None:
        self._run_in_thread(lambda: self._refresh_local_models_sync(include_slow_dirs=False))

    def deep_scan_local_models(self) -> None:
        self._run_in_thread(lambda: self._refresh_local_models_sync(include_slow_dirs=True))

    def _refresh_server_models_sync(self, base_url: str) -> None:
        base_url = base_url.strip().rstrip("/")
        endpoint = f"{base_url}/v1/models"
        models: List[str] = []
        error_message = ""
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            for item in data:
                if isinstance(item, dict):
                    model_id = str(item.get("id", "")).strip()
                    if model_id:
                        models.append(model_id)
        except Exception as exc:
            error_message = f"Could not read {endpoint}: {exc}"
        self._on_ui(lambda: self._apply_server_model_refresh(models, error_message))

    def _apply_server_model_refresh(self, models: List[str], error_message: str) -> None:
        if error_message:
            self.append_output(error_message)
            self.server_status_var.set("Server: unreachable")
            return
        self.server_models = models
        self.server_model_combo["values"] = models
        if models and not self.server_model_var.get().strip():
            self.server_model_var.set(models[0])
        if models:
            self.server_status_var.set(f"Server: running, {len(models)} model(s) available")
        else:
            self.server_status_var.set("Server: reachable, but no models reported")
        self.append_output(f"Server models: {models if models else 'none reported'}")
        self._save_persistent_state(announce=False)

    def refresh_server_models(self) -> None:
        base_url = self.llama_url_var.get().strip()
        self._run_in_thread(lambda: self._refresh_server_models_sync(base_url=base_url))

    def check_setup(self) -> None:
        base_url = self.llama_url_var.get().strip()

        def worker() -> None:
            binary = self._get_llama_server_binary()
            if binary:
                self.append_output(f"llama-server found: {binary}")
            else:
                self.append_output("llama-server not found in PATH.")
                self.append_output("Click 'Install llama.cpp (Homebrew)' or install manually.")
            self._on_ui(lambda: self.server_status_var.set("Server: checking..."))
            self._on_ui(lambda: self.model_status_var.set("Models: checking..."))
            self._refresh_local_models_sync(include_slow_dirs=False)
            self._refresh_server_models_sync(base_url=base_url)

        self._run_in_thread(worker)

    def _stream_subprocess_output(self, process: subprocess.Popen[str], prefix: str) -> int:
        if process.stdout is None:
            return process.wait()
        for line in process.stdout:
            self.append_output(f"{prefix}{line.rstrip()}")
        return process.wait()

    def install_llamacpp(self) -> None:
        def worker() -> None:
            if self._get_llama_server_binary():
                self.append_output("llama-server is already installed.")
                return

            system_name = platform.system()
            if system_name != "Darwin":
                self.append_output("Automatic install currently supports macOS Homebrew flow.")
                self.append_output("Manual option: https://github.com/ggerganov/llama.cpp")
                return

            brew = shutil.which("brew")
            if not brew:
                self.append_output("Homebrew is not installed. Install Homebrew first: https://brew.sh")
                return

            self.append_output("Installing llama.cpp with Homebrew...")
            process = subprocess.Popen(
                [brew, "install", "llama.cpp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            code = self._stream_subprocess_output(process, prefix="[brew] ")
            if code == 0:
                self.append_output("llama.cpp installation complete.")
            else:
                self.append_output(f"llama.cpp installation failed with exit code {code}.")
            self._on_ui(self.check_setup)

        self._run_in_thread(worker)

    def _ensure_hf_cli(self) -> bool:
        hf_cli = shutil.which("huggingface-cli")
        if hf_cli:
            return True

        self.append_output("huggingface-cli not found. Attempting automatic install...")
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = self._stream_subprocess_output(process, prefix="[pip] ")
        if code != 0:
            self.append_output("Automatic install failed. Run manually: pip install -U \"huggingface_hub[cli]\"")
            return False
        self.append_output("Installed huggingface_hub CLI successfully.")
        return shutil.which("huggingface-cli") is not None

    def download_recommended_model(self) -> None:
        def worker() -> None:
            selected_label = self.recommended_model_var.get().strip()
            choice = None
            for entry in RECOMMENDED_MODELS:
                if entry[0] == selected_label:
                    choice = entry
                    break
            if choice is None:
                self.append_output("Select a recommended model first.")
                return

            _, repo_id, filename = choice
            if not self._ensure_hf_cli():
                return
            hf_cli = shutil.which("huggingface-cli")
            if not hf_cli:
                self.append_output("huggingface-cli is still unavailable after install attempt.")
                return

            models_dir = Path.home() / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            self.append_output(f"Downloading {filename} from {repo_id} ...")
            process = subprocess.Popen(
                [hf_cli, "download", repo_id, filename, "--local-dir", str(models_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            code = self._stream_subprocess_output(process, prefix="[hf] ")
            if code == 0:
                self.append_output("Model download complete.")
            else:
                self.append_output(f"Model download failed with exit code {code}.")
            self.refresh_local_models()

        self._run_in_thread(worker)

    def _terminate_llama_process(self, process: subprocess.Popen[str]) -> bool:
        if process.poll() is not None:
            return True

        used_process_group = False
        process_group_id: Optional[int] = None

        if os.name != "nt":
            try:
                process_group_id = os.getpgid(process.pid)
            except (OSError, ProcessLookupError):
                process_group_id = None

            if process_group_id is not None and process_group_id == process.pid:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                    used_process_group = True
                except ProcessLookupError:
                    return True
                except OSError:
                    used_process_group = False

        if not used_process_group:
            try:
                process.terminate()
            except OSError:
                return True

        try:
            process.wait(timeout=5)
            return True
        except subprocess.TimeoutExpired:
            pass

        if used_process_group and process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                return True
            except OSError:
                try:
                    process.kill()
                except OSError:
                    return True
        else:
            try:
                process.kill()
            except OSError:
                return True

        try:
            process.wait(timeout=3)
            return True
        except subprocess.TimeoutExpired:
            return False

    def start_llama_server(self) -> None:
        model_path = self.local_model_var.get().strip()
        binary = self._get_llama_server_binary()
        if not binary:
            messagebox.showerror("Error", "llama-server not found. Install llama.cpp first.")
            return
        if not model_path:
            messagebox.showerror("Error", "Select a local GGUF model file first.")
            return
        if not Path(model_path).exists():
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        if self.server_process and self.server_process.poll() is None:
            self.append_output("llama-server is already running from this UI session.")
            self.server_status_var.set("Server: already running")
            return

        port = self.server_port_var.get().strip() or "8080"
        if not re.fullmatch(r"\d{2,5}", port):
            messagebox.showerror("Error", f"Invalid port: {port}")
            return

        command = [binary, "-m", model_path, "--port", port, "--ctx-size", "8192"]
        self.append_output(f"Starting llama-server: {' '.join(command)}")

        try:
            popen_kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if creation_flags:
                    popen_kwargs["creationflags"] = creation_flags
            else:
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(command, **popen_kwargs)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to start llama-server: {exc}")
            return

        self.server_process = process
        self.llama_url_var.set(f"http://127.0.0.1:{port}")
        self.server_status_var.set("Server: starting...")

        def stream_worker() -> None:
            exit_code = self._stream_subprocess_output(process, prefix="[llama-server] ")
            self.append_output(f"llama-server exited with code {exit_code}.")
            self._on_ui(lambda: self.server_status_var.set("Server: not running"))

        self._run_in_thread(stream_worker)
        self.append_output("llama-server launched. Use 'Refresh Server Models' in a few seconds.")

    def stop_llama_server(self, *, announce_when_missing: bool = True) -> None:
        process = self.server_process
        if process is None or process.poll() is not None:
            if announce_when_missing:
                self.append_output("No llama-server process started from this UI is running.")
            self.server_status_var.set("Server: not running")
            self.server_process = None
            return

        self.server_status_var.set("Server: stopping...")
        stopped = self._terminate_llama_process(process)
        self.server_process = None
        if stopped:
            self.append_output("llama-server stopped.")
        else:
            self.append_output("Warning: llama-server may still be running after stop attempt.")
        self.server_status_var.set("Server: not running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the data prompt calibrator UI.")
    parser.add_argument(
        "--schema-file",
        default="",
        help="Optional schema file to preload.",
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Optional input CSV to preload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    app = PromptCalibratorApp(root)
    if args.schema_file:
        app.schema_path_var.set(args.schema_file)
        app.load_schema()
    if args.input_csv:
        app.csv_path_var.set(args.input_csv)
        app.load_csv()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
