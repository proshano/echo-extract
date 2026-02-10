#!/usr/bin/env python3
"""
Standalone, resume-safe echo report extraction CLI.

This script reads free-text reports from an input CSV, calls a local llama.cpp
OpenAI-compatible endpoint, and writes one output row per input row with
immediate disk persistence.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_ID_COLUMN = "StudyID"
DEFAULT_REPORT_COLUMN = "Report"
DEFAULT_MISSING_VALUE = "NA"
PROGRESS_EVERY = 25
REQUEST_TIMEOUT_SECONDS = 300


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured echo features from free-text reports using local llama.cpp."
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN, help="Input ID column name")
    parser.add_argument(
        "--report-column",
        default=DEFAULT_REPORT_COLUMN,
        help="Input report text column name",
    )
    parser.add_argument("--schema-file", required=True, help="Path to JSON feature schema")
    parser.add_argument(
        "--llamacpp-base-url",
        default=DEFAULT_BASE_URL,
        help="llama.cpp OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per LLM call",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input rows to scan",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing output (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and overwrite output file",
    )
    parser.add_argument(
        "--write-raw-response",
        dest="write_raw_response",
        action="store_true",
        default=True,
        help="Write raw model output column (default: enabled)",
    )
    parser.add_argument(
        "--no-write-raw-response",
        dest="write_raw_response",
        action="store_false",
        help="Disable raw response column",
    )
    parser.add_argument(
        "--model-label",
        default="",
        help="Optional model metadata label stored in output _model column",
    )
    parser.add_argument(
        "--llamacpp-model",
        default="",
        help="Optional model name/id sent to /v1/chat/completions payload",
    )
    return parser.parse_args(argv)


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_schema(schema_path: Path) -> List[Dict[str, Any]]:
    if not schema_path.exists():
        raise ValueError(f"Schema file not found: {schema_path}")

    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid schema JSON in {schema_path}: {exc}") from exc

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Schema must include a non-empty 'features' list.")

    validated: List[Dict[str, Any]] = []
    names_seen: Set[str] = set()
    for idx, item in enumerate(features):
        if not isinstance(item, dict):
            raise ValueError(f"Schema feature #{idx} must be an object.")

        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        missing_rule = str(item.get("missing_value_rule", DEFAULT_MISSING_VALUE)).strip()
        has_allowed_values = "allowed_values" in item
        has_type_hint = "type_hint" in item

        if not name:
            raise ValueError(f"Schema feature #{idx} is missing 'name'.")
        if name in names_seen:
            raise ValueError(f"Duplicate feature name in schema: {name}")
        if not description:
            raise ValueError(f"Schema feature '{name}' is missing 'description'.")
        if not has_allowed_values and not has_type_hint:
            raise ValueError(
                f"Schema feature '{name}' must include either 'allowed_values' or 'type_hint'."
            )
        if not missing_rule:
            raise ValueError(f"Schema feature '{name}' has empty 'missing_value_rule'.")

        names_seen.add(name)
        normalized = dict(item)
        normalized["name"] = name
        normalized["description"] = description
        normalized["missing_value_rule"] = missing_rule
        normalized["prompt"] = str(item.get("prompt", "")).strip()
        validated.append(normalized)

    return validated


def get_schema_defaults(schema_features: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    defaults: Dict[str, str] = {}
    for feature in schema_features:
        defaults[feature["name"]] = str(feature.get("missing_value_rule", DEFAULT_MISSING_VALUE))
    return defaults


def load_input_rows(
    input_csv: Path,
    id_column: str,
    report_column: str,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    if not input_csv.exists():
        raise ValueError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if id_column not in fieldnames:
            raise ValueError(
                f"Missing required ID column '{id_column}'. Found columns: {fieldnames}"
            )
        if report_column not in fieldnames:
            raise ValueError(
                f"Missing required report column '{report_column}'. Found columns: {fieldnames}"
            )

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    return rows


def build_output_fieldnames(
    id_column: str,
    schema_features: Sequence[Dict[str, Any]],
    write_raw_response: bool,
) -> List[str]:
    fieldnames = [id_column] + [f["name"] for f in schema_features]
    fieldnames.extend(["_status", "_error", "_processed_at", "_model"])
    if write_raw_response:
        fieldnames.append("_raw_response")
    return fieldnames


def load_processed_ids(output_csv: Path, id_column: str) -> Set[str]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return set()
        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Output CSV exists but has no '{id_column}' column. Cannot resume safely."
            )

        processed: Set[str] = set()
        for row in reader:
            processed.add(str(row.get(id_column, "")).strip())
    return processed


def sanitize_llm_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

    return cleaned


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_llm_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw_response or not raw_response.strip():
        return None, "Empty LLM response."

    cleaned = sanitize_llm_text(raw_response)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "LLM response is valid JSON but not an object."
    except json.JSONDecodeError:
        pass

    extracted = extract_first_json_object(cleaned)
    if extracted is None:
        return None, "Could not find JSON object in LLM response."

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse extracted JSON object: {exc}"

    if not isinstance(parsed, dict):
        return None, "Extracted JSON is not an object."
    return parsed, None


def normalize_feature_values(
    parsed: Dict[str, Any],
    schema_features: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for feature in schema_features:
        name = feature["name"]
        missing_value = str(feature.get("missing_value_rule", DEFAULT_MISSING_VALUE))
        value = parsed.get(name, missing_value)
        if value is None:
            normalized[name] = missing_value
        elif isinstance(value, (dict, list)):
            normalized[name] = json.dumps(value, ensure_ascii=True)
        else:
            value_str = str(value).strip()
            normalized[name] = value_str if value_str else missing_value
    return normalized


def build_prompt(report_text: str, schema_features: Sequence[Dict[str, Any]]) -> str:
    feature_lines: List[str] = []
    for feature in schema_features:
        name = feature["name"]
        description = feature["description"]
        missing_value = feature.get("missing_value_rule", DEFAULT_MISSING_VALUE)
        if "allowed_values" in feature:
            allowed = feature["allowed_values"]
            allowed_text = json.dumps(allowed, ensure_ascii=True)
            detail = f"allowed_values={allowed_text}"
        else:
            detail = f"type_hint={json.dumps(feature.get('type_hint', 'string'))}"
        feature_prompt = str(feature.get("prompt", "")).strip()
        prompt_detail = (
            f" Extraction guidance: {feature_prompt}"
            if feature_prompt
            else ""
        )

        feature_lines.append(
            f'- "{name}": {description}. {detail}. Missing rule: use "{missing_value}".{prompt_detail}'
        )

    schema_key_list = [feature["name"] for feature in schema_features]

    return (
        "You are extracting structured echocardiography findings from a single report.\n"
        "Return EXACTLY one JSON object and no additional text.\n"
        "Rules:\n"
        "1) Use only the schema keys listed below.\n"
        "2) If a finding is not explicitly documented, return the schema missing value.\n"
        "3) Do not invent findings.\n"
        "4) Output must be valid JSON object.\n\n"
        f"Schema keys (exact): {json.dumps(schema_key_list, ensure_ascii=True)}\n"
        "Feature definitions:\n"
        f"{chr(10).join(feature_lines)}\n\n"
        "Report text:\n"
        f"{report_text}\n"
    )


def call_llamacpp_with_retries(
    base_url: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    model: str = "",
) -> Dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    model_name_arg = model.strip()
    if model_name_arg:
        payload["model"] = model_name_arg

    last_error = ""
    last_content = ""
    model_name = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            body = response.json()
            model_name = str(body.get("model", "")).strip()

            choices = body.get("choices", [])
            content = ""
            if isinstance(choices, list) and choices:
                content = str(choices[0].get("message", {}).get("content", "")).strip()

            if content:
                return {"success": True, "content": content, "error": "", "model": model_name}

            last_error = "LLM response content is empty."
            last_content = ""
        except requests.RequestException as exc:
            last_error = f"HTTP/request error: {exc}"
        except (ValueError, KeyError, TypeError) as exc:
            last_error = f"Invalid JSON response body: {exc}"

        if attempt < max_retries:
            delay_seconds = min(10.0, 1.5 * (2 ** (attempt - 1)))
            time.sleep(delay_seconds)

    return {
        "success": False,
        "content": last_content,
        "error": last_error or "Unknown LLM error.",
        "model": model_name,
    }


def write_row_with_flush(
    writer: csv.DictWriter,
    file_handle: Any,
    row: Dict[str, str],
) -> None:
    writer.writerow(row)
    file_handle.flush()
    os.fsync(file_handle.fileno())


def maybe_print_progress(stats: Dict[str, int], scanned: int, total_rows: int) -> None:
    if scanned % PROGRESS_EVERY == 0 or scanned == total_rows:
        print(
            f"[progress] scanned={scanned}/{total_rows} processed={stats['processed']} "
            f"skipped={stats['skipped']} errors={stats['llm_error'] + stats['parse_error']}"
        )


def build_error_row(
    record_id: str,
    id_column: str,
    feature_defaults: Dict[str, str],
    status: str,
    error: str,
    model_value: str,
    write_raw_response: bool,
    raw_response: str,
) -> Dict[str, str]:
    row: Dict[str, str] = {id_column: record_id}
    row.update(feature_defaults)
    row["_status"] = status
    row["_error"] = error
    row["_processed_at"] = now_utc_iso()
    row["_model"] = model_value
    if write_raw_response:
        row["_raw_response"] = raw_response
    return row


def run_extraction(args: argparse.Namespace) -> Dict[str, int]:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    schema_file = Path(args.schema_file)

    schema_features = load_schema(schema_file)
    feature_defaults = get_schema_defaults(schema_features)
    input_rows = load_input_rows(
        input_csv=input_csv,
        id_column=args.id_column,
        report_column=args.report_column,
        limit=args.limit,
    )
    output_fieldnames = build_output_fieldnames(
        id_column=args.id_column,
        schema_features=schema_features,
        write_raw_response=args.write_raw_response,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    processed_ids: Set[str] = set()
    append_mode = args.resume
    if args.resume and output_csv.exists():
        processed_ids = load_processed_ids(output_csv, args.id_column)
    elif not args.resume and output_csv.exists():
        output_csv.unlink()

    file_mode = "a" if append_mode and output_csv.exists() else "w"
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0 or file_mode == "w"

    stats = {"processed": 0, "skipped": 0, "ok": 0, "llm_error": 0, "parse_error": 0}

    with output_csv.open(file_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        total_rows = len(input_rows)
        for idx, source_row in enumerate(input_rows, start=1):
            record_id = str(source_row.get(args.id_column, "")).strip()
            report_text = str(source_row.get(args.report_column, "")).strip()

            if args.resume and record_id in processed_ids:
                stats["skipped"] += 1
                maybe_print_progress(stats=stats, scanned=idx, total_rows=total_rows)
                continue

            if not record_id:
                record_id = f"row_{idx}"

            prompt = build_prompt(report_text=report_text, schema_features=schema_features)
            llm_result = call_llamacpp_with_retries(
                base_url=args.llamacpp_base_url,
                prompt=prompt,
                temperature=args.temperature,
                max_retries=args.max_retries,
                model=getattr(args, "llamacpp_model", ""),
            )
            raw_response = str(llm_result.get("content", "") or "")
            model_value = args.model_label or str(llm_result.get("model", "")).strip()

            if not llm_result.get("success"):
                row = build_error_row(
                    record_id=record_id,
                    id_column=args.id_column,
                    feature_defaults=feature_defaults,
                    status="llm_error",
                    error=str(llm_result.get("error", "Unknown LLM error.")),
                    model_value=model_value,
                    write_raw_response=args.write_raw_response,
                    raw_response=raw_response,
                )
                write_row_with_flush(writer=writer, file_handle=handle, row=row)
                stats["processed"] += 1
                stats["llm_error"] += 1
                processed_ids.add(record_id)
                maybe_print_progress(stats=stats, scanned=idx, total_rows=total_rows)
                continue

            parsed, parse_error = parse_llm_json_response(raw_response)
            if parse_error:
                row = build_error_row(
                    record_id=record_id,
                    id_column=args.id_column,
                    feature_defaults=feature_defaults,
                    status="parse_error",
                    error=parse_error,
                    model_value=model_value,
                    write_raw_response=args.write_raw_response,
                    raw_response=raw_response,
                )
                write_row_with_flush(writer=writer, file_handle=handle, row=row)
                stats["processed"] += 1
                stats["parse_error"] += 1
                processed_ids.add(record_id)
                maybe_print_progress(stats=stats, scanned=idx, total_rows=total_rows)
                continue

            feature_values = normalize_feature_values(parsed=parsed or {}, schema_features=schema_features)
            row = {args.id_column: record_id, **feature_values}
            row["_status"] = "ok"
            row["_error"] = ""
            row["_processed_at"] = now_utc_iso()
            row["_model"] = model_value
            if args.write_raw_response:
                row["_raw_response"] = raw_response

            write_row_with_flush(writer=writer, file_handle=handle, row=row)
            stats["processed"] += 1
            stats["ok"] += 1
            processed_ids.add(record_id)
            maybe_print_progress(stats=stats, scanned=idx, total_rows=total_rows)

    print(
        "[done] "
        f"processed={stats['processed']} ok={stats['ok']} "
        f"llm_error={stats['llm_error']} parse_error={stats['parse_error']} "
        f"skipped={stats['skipped']}"
    )
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run_extraction(args)
        return 0
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[error] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] Unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
