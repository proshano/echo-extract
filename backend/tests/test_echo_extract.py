import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import echo_extract  # noqa: E402


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class EchoExtractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.schema_path = self.base / "schema.json"
        schema_payload = {
            "features": [
                {
                    "name": "lvef_percent",
                    "description": "LVEF value",
                    "type_hint": "numeric_or_NA",
                    "missing_value_rule": "NA",
                },
                {
                    "name": "rv_function",
                    "description": "RV function category",
                    "allowed_values": ["normal", "reduced", "NA"],
                    "missing_value_rule": "NA",
                },
            ]
        }
        self.schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parser_valid_json_response(self):
        raw = '{"lvef_percent":"55","rv_function":"normal"}'
        parsed, error = echo_extract.parse_llm_json_response(raw)
        self.assertIsNone(error)
        self.assertEqual(parsed["lvef_percent"], "55")
        self.assertEqual(parsed["rv_function"], "normal")

    def test_parser_wrapped_json_fallback(self):
        raw = 'Result:\\n```json\\n{"lvef_percent":"45","rv_function":"reduced"}\\n```\\nthanks'
        parsed, error = echo_extract.parse_llm_json_response(raw)
        self.assertIsNone(error)
        self.assertEqual(parsed["lvef_percent"], "45")
        self.assertEqual(parsed["rv_function"], "reduced")

    def test_build_prompt_includes_feature_prompt_guidance(self):
        features = [
            {
                "name": "lvef_percent",
                "description": "LVEF value",
                "type_hint": "numeric_or_NA",
                "missing_value_rule": "NA",
                "prompt": "Use explicit numeric EF value only.",
            }
        ]
        prompt = echo_extract.build_prompt("EF estimated at 50%.", features)
        self.assertIn("Use explicit numeric EF value only.", prompt)
        self.assertIn("lvef_percent", prompt)

    def test_resume_skips_existing_output_ids(self):
        input_csv = self.base / "input.csv"
        output_csv = self.base / "output.csv"

        write_csv(
            input_csv,
            ["record_id", "report_text"],
            [
                {"record_id": "1", "report_text": "EF 55%. RV normal."},
                {"record_id": "2", "report_text": "EF 45%. RV reduced."},
                {"record_id": "3", "report_text": "EF NA. RV normal."},
            ],
        )

        write_csv(
            output_csv,
            [
                "record_id",
                "lvef_percent",
                "rv_function",
                "_status",
                "_error",
                "_processed_at",
                "_model",
                "_raw_response",
            ],
            [
                {
                    "record_id": "2",
                    "lvef_percent": "45",
                    "rv_function": "reduced",
                    "_status": "ok",
                    "_error": "",
                    "_processed_at": "2026-01-01T00:00:00+00:00",
                    "_model": "test-model",
                    "_raw_response": '{"lvef_percent":"45","rv_function":"reduced"}',
                }
            ],
        )

        args = argparse.Namespace(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            id_column="record_id",
            report_column="report_text",
            schema_file=str(self.schema_path),
            llamacpp_base_url="http://127.0.0.1:8080",
            temperature=0.0,
            max_retries=2,
            limit=None,
            resume=True,
            write_raw_response=True,
            model_label="unit-test",
        )

        success_payload = {
            "success": True,
            "content": '{"lvef_percent":"55","rv_function":"normal"}',
            "error": "",
            "model": "llama-test",
        }

        with patch("echo_extract.call_llamacpp_with_retries", return_value=success_payload):
            stats = echo_extract.run_extraction(args)

        self.assertEqual(stats["processed"], 2)
        self.assertEqual(stats["skipped"], 1)

        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 3)
        ids = [row["record_id"] for row in rows]
        self.assertEqual(ids.count("2"), 1)

    def test_writes_one_row_per_input_including_errors(self):
        input_csv = self.base / "input.csv"
        output_csv = self.base / "output.csv"

        write_csv(
            input_csv,
            ["record_id", "report_text"],
            [
                {"record_id": "1", "report_text": "EF 60%. RV normal."},
                {"record_id": "2", "report_text": "Unreadable report."},
            ],
        )

        args = argparse.Namespace(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            id_column="record_id",
            report_column="report_text",
            schema_file=str(self.schema_path),
            llamacpp_base_url="http://127.0.0.1:8080",
            temperature=0.0,
            max_retries=2,
            limit=None,
            resume=False,
            write_raw_response=True,
            model_label="unit-test",
        )

        side_effects = [
            {
                "success": True,
                "content": '{"lvef_percent":"60","rv_function":"normal"}',
                "error": "",
                "model": "llama-ok",
            },
            {
                "success": False,
                "content": "",
                "error": "connection refused",
                "model": "llama-fail",
            },
        ]

        with patch("echo_extract.call_llamacpp_with_retries", side_effect=side_effects):
            stats = echo_extract.run_extraction(args)

        self.assertEqual(stats["processed"], 2)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(stats["llm_error"], 1)

        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        row_by_id = {row["record_id"]: row for row in rows}
        self.assertEqual(row_by_id["1"]["_status"], "ok")
        self.assertEqual(row_by_id["2"]["_status"], "llm_error")
        self.assertEqual(row_by_id["2"]["lvef_percent"], "NA")
        self.assertEqual(row_by_id["2"]["rv_function"], "NA")


if __name__ == "__main__":
    unittest.main()
