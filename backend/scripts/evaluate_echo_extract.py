#!/usr/bin/env python3
"""Evaluate echo extraction predictions against a gold-standard CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate echo extraction accuracy.")
    parser.add_argument("--pred-csv", required=True, help="Predicted output CSV")
    parser.add_argument("--gold-csv", required=True, help="Gold-standard CSV")
    parser.add_argument(
        "--output-summary-csv",
        required=True,
        help="Path to summary CSV with per-feature accuracy",
    )
    parser.add_argument("--id-column", default="StudyID", help="ID column used for matching")
    parser.add_argument(
        "--schema-file",
        default="",
        help="Optional schema JSON to define feature columns and order",
    )
    return parser.parse_args(argv)


def read_csv_as_dict(path: Path, id_column: str) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise ValueError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if id_column not in fieldnames:
            raise ValueError(f"CSV {path} has no ID column '{id_column}'. Found: {fieldnames}")

        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get(id_column, "")).strip()
            if not row_id:
                continue
            rows[row_id] = {k: str(v) if v is not None else "" for k, v in row.items()}
    return rows


def get_features(
    pred_rows: Dict[str, Dict[str, str]],
    gold_rows: Dict[str, Dict[str, str]],
    id_column: str,
    schema_file: str,
) -> List[str]:
    if schema_file:
        schema_path = Path(schema_file)
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        features = payload.get("features", [])
        names = [str(item.get("name", "")).strip() for item in features if isinstance(item, dict)]
        names = [name for name in names if name]
        if not names:
            raise ValueError("Schema has no valid feature names.")
        return names

    pred_cols = set()
    gold_cols = set()

    if pred_rows:
        pred_cols = set(next(iter(pred_rows.values())).keys())
    if gold_rows:
        gold_cols = set(next(iter(gold_rows.values())).keys())

    common = pred_cols.intersection(gold_cols)
    excluded = {id_column}
    excluded.update({col for col in common if col.startswith("_")})
    features = sorted(common - excluded)
    if not features:
        raise ValueError(
            "No shared feature columns found. Provide --schema-file to specify feature names."
        )
    return features


def exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def evaluate(
    pred_rows: Dict[str, Dict[str, str]],
    gold_rows: Dict[str, Dict[str, str]],
    features: List[str],
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    matched_ids = sorted(set(pred_rows).intersection(set(gold_rows)))
    if not matched_ids:
        raise ValueError("No overlapping IDs between predicted and gold CSV files.")

    per_feature_exact = {feature: 0 for feature in features}
    per_feature_total = {feature: 0 for feature in features}
    row_all_correct = 0

    for row_id in matched_ids:
        pred = pred_rows[row_id]
        gold = gold_rows[row_id]

        all_correct = True
        for feature in features:
            pred_val = str(pred.get(feature, ""))
            gold_val = str(gold.get(feature, ""))
            per_feature_total[feature] += 1
            if exact_match(pred_val, gold_val):
                per_feature_exact[feature] += 1
            else:
                all_correct = False

        if all_correct:
            row_all_correct += 1

    summary_rows: List[Dict[str, str]] = []
    for feature in features:
        total = per_feature_total[feature]
        exact = per_feature_exact[feature]
        acc = (exact / total) if total else 0.0
        summary_rows.append(
            {
                "feature": feature,
                "total_compared": str(total),
                "exact_matches": str(exact),
                "accuracy": f"{acc:.6f}",
            }
        )

    row_level_rate = row_all_correct / len(matched_ids)
    overall = {
        "matched_rows": float(len(matched_ids)),
        "row_level_all_features_correct_rate": row_level_rate,
    }
    return summary_rows, overall


def write_summary(path: Path, summary_rows: List[Dict[str, str]], overall: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "total_compared", "exact_matches", "accuracy"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
        writer.writerow(
            {
                "feature": "__row_level_all_features_correct__",
                "total_compared": str(int(overall["matched_rows"])),
                "exact_matches": str(
                    int(round(overall["matched_rows"] * overall["row_level_all_features_correct_rate"]))
                ),
                "accuracy": f"{overall['row_level_all_features_correct_rate']:.6f}",
            }
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pred_rows = read_csv_as_dict(Path(args.pred_csv), args.id_column)
    gold_rows = read_csv_as_dict(Path(args.gold_csv), args.id_column)
    features = get_features(pred_rows, gold_rows, args.id_column, args.schema_file)
    summary_rows, overall = evaluate(pred_rows, gold_rows, features)
    write_summary(Path(args.output_summary_csv), summary_rows, overall)

    print(f"Matched rows: {int(overall['matched_rows'])}")
    print(f"Features evaluated: {len(features)}")
    print(
        "Row-level all-features-correct rate: "
        f"{overall['row_level_all_features_correct_rate']:.4f}"
    )
    for row in summary_rows:
        print(
            f"{row['feature']}: {row['exact_matches']}/{row['total_compared']} "
            f"({float(row['accuracy']):.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
