#!/usr/bin/env python3
"""Validate grouped rating CSV used by hierarchical SSR tuning."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate grouped ratings CSV (group,text,label).")
    p.add_argument("--csv", required=True, help="Path to grouped CSV file.")
    p.add_argument("--group-col", default="group")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--min-group-n", type=int, default=300, help="Warn if group has fewer rows.")
    p.add_argument("--sample-per-group", type=int, default=1200, help="Target sample per group.")
    p.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero exit code also for warnings.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    required = [args.group_col, args.text_col, args.label_col]
    rows_total = 0
    rows_valid = 0
    invalid_rows = 0
    missing_rows = 0
    bad_label_rows = 0
    empty_text_rows = 0
    group_sizes: Counter[str] = Counter()
    label_dist: Counter[int] = Counter()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("ERROR: CSV has no header.")
        missing_cols = [c for c in required if c not in reader.fieldnames]
        if missing_cols:
            raise SystemExit(
                "ERROR: missing required columns: "
                + ", ".join(missing_cols)
                + f" | found: {reader.fieldnames}"
            )

        for row in reader:
            rows_total += 1
            group = str(row.get(args.group_col, "")).strip()
            text = str(row.get(args.text_col, "")).strip()
            label_raw = str(row.get(args.label_col, "")).strip()

            if not group or not text or not label_raw:
                invalid_rows += 1
                missing_rows += 1
                if not text:
                    empty_text_rows += 1
                continue

            try:
                label = int(round(float(label_raw)))
            except Exception:
                invalid_rows += 1
                bad_label_rows += 1
                continue

            if label < 1 or label > 5:
                invalid_rows += 1
                bad_label_rows += 1
                continue

            rows_valid += 1
            group_sizes[group] += 1
            label_dist[label] += 1

    if rows_total == 0:
        raise SystemExit("ERROR: CSV has header but no data rows.")

    warnings: list[str] = []
    errors: list[str] = []
    if rows_valid == 0:
        errors.append("No valid rows after validation.")
    if len(group_sizes) == 0:
        errors.append("No valid groups detected.")

    under_min = {g: n for g, n in group_sizes.items() if n < args.min_group_n}
    under_sample = {g: n for g, n in group_sizes.items() if n < args.sample_per_group}
    if under_min:
        warnings.append(
            f"{len(under_min)} groups below min-group-n={args.min_group_n}"
        )
    if under_sample:
        warnings.append(
            f"{len(under_sample)} groups below sample-per-group={args.sample_per_group}"
        )

    print("=== Grouped CSV Validation ===")
    print(f"File: {path}")
    print(f"Rows total: {rows_total}")
    print(f"Rows valid: {rows_valid}")
    print(f"Rows invalid: {invalid_rows}")
    print(f"- Missing required fields: {missing_rows}")
    print(f"- Invalid labels (not 1..5): {bad_label_rows}")
    print(f"- Empty text rows: {empty_text_rows}")
    print(f"Groups valid: {len(group_sizes)}")
    print("Label distribution (valid rows):")
    for label in range(1, 6):
        print(f"- {label}: {label_dist.get(label, 0)}")

    print("\nTop groups by size:")
    for group, n in sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"- {group}: {n}")

    if under_min:
        print(f"\nGroups below min-group-n={args.min_group_n}:")
        for group, n in sorted(under_min.items(), key=lambda x: x[1]):
            print(f"- {group}: {n}")

    if under_sample:
        print(f"\nGroups below sample-per-group={args.sample_per_group}:")
        for group, n in sorted(under_sample.items(), key=lambda x: x[1]):
            print(f"- {group}: {n}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(2)

    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
        if args.fail_on_warning:
            raise SystemExit(1)

    print("\nValidation status: OK")


if __name__ == "__main__":
    main()
