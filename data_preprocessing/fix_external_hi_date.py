#!/usr/bin/env python3
"""
Manually fix erroneous assay dates in an external HI titre table.

Example:
    python fix_external_hi_date.py \
        --input external_hi_titre_table.csv \
        --output external_hi_titre_table_datefixed.csv \
        --old-date 2003-02-15 \
        --new-date 2013-02-15

In-place update with backup:
    python fix_external_hi_date.py \
        --input external_hi_titre_table.csv \
        --old-date 2003-02-15 \
        --new-date 2013-02-15 \
        --in-place \
        --backup

Dry-run:
    python fix_external_hi_date.py \
        --input external_hi_titre_table.csv \
        --old-date 2003-02-15 \
        --new-date 2013-02-15 \
        --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def validate_iso_date(value: str, arg_name: str) -> str:
    """
    Validate YYYY-MM-DD date strings and return the normalized ISO date string.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{arg_name} must be in YYYY-MM-DD format, but got: {value!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one date value with another in the date column of an external "
            "HI titre table CSV."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input external HI titre table CSV.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output CSV path. If omitted, '<input_stem>_datefixed.csv' is used. "
            "Cannot be used together with --in-place."
        ),
    )
    parser.add_argument(
        "--old-date",
        required=True,
        help="Erroneous date to replace, in YYYY-MM-DD format. Example: 2003-02-15",
    )
    parser.add_argument(
        "--new-date",
        required=True,
        help="Correct date to write, in YYYY-MM-DD format. Example: 2013-02-15",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Name of the date column to edit. Default: date",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a separate output file.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help=(
            "When used with --in-place, create '<input>.bak' before overwriting. "
            "Ignored unless --in-place is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be changed, but do not write a file.",
    )

    args = parser.parse_args()

    args.old_date = validate_iso_date(args.old_date, "--old-date")
    args.new_date = validate_iso_date(args.new_date, "--new-date")

    if args.in_place and args.output:
        parser.error("--output cannot be used together with --in-place.")

    return args


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_datefixed{input_path.suffix}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            f"This script expects a CSV file. Got: {input_path.name}"
        )

    df = pd.read_csv(input_path, dtype={args.date_column: "string"})

    if args.date_column not in df.columns:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Date column {args.date_column!r} was not found. "
            f"Available columns: {available}"
        )

    # Normalize date column as string, while preserving missing values.
    date_series = df[args.date_column].astype("string").str.strip()
    mask = date_series == args.old_date
    n_changed = int(mask.sum())

    print(f"Input: {input_path}")
    print(f"Date column: {args.date_column}")
    print(f"Old date: {args.old_date}")
    print(f"New date: {args.new_date}")
    print(f"Rows to update: {n_changed}")

    if args.dry_run:
        print("Dry-run mode: no file was written.")
        return

    if n_changed == 0:
        print("Warning: no rows matched the old date. Output will be identical except for CSV formatting.")

    df.loc[mask, args.date_column] = args.new_date

    if args.in_place:
        output_path = input_path
        if args.backup:
            backup_path = input_path.with_suffix(input_path.suffix + ".bak")
            shutil.copy2(input_path, backup_path)
            print(f"Backup written: {backup_path}")
    else:
        output_path = Path(args.output) if args.output else default_output_path(input_path)

    df.to_csv(output_path, index=False)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
