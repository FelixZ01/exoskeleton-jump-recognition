#!/usr/bin/env python3
"""Interpolate missing joint-angle values in a directory tree."""

import argparse
import csv
import shutil
from pathlib import Path

from exojump.imputation import impute_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Maximum consecutive values to fill")
    parser.add_argument("--fill-edges", action="store_true", help="Allow edge extrapolation")
    args = parser.parse_args()
    source_root = Path(args.input)
    output_root = Path(args.output)
    rows = []
    for source in sorted(source_root.rglob("*.csv")):
        destination = output_root / source.relative_to(source_root)
        if source.name == "sEMG.csv":
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows.append({"file": source.relative_to(source_root).as_posix(), "status": "copied"})
            continue
        if source.name != "IMU.csv" and not source.name.startswith("IMU_"):
            continue
        try:
            metrics = impute_file(source, destination, limit=args.limit, fill_edges=args.fill_edges)
            rows.append({"file": source.relative_to(source_root).as_posix(), "status": "ok", **metrics})
        except Exception as exc:
            rows.append({"file": source.relative_to(source_root).as_posix(), "status": "error", "error": str(exc)})
    output_root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["file", "status"]
    with (output_root / "imputation_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imputed {sum(row['status'] == 'ok' for row in rows)} IMU files; copied {sum(row['status'] == 'copied' for row in rows)} sEMG files")


if __name__ == "__main__":
    main()
