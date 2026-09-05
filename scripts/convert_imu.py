#!/usr/bin/env python3
"""Convert a directory of packed IMU exports."""

import argparse
import json
from pathlib import Path

from exojump.imu import convert_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sensor-groups",
        default="4,2,1,3",
        help="Zero-based three-axis payload groups mapped to R1,R2,R3,R4",
    )
    args = parser.parse_args()
    groups = tuple(int(value) for value in args.sensor_groups.split(","))
    if len(groups) != 4:
        parser.error("--sensor-groups must contain exactly four integers")
    results = convert_tree(args.input, args.output, groups)
    report = Path(args.output) / "conversion_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = sum(row["status"] != "ok" for row in results)
    print(f"Converted {len(results) - failures}/{len(results)} files; report: {report}")


if __name__ == "__main__":
    main()
