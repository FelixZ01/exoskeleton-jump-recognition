#!/usr/bin/env python3
"""Summarise active slots in packed IMU acquisition files."""

import argparse
import json
from pathlib import Path

from exojump.sensor_audit import audit_imu_slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--sample-rows", type=int, default=500)
    args = parser.parse_args()
    result = audit_imu_slots(args.input, args.sample_rows)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
