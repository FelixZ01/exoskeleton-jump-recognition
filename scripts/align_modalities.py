#!/usr/bin/env python3
"""Discover and align normalised IMU/sEMG session pairs."""

import argparse
import json
from pathlib import Path

from exojump.alignment import align_pair, discover_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", required=True, help="Normalised IMU root")
    parser.add_argument("--semg", required=True, help="Timestamp-normalised sEMG root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = discover_pairs(args.imu, args.semg)
    results = []
    for pair in pairs:
        try:
            results.append({"status": "ok", **align_pair(pair, args.output)})
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "subject": pair.subject,
                    "movement": pair.movement,
                    "session": pair.session,
                    "error": str(exc),
                }
            )
    report = Path(args.output) / "alignment_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = sum(row["status"] != "ok" for row in results)
    print(f"Aligned {len(results) - failures}/{len(results)} pairs; report: {report}")


if __name__ == "__main__":
    main()
