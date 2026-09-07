#!/usr/bin/env python3
"""Compare pressure- and IMU-derived event centres without exposing identities."""

import argparse
import json
from pathlib import Path

import numpy as np


def _session_centres(path: str) -> dict[tuple[str, str], int]:
    arrays = np.load(path, allow_pickle=False)
    result = {}
    for subject, session, center in zip(
        arrays["subject"].astype(str),
        arrays["session"].astype(str),
        arrays["event_center"],
        strict=True,
    ):
        result[(subject, session)] = int(center)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-dataset", required=True)
    parser.add_argument("--imu-dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pressure = _session_centres(args.pressure_dataset)
    imu = _session_centres(args.imu_dataset)
    keys = sorted(set(pressure) & set(imu))
    offsets = np.asarray([imu[key] - pressure[key] for key in keys], dtype=float)
    absolute = np.abs(offsets)
    result = {
        "matched_sessions": len(keys),
        "sampling_rate_hz": 1000,
        "signed_offset_ms": {
            "mean": float(np.mean(offsets)),
            "median": float(np.median(offsets)),
            "std": float(np.std(offsets)),
        },
        "absolute_offset_ms": {
            "mean": float(np.mean(absolute)),
            "median": float(np.median(absolute)),
            "p90": float(np.percentile(absolute, 90)),
        },
        "agreement_fraction": {
            "within_100_ms": float(np.mean(absolute <= 100)),
            "within_250_ms": float(np.mean(absolute <= 250)),
            "within_500_ms": float(np.mean(absolute <= 500)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
