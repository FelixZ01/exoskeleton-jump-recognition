#!/usr/bin/env python3
"""Export small identity-free signal examples for public documentation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exojump.constants import IMU_CHANNELS, MOVEMENT_NAMES, SEMG_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arrays = np.load(args.dataset, allow_pickle=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "anonymised model-ready windows",
        "sampling_rate_hz": 1000,
        "contains_identifiers": False,
        "contains_video": False,
        "samples": [],
    }
    for label, movement in MOVEMENT_NAMES.items():
        matches = np.flatnonzero(arrays["y"] == label)
        if not len(matches):
            continue
        index = int(matches[0])
        time_ms = np.arange(arrays["X_imu"].shape[1], dtype=np.int64)
        for modality, key, channels in (
            ("imu", "X_imu", IMU_CHANNELS),
            ("semg", "X_semg", SEMG_CHANNELS),
        ):
            filename = f"{movement}_{modality}_example.csv"
            frame = pd.DataFrame(arrays[key][index], columns=channels)
            frame.insert(0, "time_ms", time_ms)
            frame.to_csv(output / filename, index=False, float_format="%.6f")
            manifest["samples"].append(
                {
                    "file": filename,
                    "movement": movement,
                    "modality": modality,
                    "rows": len(frame),
                    "channels": len(channels),
                }
            )
    (output / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(manifest['samples'])} public examples to {output}")


if __name__ == "__main__":
    main()
