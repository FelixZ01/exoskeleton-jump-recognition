#!/usr/bin/env python3
"""Retrain the best IMU model after removing each three-axis sensor node."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from exojump.deep_benchmark import run_deep_benchmark


SENSOR_BLOCKS = {
    "R1": (0, 1, 2),
    "R2": (3, 4, 5),
    "R3": (6, 7, 8),
    "R4": (9, 10, 11),
}


def remove_sensor(arrays: dict[str, np.ndarray], sensor: str) -> dict[str, np.ndarray]:
    """Return a dataset copy with one roll/pitch/yaw sensor block removed."""

    keep = [index for index in range(arrays["X_imu"].shape[-1]) if index not in SENSOR_BLOCKS[sensor]]
    result = {key: np.asarray(value) for key, value in arrays.items()}
    result["X_imu"] = result["X_imu"][:, :, keep]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--baseline-balanced-accuracy", type=float, default=0.9088717163278567)
    args = parser.parse_args()

    source = np.load(args.dataset, allow_pickle=False)
    arrays = {key: source[key] for key in source.files}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = (42,) if args.quick else (42, 7, 123)
    epochs = 3 if args.quick else 30
    patience = 2 if args.quick else 5
    bootstrap = 200 if args.quick else 2000
    summary = {
        "status": "complete",
        "design": "leave-one-IMU-node-out retraining ablation",
        "architecture": "tcn",
        "seeds": list(seeds),
        "epochs": epochs,
        "full_sensor_baseline_balanced_accuracy": args.baseline_balanced_accuracy,
        "ablations": {},
    }

    for sensor in SENSOR_BLOCKS:
        reduced_path = output / f"without_{sensor}.npz"
        np.savez_compressed(reduced_path, **remove_sensor(arrays, sensor))
        result = run_deep_benchmark(
            reduced_path,
            output / f"without_{sensor}",
            architectures=("tcn",),
            modalities=("imu",),
            seeds=seeds,
            epochs=epochs,
            patience=patience,
            device=args.device,
            bootstrap_resamples=bootstrap,
        )
        experiment = result["experiments"]["tcn:imu"]
        balanced = experiment["balanced_accuracy_mean_across_seeds"]
        summary["ablations"][sensor] = {
            "removed_channels": [f"{sensor}_Roll", f"{sensor}_Pitch", f"{sensor}_Yaw"],
            "remaining_imu_channels": 9,
            "balanced_accuracy_mean": balanced,
            "balanced_accuracy_std": experiment["balanced_accuracy_std_across_seeds"],
            "macro_f1_mean": experiment["macro_f1_mean_across_seeds"],
            "macro_f1_std": experiment["macro_f1_std_across_seeds"],
            "balanced_accuracy_drop": args.baseline_balanced_accuracy - balanced,
        }
        reduced_path.unlink()

    (output / "sensor_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    archive = shutil.make_archive(str(output.parent / "sensor_ablation_results"), "zip", output)
    print(json.dumps(summary, indent=2))
    print(f"RUN_COMPLETE: {archive}")


if __name__ == "__main__":
    main()
