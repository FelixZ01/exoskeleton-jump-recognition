#!/usr/bin/env python3
"""Run the reportable IMU-only event-detection benchmark in one command."""

import argparse
import json
from pathlib import Path
import shutil

from exojump.deep_benchmark import run_deep_benchmark


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = (42,) if args.quick else (42, 7, 123)
    epochs = 3 if args.quick else 30
    patience = 2 if args.quick else 5
    bootstrap = 200 if args.quick else 2000
    results = run_deep_benchmark(
        args.dataset,
        output / "deep_models",
        architectures=("cnn", "tcn"),
        modalities=("imu",),
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=args.device,
        bootstrap_resamples=bootstrap,
    )

    compact = {
        "status": "complete",
        "event_detector": "imu_angular_velocity_energy",
        "input_modality": "imu_only",
        "seeds": list(seeds),
        "epochs": epochs,
        "models": {},
    }
    rows = []
    for key, experiment in results["experiments"].items():
        item = {
            "balanced_accuracy_mean": experiment["balanced_accuracy_mean_across_seeds"],
            "balanced_accuracy_std": experiment["balanced_accuracy_std_across_seeds"],
            "macro_f1_mean": experiment["macro_f1_mean_across_seeds"],
            "macro_f1_std": experiment["macro_f1_std_across_seeds"],
        }
        compact["models"][key] = item
        rows.append(
            f"| {experiment['architecture'].upper()} | IMU | "
            f"{_pct(item['balanced_accuracy_mean'])} ± {_pct(item['balanced_accuracy_std'])} | "
            f"{_pct(item['macro_f1_mean'])} ± {_pct(item['macro_f1_std'])} |"
        )
    (output / "imu_event_metrics.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# IMU-only event detection benchmark",
        "",
        "Jump windows were located from IMU angular dynamics without plantar pressure or sEMG.",
        "Evaluation holds out complete participants and aggregates predictions by recording.",
        "",
        "| Model | Input | Balanced accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
        *rows,
        "",
    ]
    (output / "IMU_EVENT_BENCHMARK.md").write_text("\n".join(report), encoding="utf-8")
    (output / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    archive = shutil.make_archive(str(output.parent / "imu_event_results"), "zip", output)
    print(json.dumps(compact, indent=2))
    print(f"RUN_COMPLETE: {archive}")


if __name__ == "__main__":
    main()
