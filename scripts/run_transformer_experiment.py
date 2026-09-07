#!/usr/bin/env python3
"""Run regularised Transformer baselines on full and compact sensor inputs."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from exojump.deep_benchmark import run_deep_benchmark


def _run(dataset, output, modalities, seeds, epochs, patience, device, bootstrap):
    return run_deep_benchmark(
        dataset,
        output,
        architectures=("transformer",),
        modalities=modalities,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=device,
        bootstrap_resamples=bootstrap,
    )


def _metrics(experiment):
    folds = [fold for run in experiment["seed_runs"] for fold in run["folds"]]
    return {
        "balanced_accuracy_mean": experiment["balanced_accuracy_mean_across_seeds"],
        "balanced_accuracy_std": experiment["balanced_accuracy_std_across_seeds"],
        "macro_f1_mean": experiment["macro_f1_mean_across_seeds"],
        "macro_f1_std": experiment["macro_f1_std_across_seeds"],
        "parameter_count": experiment["seed_runs"][0]["parameter_count"],
        "mean_best_epoch": float(np.mean([fold["best_epoch"] for fold in folds])),
        "mean_epochs_completed": float(
            np.mean([fold["epochs_completed"] for fold in folds])
        ),
    }


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

    full = _run(
        args.dataset,
        output / "full_inputs",
        ("imu", "fusion"),
        seeds,
        epochs,
        patience,
        args.device,
        bootstrap,
    )
    source = np.load(args.dataset, allow_pickle=False)
    arrays = {key: source[key] for key in source.files}
    arrays["X_imu"] = arrays["X_imu"][:, :, 6:12]
    compact_dataset = output / "R3_R4.npz"
    np.savez_compressed(compact_dataset, **arrays)
    compact = _run(
        compact_dataset,
        output / "R3_R4",
        ("imu",),
        seeds,
        epochs,
        patience,
        args.device,
        bootstrap,
    )
    compact_dataset.unlink()

    summary = {
        "status": "complete",
        "design": "regularised time-series Transformer comparison",
        "seeds": list(seeds),
        "epochs": epochs,
        "regularisation": {
            "encoder_layers": 2,
            "attention_heads": 4,
            "dropout": 0.4,
            "weight_decay": 1e-4,
            "early_stopping_patience": patience,
            "token_stride": 8,
            "participant_held_out": True,
        },
        "experiments": {
            "transformer_imu": _metrics(full["experiments"]["transformer:imu"]),
            "transformer_fusion": _metrics(full["experiments"]["transformer:fusion"]),
            "transformer_R3_R4": _metrics(compact["experiments"]["transformer:imu"]),
        },
    }
    (output / "transformer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    archive = shutil.make_archive(str(output.parent / "transformer_results"), "zip", output)
    print(json.dumps(summary, indent=2))
    print(f"RUN_COMPLETE: {archive}")


if __name__ == "__main__":
    main()
