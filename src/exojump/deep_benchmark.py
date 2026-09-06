"""Repeated nested participant-held-out evaluation for neural architectures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import classification_metrics, grouped_bootstrap_intervals
from .model import ARCHITECTURES, MODALITIES
from .training import train


def _session_arrays(folds: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = []
    predictions = []
    groups = []
    for fold in folds:
        for record in fold["session_predictions"]:
            true_labels.append(record["true_label"])
            predictions.append(record["predicted_label"])
            groups.append(record["subject"])
    return (
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(groups),
    )


def run_deep_benchmark(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    architectures: tuple[str, ...] = ("cnn", "bilstm", "tcn"),
    modalities: tuple[str, ...] = ("fusion",),
    seeds: tuple[int, ...] = (42, 7, 123),
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    normalisation: str = "per_window",
    patience: int = 5,
    device: str = "auto",
    bootstrap_resamples: int = 2000,
) -> dict[str, object]:
    """Run identical folds for each architecture, modality, and random seed."""

    unknown_architectures = sorted(set(architectures) - set(ARCHITECTURES))
    unknown_modalities = sorted(set(modalities) - set(MODALITIES))
    if unknown_architectures:
        raise ValueError(
            f"Unknown architectures: {unknown_architectures}; choose from {ARCHITECTURES}"
        )
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {unknown_modalities}; choose from {MODALITIES}")
    if not seeds:
        raise ValueError("At least one random seed is required")

    arrays = np.load(dataset_path, allow_pickle=False)
    subjects = sorted(np.unique(arrays["subject"].astype(str)))
    if len(subjects) < 3:
        raise ValueError("At least three participants are required")
    output_dir = Path(output_dir)
    experiments: dict[str, object] = {}

    for architecture in architectures:
        for modality in modalities:
            seed_runs = []
            for seed in seeds:
                folds = []
                for index, held_out in enumerate(subjects):
                    candidates = [subject for subject in subjects if subject != held_out]
                    validation_subject = candidates[index % len(candidates)]
                    fold_dir = (
                        output_dir
                        / architecture
                        / modality
                        / f"seed_{seed}"
                        / f"fold_{held_out}"
                    )
                    metrics = train(
                        dataset_path,
                        fold_dir,
                        test_subject=held_out,
                        validation_subject=validation_subject,
                        architecture=architecture,
                        modality=modality,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        normalisation=normalisation,
                        patience=patience,
                        seed=seed,
                        device=device,
                    )
                    folds.append(metrics)
                    print(
                        f"{architecture}/{modality}/seed={seed}/{held_out}: "
                        f"balanced_accuracy="
                        f"{metrics['session_metrics']['balanced_accuracy']:.3f}",
                        flush=True,
                    )
                true_labels, predictions, groups = _session_arrays(folds)
                fold_balanced = np.asarray(
                    [fold["session_metrics"]["balanced_accuracy"] for fold in folds]
                )
                fold_macro_f1 = np.asarray(
                    [fold["session_metrics"]["macro_f1"] for fold in folds]
                )
                seed_runs.append(
                    {
                        "seed": seed,
                        "pooled_session_metrics": classification_metrics(
                            true_labels, predictions
                        ),
                        "participant_clustered_95ci": grouped_bootstrap_intervals(
                            true_labels,
                            predictions,
                            groups,
                            resamples=bootstrap_resamples,
                            seed=seed,
                        ),
                        "fold_balanced_accuracy_mean": float(fold_balanced.mean()),
                        "fold_balanced_accuracy_std": float(fold_balanced.std(ddof=1)),
                        "fold_macro_f1_mean": float(fold_macro_f1.mean()),
                        "fold_macro_f1_std": float(fold_macro_f1.std(ddof=1)),
                        "total_training_seconds": float(
                            sum(fold["training_seconds"] for fold in folds)
                        ),
                        "parameter_count": int(folds[0]["parameter_count"]),
                        "folds": folds,
                    }
                )
            balanced = np.asarray(
                [run["fold_balanced_accuracy_mean"] for run in seed_runs]
            )
            macro_f1 = np.asarray([run["fold_macro_f1_mean"] for run in seed_runs])
            experiments[f"{architecture}:{modality}"] = {
                "architecture": architecture,
                "modality": modality,
                "seed_count": len(seeds),
                "balanced_accuracy_mean_across_seeds": float(balanced.mean()),
                "balanced_accuracy_std_across_seeds": float(
                    balanced.std(ddof=1) if len(balanced) > 1 else 0.0
                ),
                "macro_f1_mean_across_seeds": float(macro_f1.mean()),
                "macro_f1_std_across_seeds": float(
                    macro_f1.std(ddof=1) if len(macro_f1) > 1 else 0.0
                ),
                "seed_runs": seed_runs,
            }

    results = {
        "design": "nested participant-held-out architecture and modality benchmark",
        "dataset": str(dataset_path),
        "participants": len(subjects),
        "architectures": list(architectures),
        "modalities": list(modalities),
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "normalisation": normalisation,
        "patience": patience,
        "device_requested": device,
        "experiments": experiments,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deep_benchmark.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results
