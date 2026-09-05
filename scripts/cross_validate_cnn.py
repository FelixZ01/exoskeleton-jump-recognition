#!/usr/bin/env python3
"""Run nested participant-held-out evaluation for every participant."""

import argparse
import json
from pathlib import Path

import numpy as np

from exojump.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--normalisation", choices=("per_window", "train_global"), default="per_window")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects = sorted(np.unique(np.load(args.dataset)["subject"].astype(str)))
    if len(subjects) < 3:
        parser.error("At least three participants are required")
    output_root = Path(args.output)
    folds = []
    for index, held_out in enumerate(subjects):
        candidates = [subject for subject in subjects if subject != held_out]
        validation_subject = candidates[index % len(candidates)]
        metrics = train(
            args.dataset,
            output_root / f"fold_{held_out}",
            test_subject=held_out,
            validation_subject=validation_subject,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            normalisation=args.normalisation,
            patience=args.patience,
            seed=args.seed,
        )
        folds.append(metrics)
        print(
            f"{held_out}: session balanced accuracy="
            f"{metrics['session_metrics']['balanced_accuracy']:.3f}"
        )

    balanced = np.asarray([fold["session_metrics"]["balanced_accuracy"] for fold in folds])
    macro_f1 = np.asarray([fold["session_metrics"]["macro_f1"] for fold in folds])
    summary = {
        "design": "nested participant-held-out evaluation",
        "participants": len(subjects),
        "session_balanced_accuracy_mean": float(balanced.mean()),
        "session_balanced_accuracy_std": float(balanced.std(ddof=1)),
        "session_macro_f1_mean": float(macro_f1.mean()),
        "session_macro_f1_std": float(macro_f1.std(ddof=1)),
        "folds": folds,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cross_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
