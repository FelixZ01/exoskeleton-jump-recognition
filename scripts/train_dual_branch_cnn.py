#!/usr/bin/env python3
"""Train a participant-held-out dual-branch CNN baseline."""

import argparse
import json

from exojump.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-subject")
    parser.add_argument("--validation-subject")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--normalisation", choices=("per_window", "train_global"), default="per_window")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(
        args.dataset,
        args.output,
        test_subject=args.test_subject,
        validation_subject=args.validation_subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        normalisation=args.normalisation,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
