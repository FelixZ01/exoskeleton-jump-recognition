#!/usr/bin/env python3
"""Run repeated CNN, BiLSTM, and TCN participant-held-out experiments."""

import argparse
import json

from exojump.deep_benchmark import run_deep_benchmark
from exojump.model import ARCHITECTURES, MODALITIES


def _csv_strings(text: str) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in text.split(",") if value.strip())


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--architectures",
        default="cnn,bilstm,tcn",
        help=f"Comma-separated subset of: {','.join(ARCHITECTURES)}",
    )
    parser.add_argument(
        "--modalities",
        default="fusion",
        help=f"Comma-separated subset of: {','.join(MODALITIES)}",
    )
    parser.add_argument("--seeds", default="42,7,123")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--normalisation",
        choices=("per_window", "train_global"),
        default="per_window",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    results = run_deep_benchmark(
        args.dataset,
        args.output,
        architectures=_csv_strings(args.architectures),
        modalities=_csv_strings(args.modalities),
        seeds=_csv_ints(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        normalisation=args.normalisation,
        patience=args.patience,
        device=args.device,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    summary = {
        key: {
            "balanced_accuracy": value["balanced_accuracy_mean_across_seeds"],
            "macro_f1": value["macro_f1_mean_across_seeds"],
        }
        for key, value in results["experiments"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
