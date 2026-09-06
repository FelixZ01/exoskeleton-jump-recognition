#!/usr/bin/env python3
"""Benchmark classical and optional MiniROCKET models on event windows."""

import argparse
import json

from exojump.benchmark import CLASSICAL_MODELS, evaluate_window_models
from exojump.model import MODALITIES


def _csv_values(text: str) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models",
        default="logistic,svm,random_forest",
        help=f"Comma-separated subset of: {','.join(CLASSICAL_MODELS)}",
    )
    parser.add_argument(
        "--modalities",
        default="fusion",
        help=f"Comma-separated subset of: {','.join(MODALITIES)}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    results = evaluate_window_models(
        args.dataset,
        args.output,
        models=_csv_values(args.models),
        modalities=_csv_values(args.modalities),
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    summary = {
        key: value["pooled_session_metrics"]
        for key, value in results["experiments"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
