#!/usr/bin/env python3
"""Convert benchmark JSON files into a compact Markdown comparison table."""

import argparse
import json
from pathlib import Path


def _percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", help="Path to classical_benchmark.json")
    parser.add_argument("--deep", help="Path to deep_benchmark.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.classical and not args.deep:
        parser.error("Provide --classical, --deep, or both")

    rows = []
    if args.classical:
        classical = json.loads(Path(args.classical).read_text(encoding="utf-8"))
        for experiment in classical["experiments"].values():
            metrics = experiment["pooled_session_metrics"]
            interval = experiment["participant_clustered_95ci"]["balanced_accuracy"]
            rows.append(
                (
                    experiment["model"],
                    experiment["modality"],
                    "fixed",
                    _percentage(metrics["balanced_accuracy"]),
                    _percentage(metrics["macro_f1"]),
                    f"{_percentage(interval['lower'])}–{_percentage(interval['upper'])}",
                )
            )
    if args.deep:
        deep = json.loads(Path(args.deep).read_text(encoding="utf-8"))
        for experiment in deep["experiments"].values():
            balanced_mean = experiment["balanced_accuracy_mean_across_seeds"]
            balanced_std = experiment["balanced_accuracy_std_across_seeds"]
            macro_mean = experiment["macro_f1_mean_across_seeds"]
            macro_std = experiment["macro_f1_std_across_seeds"]
            intervals = [
                run["participant_clustered_95ci"]["balanced_accuracy"]
                for run in experiment["seed_runs"]
            ]
            rows.append(
                (
                    experiment["architecture"],
                    experiment["modality"],
                    str(experiment["seed_count"]),
                    f"{_percentage(balanced_mean)} ± {_percentage(balanced_std)}",
                    f"{_percentage(macro_mean)} ± {_percentage(macro_std)}",
                    f"{_percentage(min(item['lower'] for item in intervals))}–"
                    f"{_percentage(max(item['upper'] for item in intervals))}",
                )
            )

    lines = [
        "# Model benchmark summary",
        "",
        "All values are recording-level results with complete participants held out.",
        "Confidence intervals use participant-clustered resampling.",
        "",
        "| Model | Modality | Seeds | Balanced accuracy | Macro-F1 | 95% CI (balanced accuracy) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    lines.extend(
        [
            "",
            "> Compare models only when they use the same dataset version and "
            "event-window definition.",
        ]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} model rows to {output_path}")


if __name__ == "__main__":
    main()
