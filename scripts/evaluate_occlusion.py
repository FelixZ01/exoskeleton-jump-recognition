#!/usr/bin/env python3
"""Explain one trained fold with channel and temporal occlusion."""

import argparse
import json
from pathlib import Path

import numpy as np

from exojump.interpretability import occlusion_importance
from exojump.model import build_model
from exojump.training import prepare_fold, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-segments", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    import torch

    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(args.dataset, allow_pickle=False)
    fold = prepare_fold(
        arrays,
        test_subject=metrics["held_out_subject"],
        validation_subject=metrics["validation_subject"],
        normalisation=metrics["normalisation"],
    )
    device = resolve_device(args.device)
    model = build_model(
        fold["x_imu_train"].shape[-1],
        fold["x_semg_train"].shape[-1],
        2,
        architecture=metrics["architecture"],
        modality=metrics["modality"],
    ).to(device)
    state = torch.load(model_dir / "model_state_dict.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_mask = fold["test_mask"]
    result = {
        "architecture": metrics["architecture"],
        "modality": metrics["modality"],
        "held_out_subject": metrics["held_out_subject"],
        **occlusion_importance(
            model,
            fold["x_imu_test"],
            fold["x_semg_test"],
            fold["y_test"],
            arrays["subject"][test_mask],
            arrays["session"][test_mask],
            modality=metrics["modality"],
            batch_size=args.batch_size,
            device=device,
            time_segments=args.time_segments,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
