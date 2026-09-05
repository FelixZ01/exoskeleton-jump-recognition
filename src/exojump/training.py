"""Participant-held-out training for the dual-branch baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .dataset import participant_split
from .model import build_model


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = (0, 1)
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def train(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_subject: str | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arrays = np.load(dataset_path)
    train_mask, test_mask, held_out = participant_split(arrays["subject"], test_subject)
    x_imu_train, x_imu_test = _standardise(arrays["X_imu"][train_mask], arrays["X_imu"][test_mask])
    x_semg_train, x_semg_test = _standardise(arrays["X_semg"][train_mask], arrays["X_semg"][test_mask])
    y_train, y_test = arrays["y"][train_mask], arrays["y"][test_mask]

    train_data = TensorDataset(
        torch.tensor(x_imu_train, dtype=torch.float32),
        torch.tensor(x_semg_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = build_model(x_imu_train.shape[-1], x_semg_train.shape[-1], 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    history = []
    for _ in range(epochs):
        losses = []
        for imu_batch, semg_batch, label_batch in loader:
            optimiser.zero_grad()
            loss = loss_function(model(imu_batch, semg_batch), label_batch)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(x_imu_test, dtype=torch.float32),
            torch.tensor(x_semg_test, dtype=torch.float32),
        )
        predictions = logits.argmax(dim=1).numpy()
    accuracy = float((predictions == y_test).mean())
    class_recalls = []
    for label in (0, 1):
        mask = y_test == label
        class_recalls.append(float((predictions[mask] == label).mean()) if mask.any() else None)
    valid_recalls = [value for value in class_recalls if value is not None]
    balanced_accuracy = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    metrics = {
        "held_out_subject": held_out,
        "train_windows": int(train_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": class_recalls,
        "epochs": epochs,
        "final_training_loss": history[-1],
        "seed": seed,
        "warning": "Pilot result from one held-out participant; repeat across all participant folds.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
