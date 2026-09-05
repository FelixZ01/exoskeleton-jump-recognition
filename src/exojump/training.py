"""Participant-held-out training for the dual-branch baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .dataset import participant_split
from .metrics import aggregate_session_probabilities, classification_metrics
from .model import build_model


def _normalise(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "train_global":
        mean = train.mean(axis=(0, 1), keepdims=True)
        std = train.std(axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1.0
        return (train - mean) / std, (validation - mean) / std, (test - mean) / std
    if method == "per_window":
        outputs = []
        for values in (train, validation, test):
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0
            outputs.append((values - mean) / std)
        return tuple(outputs)
    raise ValueError("normalisation must be 'per_window' or 'train_global'")


def train(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_subject: str | None = None,
    validation_subject: str | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    normalisation: str = "per_window",
    patience: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arrays = np.load(dataset_path)
    candidate_train_mask, test_mask, held_out = participant_split(arrays["subject"], test_subject)
    training_subjects = sorted(np.unique(arrays["subject"][candidate_train_mask].astype(str)))
    if len(training_subjects) < 2:
        raise ValueError("At least three participants are required for train/validation/test splits")
    validation_subject = validation_subject or training_subjects[-1]
    if validation_subject not in training_subjects:
        raise ValueError(f"Validation participant must be one of {training_subjects}")
    validation_mask = arrays["subject"].astype(str) == validation_subject
    train_mask = candidate_train_mask & ~validation_mask

    x_imu_train, x_imu_validation, x_imu_test = _normalise(
        arrays["X_imu"][train_mask],
        arrays["X_imu"][validation_mask],
        arrays["X_imu"][test_mask],
        normalisation,
    )
    x_semg_train, x_semg_validation, x_semg_test = _normalise(
        arrays["X_semg"][train_mask],
        arrays["X_semg"][validation_mask],
        arrays["X_semg"][test_mask],
        normalisation,
    )
    y_train = arrays["y"][train_mask]
    y_validation = arrays["y"][validation_mask]
    y_test = arrays["y"][test_mask]

    train_data = TensorDataset(
        torch.tensor(x_imu_train, dtype=torch.float32),
        torch.tensor(x_semg_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    validation_data = TensorDataset(
        torch.tensor(x_imu_validation, dtype=torch.float32),
        torch.tensor(x_semg_validation, dtype=torch.float32),
        torch.tensor(y_validation, dtype=torch.long),
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    model = build_model(x_imu_train.shape[-1], x_semg_train.shape[-1], 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * np.maximum(class_counts, 1))
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for imu_batch, semg_batch, label_batch in loader:
            optimiser.zero_grad()
            loss = loss_function(model(imu_batch, semg_batch), label_batch)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for imu_batch, semg_batch, label_batch in validation_loader:
                validation_losses.append(loss_function(model(imu_batch, semg_batch), label_batch).item())
        training_loss = float(np.mean(losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(x_imu_test, dtype=torch.float32),
            torch.tensor(x_semg_test, dtype=torch.float32),
        )
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
    window_metrics = classification_metrics(y_test, predictions)
    session_true, session_pred, session_keys = aggregate_session_probabilities(
        probabilities,
        y_test,
        arrays["subject"][test_mask],
        arrays["session"][test_mask],
    )
    session_metrics = classification_metrics(session_true, session_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    metrics = {
        "held_out_subject": held_out,
        "validation_subject": validation_subject,
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "window_metrics": window_metrics,
        "session_metrics": session_metrics,
        "test_sessions": len(session_keys),
        "training_class_counts": class_counts.tolist(),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "normalisation": normalisation,
        "history": history,
        "seed": seed,
        "warning": "Pilot result from one held-out participant; use all six outer folds for reporting.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
