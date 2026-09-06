"""Participant-held-out training shared by CNN, BiLSTM, and TCN models."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .dataset import participant_split
from .metrics import aggregate_session_outputs, classification_metrics
from .model import ARCHITECTURES, MODALITIES, build_model


def normalise_fold(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise without using validation or test statistics in global mode."""

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


# Backwards compatibility for code that imported the previous private helper.
_normalise = normalise_fold


def prepare_fold(
    arrays,
    *,
    test_subject: str | None,
    validation_subject: str | None,
    normalisation: str,
) -> dict[str, object]:
    """Create leakage-safe masks and normalised tensors for one nested fold."""

    candidate_train_mask, test_mask, held_out = participant_split(
        arrays["subject"], test_subject
    )
    subjects = arrays["subject"].astype(str)
    training_subjects = sorted(np.unique(subjects[candidate_train_mask]))
    if len(training_subjects) < 2:
        raise ValueError(
            "At least three participants are required for train/validation/test splits"
        )
    validation_subject = validation_subject or training_subjects[-1]
    if validation_subject not in training_subjects:
        raise ValueError(f"Validation participant must be one of {training_subjects}")
    validation_mask = subjects == validation_subject
    train_mask = candidate_train_mask & ~validation_mask

    imu = normalise_fold(
        arrays["X_imu"][train_mask],
        arrays["X_imu"][validation_mask],
        arrays["X_imu"][test_mask],
        normalisation,
    )
    semg = normalise_fold(
        arrays["X_semg"][train_mask],
        arrays["X_semg"][validation_mask],
        arrays["X_semg"][test_mask],
        normalisation,
    )
    return {
        "held_out_subject": held_out,
        "validation_subject": validation_subject,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "x_imu_train": imu[0],
        "x_imu_validation": imu[1],
        "x_imu_test": imu[2],
        "x_semg_train": semg[0],
        "x_semg_validation": semg[1],
        "x_semg_test": semg[2],
        "y_train": arrays["y"][train_mask],
        "y_validation": arrays["y"][validation_mask],
        "y_test": arrays["y"][test_mask],
    }


def resolve_device(requested: str):
    """Resolve auto/cpu/cuda/mps lazily after torch is installed."""

    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def predict_probabilities(
    model,
    x_imu: np.ndarray,
    x_semg: np.ndarray,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    """Run batched inference and return class probabilities on the CPU."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.tensor(x_imu, dtype=torch.float32),
        torch.tensor(x_semg, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for imu_batch, semg_batch in loader:
            logits = model(imu_batch.to(device), semg_batch.to(device))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_subject: str | None = None,
    validation_subject: str | None = None,
    architecture: str = "cnn",
    modality: str = "fusion",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    normalisation: str = "per_window",
    patience: int = 5,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    """Train and evaluate one nested participant-held-out fold."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    architecture = architecture.lower()
    modality = modality.lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = resolve_device(device)

    arrays = np.load(dataset_path, allow_pickle=False)
    fold = prepare_fold(
        arrays,
        test_subject=test_subject,
        validation_subject=validation_subject,
        normalisation=normalisation,
    )
    y_train = fold["y_train"]
    y_validation = fold["y_validation"]
    y_test = fold["y_test"]

    train_data = TensorDataset(
        torch.tensor(fold["x_imu_train"], dtype=torch.float32),
        torch.tensor(fold["x_semg_train"], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    validation_data = TensorDataset(
        torch.tensor(fold["x_imu_validation"], dtype=torch.float32),
        torch.tensor(fold["x_semg_validation"], dtype=torch.float32),
        torch.tensor(y_validation, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    model = build_model(
        fold["x_imu_train"].shape[-1],
        fold["x_semg_train"].shape[-1],
        2,
        architecture=architecture,
        modality=modality,
    ).to(resolved_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * np.maximum(class_counts, 1))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=resolved_device)
    )
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for imu_batch, semg_batch, label_batch in loader:
            optimiser.zero_grad()
            logits = model(imu_batch.to(resolved_device), semg_batch.to(resolved_device))
            loss = loss_function(logits, label_batch.to(resolved_device))
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for imu_batch, semg_batch, label_batch in validation_loader:
                logits = model(imu_batch.to(resolved_device), semg_batch.to(resolved_device))
                validation_losses.append(
                    loss_function(logits, label_batch.to(resolved_device)).item()
                )
        training_loss = float(np.mean(losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    training_seconds = time.perf_counter() - training_started

    if best_state is not None:
        model.load_state_dict(best_state)

    inference_started = time.perf_counter()
    probabilities = predict_probabilities(
        model,
        fold["x_imu_test"],
        fold["x_semg_test"],
        batch_size=batch_size,
        device=resolved_device,
    )
    inference_seconds = time.perf_counter() - inference_started
    predictions = probabilities.argmax(axis=1)
    window_metrics = classification_metrics(y_test, predictions)
    test_mask = fold["test_mask"]
    session_true, session_pred, session_probabilities, session_keys = aggregate_session_outputs(
        probabilities,
        y_test,
        arrays["subject"][test_mask],
        arrays["session"][test_mask],
    )
    session_metrics = classification_metrics(session_true, session_pred)
    session_predictions = []
    for actual, predicted, probability, key in zip(
        session_true,
        session_pred,
        session_probabilities,
        session_keys,
        strict=True,
    ):
        subject, session = key.split("/", 1)
        session_predictions.append(
            {
                "subject": subject,
                "session": session,
                "true_label": int(actual),
                "predicted_label": int(predicted),
                "probability_vertical_jump": float(probability[0]),
                "probability_long_jump": float(probability[1]),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), output_dir / "model_state_dict.pt")
    metrics = {
        "architecture": architecture,
        "modality": modality,
        "held_out_subject": fold["held_out_subject"],
        "validation_subject": fold["validation_subject"],
        "train_windows": int(fold["train_mask"].sum()),
        "validation_windows": int(fold["validation_mask"].sum()),
        "test_windows": int(test_mask.sum()),
        "window_metrics": window_metrics,
        "session_metrics": session_metrics,
        "session_predictions": session_predictions,
        "test_sessions": len(session_keys),
        "training_class_counts": class_counts.tolist(),
        "parameter_count": int(parameter_count),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "device": str(resolved_device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "normalisation": normalisation,
        "history": history,
        "seed": seed,
        "warning": "One outer fold only; use every participant fold for reporting.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
