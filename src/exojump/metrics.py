"""Small dependency-free classification metrics used by training scripts."""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Return binary accuracy, balanced accuracy, macro F1, and confusion matrix."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with matching shapes")

    matrix = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Only binary labels 0 and 1 are supported")
        matrix[actual, predicted] += 1

    recalls: list[float | None] = []
    f1_scores: list[float | None] = []
    for label in (0, 1):
        true_positive = int(matrix[label, label])
        false_negative = int(matrix[label, :].sum() - true_positive)
        false_positive = int(matrix[:, label].sum() - true_positive)
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recalls.append(recall)
        if recall is None:
            f1_scores.append(None)
        elif precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    valid_recalls = [value for value in recalls if value is not None]
    valid_f1 = [value for value in f1_scores if value is not None]
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(valid_recalls)),
        "macro_f1": float(np.mean(valid_f1)),
        "per_class_recall": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def aggregate_session_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average overlapping-window probabilities before scoring each recording."""

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects).astype(str)
    sessions = np.asarray(sessions).astype(str)
    keys = np.asarray([f"{subject}/{session}" for subject, session in zip(subjects, sessions, strict=True)])
    true_labels: list[int] = []
    predictions: list[int] = []
    ordered_keys: list[str] = []
    for key in sorted(np.unique(keys)):
        mask = keys == key
        session_labels = np.unique(labels[mask])
        if len(session_labels) != 1:
            raise ValueError(f"Session {key!r} contains conflicting labels")
        true_labels.append(int(session_labels[0]))
        predictions.append(int(probabilities[mask].mean(axis=0).argmax()))
        ordered_keys.append(key)
    return np.asarray(true_labels), np.asarray(predictions), ordered_keys
