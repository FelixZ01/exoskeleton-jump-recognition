"""Occlusion-based interpretation for trained multimodal sequence models."""

from __future__ import annotations

import numpy as np

from .constants import IMU_CHANNELS, SEMG_CHANNELS
from .metrics import aggregate_session_outputs, classification_metrics
from .model import MODALITIES
from .training import predict_probabilities


def _session_metrics(
    model,
    x_imu: np.ndarray,
    x_semg: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    *,
    batch_size: int,
    device,
) -> dict[str, object]:
    probabilities = predict_probabilities(
        model,
        x_imu,
        x_semg,
        batch_size=batch_size,
        device=device,
    )
    actual, predicted, _, _ = aggregate_session_outputs(
        probabilities, labels, subjects, sessions
    )
    return classification_metrics(actual, predicted)


def occlusion_importance(
    model,
    x_imu: np.ndarray,
    x_semg: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    *,
    modality: str,
    batch_size: int,
    device,
    time_segments: int = 8,
) -> dict[str, object]:
    """Measure performance change after zeroing channels or time regions.

    Inputs should already be normalised. Zero therefore represents the reference
    level rather than an arbitrary raw sensor value. Positive performance drops
    indicate that the removed channel or time segment supported classification.
    """

    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    if time_segments < 2:
        raise ValueError("time_segments must be at least two")

    baseline = _session_metrics(
        model,
        x_imu,
        x_semg,
        labels,
        subjects,
        sessions,
        batch_size=batch_size,
        device=device,
    )
    channel_results = []
    channel_groups = []
    if modality in {"imu", "fusion"}:
        channel_groups.append(("imu", x_imu, IMU_CHANNELS))
    if modality in {"semg", "fusion"}:
        channel_groups.append(("semg", x_semg, SEMG_CHANNELS))

    for source, values, configured_names in channel_groups:
        names = (
            configured_names
            if len(configured_names) == values.shape[2]
            else [f"{source}_channel_{index + 1}" for index in range(values.shape[2])]
        )
        for index, name in enumerate(names):
            occluded_imu = x_imu.copy() if source == "imu" else x_imu
            occluded_semg = x_semg.copy() if source == "semg" else x_semg
            target = occluded_imu if source == "imu" else occluded_semg
            target[:, :, index] = 0.0
            metrics = _session_metrics(
                model,
                occluded_imu,
                occluded_semg,
                labels,
                subjects,
                sessions,
                batch_size=batch_size,
                device=device,
            )
            channel_results.append(
                {
                    "source": source,
                    "channel": name,
                    "balanced_accuracy_after_occlusion": metrics["balanced_accuracy"],
                    "balanced_accuracy_drop": baseline["balanced_accuracy"]
                    - metrics["balanced_accuracy"],
                    "macro_f1_after_occlusion": metrics["macro_f1"],
                    "macro_f1_drop": baseline["macro_f1"] - metrics["macro_f1"],
                }
            )
    channel_results.sort(key=lambda item: item["balanced_accuracy_drop"], reverse=True)

    boundaries = np.linspace(0, x_imu.shape[1], time_segments + 1, dtype=int)
    time_results = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        occluded_imu = x_imu.copy()
        occluded_semg = x_semg.copy()
        if modality in {"imu", "fusion"}:
            occluded_imu[:, start:stop, :] = 0.0
        if modality in {"semg", "fusion"}:
            occluded_semg[:, start:stop, :] = 0.0
        metrics = _session_metrics(
            model,
            occluded_imu,
            occluded_semg,
            labels,
            subjects,
            sessions,
            batch_size=batch_size,
            device=device,
        )
        time_results.append(
            {
                "segment": index + 1,
                "start_sample": int(start),
                "stop_sample": int(stop),
                "balanced_accuracy_after_occlusion": metrics["balanced_accuracy"],
                "balanced_accuracy_drop": baseline["balanced_accuracy"]
                - metrics["balanced_accuracy"],
                "macro_f1_after_occlusion": metrics["macro_f1"],
                "macro_f1_drop": baseline["macro_f1"] - metrics["macro_f1"],
            }
        )
    time_results.sort(key=lambda item: item["balanced_accuracy_drop"], reverse=True)
    return {
        "method": "zero occlusion on normalised test windows",
        "baseline_session_metrics": baseline,
        "channel_importance": channel_results,
        "time_segment_importance": time_results,
        "interpretation": (
            "A positive drop indicates that the occluded region supported the model. "
            "A negative drop means performance improved after removal and may indicate noise."
        ),
    }
