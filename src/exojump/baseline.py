"""Session-level feature baseline for leakage-aware model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import IMU_CHANNELS, MOVEMENT_LABELS, PRESSURE_CHANNELS, SEMG_CHANNELS
from .dataset import discover_aligned_sessions
from .metrics import classification_metrics


def _channel_features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    features: list[float] = []
    for channel in values.T:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            features.extend([np.nan] * 8)
            continue
        differences = np.diff(finite)
        features.extend(
            [
                float(np.mean(finite)),
                float(np.std(finite)),
                float(np.sqrt(np.mean(np.square(finite)))),
                float(np.percentile(finite, 25)),
                float(np.median(finite)),
                float(np.percentile(finite, 75)),
                float(np.max(finite) - np.min(finite)),
                float(np.mean(np.abs(differences))) if differences.size else 0.0,
            ]
        )
    return np.asarray(features, dtype=np.float64)


def extract_session_features(aligned_root: str | Path) -> dict[str, np.ndarray]:
    """Extract robust summary features from every complete aligned recording."""

    imu_rows: list[np.ndarray] = []
    semg_rows: list[np.ndarray] = []
    pressure_rows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    sessions: list[str] = []
    for subject, movement, session, imu_path, semg_path in discover_aligned_sessions(aligned_root):
        imu = pd.read_csv(imu_path, usecols=lambda column: column in IMU_CHANNELS)
        semg = pd.read_csv(
            semg_path,
            usecols=lambda column: column in {*SEMG_CHANNELS, *PRESSURE_CHANNELS, "count_foot"},
        )
        if set(IMU_CHANNELS) - set(imu.columns) or set(SEMG_CHANNELS) - set(semg.columns):
            continue
        imu_rows.append(_channel_features(imu, IMU_CHANNELS))
        semg_rows.append(_channel_features(semg, SEMG_CHANNELS))
        if "sum_foot" not in semg.columns and "count_foot" in semg.columns:
            semg["sum_foot"] = semg["count_foot"]
        pressure_rows.append(_channel_features(semg, PRESSURE_CHANNELS))
        labels.append(MOVEMENT_LABELS[movement])
        subjects.append(subject)
        sessions.append(session)
    if not labels:
        raise ValueError("No complete aligned sessions were found")
    subject_codes = {
        subject: f"P{index:02d}" for index, subject in enumerate(sorted(set(subjects)), start=1)
    }
    return {
        "X_imu": np.stack(imu_rows),
        "X_semg": np.stack(semg_rows),
        "X_pressure": np.stack(pressure_rows),
        "y": np.asarray(labels, dtype=np.int64),
        "subject": np.asarray([subject_codes[subject] for subject in subjects]),
        "session": np.asarray(sessions),
    }


def evaluate_lopo_baseline(aligned_root: str | Path, output_path: str | Path) -> dict[str, object]:
    """Evaluate logistic regression with leave-one-participant-out folds."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    arrays = extract_session_features(aligned_root)
    subjects = arrays["subject"].astype(str)
    labels = arrays["y"]
    results: dict[str, object] = {
        "design": "leave-one-participant-out; one prediction per recording",
        "participants": int(len(np.unique(subjects))),
        "sessions": int(len(labels)),
        "models": {},
    }
    for modality, features in {
        "imu": arrays["X_imu"],
        "semg": arrays["X_semg"],
        "pressure": arrays["X_pressure"],
        "fusion": np.concatenate((arrays["X_imu"], arrays["X_semg"]), axis=1),
        "all_modalities": np.concatenate(
            (arrays["X_imu"], arrays["X_semg"], arrays["X_pressure"]), axis=1
        ),
    }.items():
        predictions = np.empty_like(labels)
        folds = []
        for held_out in sorted(np.unique(subjects)):
            test_mask = subjects == held_out
            train_mask = ~test_mask
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
            )
            estimator.fit(features[train_mask], labels[train_mask])
            fold_predictions = estimator.predict(features[test_mask])
            predictions[test_mask] = fold_predictions
            folds.append(
                {
                    "held_out_participant": held_out,
                    "train_sessions": int(train_mask.sum()),
                    "test_sessions": int(test_mask.sum()),
                    **classification_metrics(labels[test_mask], fold_predictions),
                }
            )
        results["models"][modality] = {**classification_metrics(labels, predictions), "folds": folds}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
