"""Classical and MiniROCKET benchmarks on the shared event-window dataset."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .metrics import (
    aggregate_session_outputs,
    classification_metrics,
    grouped_bootstrap_intervals,
)
from .model import MODALITIES


CLASSICAL_MODELS = ("logistic", "svm", "random_forest", "minirocket")


def select_modality(arrays, modality: str) -> np.ndarray:
    """Return a time-by-channel tensor for one declared modality."""

    modality = modality.lower()
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    if modality == "imu":
        return np.asarray(arrays["X_imu"], dtype=np.float32)
    if modality == "semg":
        return np.asarray(arrays["X_semg"], dtype=np.float32)
    return np.concatenate(
        (
            np.asarray(arrays["X_imu"], dtype=np.float32),
            np.asarray(arrays["X_semg"], dtype=np.float32),
        ),
        axis=2,
    )


def window_summary_features(values: np.ndarray) -> np.ndarray:
    """Extract eight robust features per channel from fixed-length windows."""

    values = np.asarray(values, dtype=np.float64)
    differences = np.diff(values, axis=1)
    features = (
        np.nanmean(values, axis=1),
        np.nanstd(values, axis=1),
        np.sqrt(np.nanmean(np.square(values), axis=1)),
        np.nanpercentile(values, 25, axis=1),
        np.nanmedian(values, axis=1),
        np.nanpercentile(values, 75, axis=1),
        np.nanmax(values, axis=1) - np.nanmin(values, axis=1),
        np.nanmean(np.abs(differences), axis=1),
    )
    return np.concatenate(features, axis=1)


def _score_to_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 2:
        scores = scores[:, 1] - scores[:, 0]
    scores = np.clip(scores, -40, 40)
    positive = 1 / (1 + np.exp(-scores))
    return np.column_stack((1 - positive, positive))


def _fit_predict_standard_model(
    model_name: str,
    train_features: np.ndarray,
    y_train: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if model_name == "logistic":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    elif model_name == "svm":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        )
    elif model_name == "random_forest":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    else:
        raise ValueError(f"Unknown standard model {model_name!r}")
    estimator.fit(train_features, y_train)
    probabilities = estimator.predict_proba(test_features)
    final_estimator = estimator.steps[-1][1]
    parameter_count = int(
        sum(
            np.asarray(value).size
            for value in vars(final_estimator).values()
            if isinstance(value, np.ndarray)
        )
    )
    return probabilities, parameter_count


def _fit_predict_minirocket(
    train_values: np.ndarray,
    y_train: np.ndarray,
    test_values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    # Some managed/Conda environments make site-packages read-only. An explicit
    # writable cache keeps Numba's compiled MiniROCKET kernels portable.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "exojump_numba_cache")
    )
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MiniROCKET requires the optional time-series dependencies. "
            "Install them with: pip install -e '.[time-series]'"
        ) from exc
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    # sktime panel estimators use [instances, channels, time].
    train_panel = np.transpose(train_values, (0, 2, 1))
    test_panel = np.transpose(test_values, (0, 2, 1))
    transform = MiniRocketMultivariate(num_kernels=10_000, random_state=seed)
    try:
        transformed_train = transform.fit_transform(train_panel)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MiniROCKET requires sktime and numba. "
            "Install them with: pip install -e '.[time-series]'"
        ) from exc
    transformed_test = transform.transform(test_panel)
    scaler = StandardScaler(with_mean=False)
    transformed_train = scaler.fit_transform(transformed_train)
    transformed_test = scaler.transform(transformed_test)
    classifier = RidgeClassifier(alpha=1.0, class_weight="balanced")
    classifier.fit(transformed_train, y_train)
    probabilities = _score_to_probabilities(classifier.decision_function(transformed_test))
    parameter_count = int(
        np.asarray(classifier.coef_).size + np.asarray(classifier.intercept_).size
    )
    return probabilities, parameter_count


def evaluate_window_models(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    models: tuple[str, ...] = ("logistic", "svm", "random_forest"),
    modalities: tuple[str, ...] = ("fusion",),
    seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> dict[str, object]:
    """Run fixed-hyperparameter LOPO benchmarks and aggregate predictions by trial."""

    arrays = np.load(dataset_path, allow_pickle=False)
    subjects = arrays["subject"].astype(str)
    labels = arrays["y"].astype(np.int64)
    sessions = arrays["session"].astype(str)
    unknown_models = sorted(set(models) - set(CLASSICAL_MODELS))
    unknown_modalities = sorted(set(modalities) - set(MODALITIES))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}; choose from {CLASSICAL_MODELS}")
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {unknown_modalities}; choose from {MODALITIES}")

    output: dict[str, object] = {
        "design": "leave-one-participant-out with recording-level probability aggregation",
        "dataset": str(dataset_path),
        "participants": int(len(np.unique(subjects))),
        "windows": int(len(labels)),
        "seed": seed,
        "experiments": {},
    }
    for modality in modalities:
        values = select_modality(arrays, modality)
        summary_features = window_summary_features(values)
        for model_name in models:
            folds = []
            all_true: list[np.ndarray] = []
            all_predicted: list[np.ndarray] = []
            all_groups: list[np.ndarray] = []
            all_records: list[dict[str, object]] = []
            parameter_counts = []
            for held_out in sorted(np.unique(subjects)):
                test_mask = subjects == held_out
                train_mask = ~test_mask
                if model_name == "minirocket":
                    probabilities, parameter_count = _fit_predict_minirocket(
                        values[train_mask], labels[train_mask], values[test_mask], seed
                    )
                else:
                    probabilities, parameter_count = _fit_predict_standard_model(
                        model_name,
                        summary_features[train_mask],
                        labels[train_mask],
                        summary_features[test_mask],
                        seed,
                    )
                actual, predicted, mean_probabilities, keys = aggregate_session_outputs(
                    probabilities,
                    labels[test_mask],
                    subjects[test_mask],
                    sessions[test_mask],
                )
                fold_metrics = classification_metrics(actual, predicted)
                folds.append(
                    {
                        "held_out_subject": held_out,
                        "train_windows": int(train_mask.sum()),
                        "test_windows": int(test_mask.sum()),
                        "test_sessions": len(keys),
                        **fold_metrics,
                    }
                )
                all_true.append(actual)
                all_predicted.append(predicted)
                all_groups.append(np.full(len(actual), held_out))
                parameter_counts.append(parameter_count)
                for truth, prediction, probability, key in zip(
                    actual, predicted, mean_probabilities, keys, strict=True
                ):
                    subject, session = key.split("/", 1)
                    all_records.append(
                        {
                            "subject": subject,
                            "session": session,
                            "true_label": int(truth),
                            "predicted_label": int(prediction),
                            "probability_vertical_jump": float(probability[0]),
                            "probability_long_jump": float(probability[1]),
                        }
                    )
            pooled_true = np.concatenate(all_true)
            pooled_predicted = np.concatenate(all_predicted)
            pooled_groups = np.concatenate(all_groups)
            key = f"{model_name}:{modality}"
            output["experiments"][key] = {
                "model": model_name,
                "modality": modality,
                "feature_type": "raw MiniROCKET transform"
                if model_name == "minirocket"
                else "eight summary features per channel",
                "pooled_session_metrics": classification_metrics(
                    pooled_true, pooled_predicted
                ),
                "participant_clustered_95ci": grouped_bootstrap_intervals(
                    pooled_true,
                    pooled_predicted,
                    pooled_groups,
                    resamples=bootstrap_resamples,
                    seed=seed,
                ),
                "parameter_count_by_fold": parameter_counts,
                "folds": folds,
                "session_predictions": all_records,
            }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output
