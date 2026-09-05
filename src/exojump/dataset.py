"""Build model-ready, participant-aware windows from aligned session pairs."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .constants import IMU_CHANNELS, MOVEMENT_LABELS, SEMG_CHANNELS


SESSION_RE = re.compile(r"(20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
SUBJECT_RE = re.compile(r"(\d{2}_[A-Za-z]+)_(?:IMU|sEMG)_data", re.IGNORECASE)


def _recovered_key(path: Path) -> tuple[str, str, str] | None:
    """Recover participant, movement, and session from the archived directory layout."""

    text = path.as_posix()
    subject_match = SUBJECT_RE.search(text)
    session_match = SESSION_RE.search(text)
    lowered = text.lower()
    movement = "tiaogao" if "tiaogao" in lowered else "tiaoyuan" if "tiaoyuan" in lowered else ""
    if not subject_match or not session_match or not movement:
        return None
    return subject_match.group(1).upper(), movement, session_match.group(1)


def discover_aligned_sessions(root: str | Path) -> list[tuple[str, str, str, Path, Path]]:
    """Find aligned IMU/sEMG pairs in canonical or recovered archive layouts."""

    root = Path(root)
    sessions = []
    for imu_path in sorted(root.rglob("IMU.csv")):
        semg_path = imu_path.with_name("sEMG.csv")
        if not semg_path.exists():
            continue
        relative = imu_path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        subject, movement, session = relative.parts[-4:-1]
        if movement not in MOVEMENT_LABELS:
            continue
        sessions.append((subject, movement, session, imu_path, semg_path))
    if sessions:
        return sessions

    # Recovered archive layout:
    # aligned_IMU/<participant>_IMU_data/.../<session>/IMU_<session>.csv
    # aligned_sEMG/<participant>_sEMG_data/.../<session>/processed_data_<session>.csv
    imu_root = root / "aligned_IMU"
    semg_root = root / "aligned_sEMG"
    if not imu_root.is_dir() or not semg_root.is_dir():
        return []

    semg_by_key: dict[tuple[str, str, str], Path] = {}
    for semg_path in sorted(semg_root.rglob("*.csv")):
        key = _recovered_key(semg_path)
        if key is not None and key not in semg_by_key:
            semg_by_key[key] = semg_path

    for imu_path in sorted(imu_root.rglob("IMU_*.csv")):
        key = _recovered_key(imu_path)
        if key is None or key not in semg_by_key:
            continue
        subject, movement, session = key
        sessions.append((subject, movement, session, imu_path, semg_by_key[key]))
    return sessions


def build_windows(
    aligned_root: str | Path,
    *,
    window_size: int = 1000,
    stride: int = 500,
    max_missing_fraction: float = 0.05,
) -> dict[str, np.ndarray]:
    """Create fixed windows; reject windows with excessive missing values."""

    imu_windows: list[np.ndarray] = []
    semg_windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    sessions_out: list[str] = []

    for subject, movement, session, imu_path, semg_path in discover_aligned_sessions(aligned_root):
        imu = pd.read_csv(imu_path)
        semg = pd.read_csv(semg_path)
        missing_imu = [column for column in IMU_CHANNELS if column not in imu.columns]
        missing_semg = [column for column in SEMG_CHANNELS if column not in semg.columns]
        if missing_imu or missing_semg:
            continue
        length = min(len(imu), len(semg))
        imu_values = imu.loc[: length - 1, IMU_CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        semg_values = semg.loc[: length - 1, SEMG_CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        for start in range(0, max(0, length - window_size + 1), stride):
            stop = start + window_size
            imu_window = imu_values[start:stop]
            semg_window = semg_values[start:stop]
            missing_fraction = (np.isnan(imu_window).sum() + np.isnan(semg_window).sum()) / (imu_window.size + semg_window.size)
            if missing_fraction > max_missing_fraction:
                continue
            # Remaining short gaps use within-window linear interpolation.
            imu_window = pd.DataFrame(imu_window).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
            semg_window = pd.DataFrame(semg_window).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
            imu_windows.append(imu_window)
            semg_windows.append(semg_window)
            labels.append(MOVEMENT_LABELS[movement])
            subjects.append(subject)
            sessions_out.append(session)

    if not imu_windows:
        raise ValueError("No eligible aligned windows were found")
    subject_codes = {
        subject: f"P{index:02d}" for index, subject in enumerate(sorted(set(subjects)), start=1)
    }
    return {
        "X_imu": np.stack(imu_windows),
        "X_semg": np.stack(semg_windows),
        "y": np.asarray(labels, dtype=np.int64),
        "subject": np.asarray([subject_codes[subject] for subject in subjects]),
        "session": np.asarray(sessions_out),
    }


def save_windows(output_path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def participant_split(subjects: np.ndarray, test_subject: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return train/test masks with an entire participant held out."""

    unique = sorted(str(subject) for subject in np.unique(subjects))
    if len(unique) < 2:
        raise ValueError("At least two participants are required for a held-out split")
    test_subject = test_subject or unique[-1]
    if test_subject not in unique:
        raise ValueError(f"Unknown test participant {test_subject!r}; choose from {unique}")
    test_mask = subjects.astype(str) == test_subject
    return ~test_mask, test_mask, test_subject
