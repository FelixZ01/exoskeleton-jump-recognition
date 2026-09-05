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


def pressure_event_center(semg: pd.DataFrame, smooth_samples: int = 51) -> int:
    """Locate the flight phase from the minimum smoothed plantar-pressure signal."""

    if "sum_foot" in semg.columns:
        pressure = pd.to_numeric(semg["sum_foot"], errors="coerce")
    elif "count_foot" in semg.columns:
        pressure = pd.to_numeric(semg["count_foot"], errors="coerce")
    else:
        pressure_columns = [
            column for column in semg.columns if column.startswith("FP_CH4") or column.startswith("FP_CH5")
        ]
        if not pressure_columns:
            raise ValueError("No plantar-pressure channels were found")
        pressure = semg[pressure_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    pressure = pressure.interpolate(limit_direction="both")
    if pressure.isna().all() or len(pressure) < 3:
        raise ValueError("Plantar-pressure signal is empty")
    window = max(3, min(smooth_samples, len(pressure) // 5))
    smoothed = pressure.rolling(window=window, center=True, min_periods=1).median().to_numpy()
    margin = min(len(smoothed) // 10, 500)
    start, stop = margin, len(smoothed) - margin
    if stop <= start:
        start, stop = 0, len(smoothed)
    return int(start + np.nanargmin(smoothed[start:stop]))


def _window_starts(
    length: int,
    window_size: int,
    stride: int,
    mode: str,
    event_center: int | None,
    event_offsets: tuple[int, ...],
) -> list[int]:
    if mode == "sliding":
        return list(range(0, max(0, length - window_size + 1), stride))
    if mode != "pressure_event" or event_center is None:
        raise ValueError("window_mode must be 'sliding' or 'pressure_event'")
    base = event_center - window_size // 2
    starts = sorted({base + offset for offset in event_offsets})
    valid = [start for start in starts if start >= 0 and start + window_size <= length]
    if valid:
        return valid
    return [max(0, min(base, length - window_size))] if length >= window_size else []


def build_windows(
    aligned_root: str | Path,
    *,
    window_size: int = 1000,
    stride: int = 500,
    max_missing_fraction: float = 0.05,
    window_mode: str = "sliding",
    event_offsets: tuple[int, ...] = (-250, 0, 250),
) -> dict[str, np.ndarray]:
    """Create fixed windows; reject windows with excessive missing values."""

    imu_windows: list[np.ndarray] = []
    semg_windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    sessions_out: list[str] = []
    window_starts: list[int] = []
    event_centers: list[int] = []

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
        event_center = pressure_event_center(semg) if window_mode == "pressure_event" else None
        starts = _window_starts(
            length,
            window_size,
            stride,
            window_mode,
            event_center,
            event_offsets,
        )
        for start in starts:
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
            window_starts.append(start)
            event_centers.append(event_center if event_center is not None else -1)

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
        "window_start": np.asarray(window_starts, dtype=np.int64),
        "event_center": np.asarray(event_centers, dtype=np.int64),
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
