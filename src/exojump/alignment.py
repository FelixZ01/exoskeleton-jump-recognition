"""Pair and align normalised sEMG and IMU recordings on a common time grid."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


@dataclass(frozen=True)
class SessionPair:
    subject: str
    movement: str
    session: str
    imu_path: Path
    semg_path: Path


def _session_key(path: Path) -> str | None:
    match = SESSION_RE.search(path.name)
    return match.group(1) if match else None


def discover_pairs(imu_root: str | Path, semg_root: str | Path) -> list[SessionPair]:
    """Match files by recording timestamp and infer subject/movement from paths."""

    imu_root = Path(imu_root)
    semg_root = Path(semg_root)
    semg_by_session: dict[str, list[Path]] = {}
    for path in semg_root.rglob("processed_data_*.csv"):
        key = _session_key(path)
        if key:
            semg_by_session.setdefault(key, []).append(path)

    pairs: list[SessionPair] = []
    for imu_path in sorted(imu_root.rglob("IMU_*.csv")):
        session = _session_key(imu_path)
        if not session:
            continue
        candidates = semg_by_session.get(session, [])
        if len(candidates) != 1:
            continue
        joined = "/".join(imu_path.parts).lower()
        movement = "tiaogao" if "tiaogao" in joined else "tiaoyuan" if "tiaoyuan" in joined else "unknown"
        subject_match = re.search(r"(\d{2}_[A-Za-z]+)_IMU_data", joined, re.IGNORECASE)
        subject = subject_match.group(1).upper() if subject_match else "unknown"
        pairs.append(SessionPair(subject, movement, session, imu_path, candidates[0]))
    return pairs


def _prepare(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["__time"] = pd.to_datetime(result[timestamp_column], errors="coerce").dt.round("ms")
    result = result.dropna(subset=["__time"]).sort_values("__time")
    numeric = [column for column in result.columns if column not in {timestamp_column, "Timestamp", "__time"}]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    # Duplicate millisecond samples are averaged to make the reindex operation deterministic.
    aggregations = {column: "mean" for column in numeric}
    if "Timestamp" in result.columns and timestamp_column != "Timestamp":
        aggregations["Timestamp"] = "first"
    return result.groupby("__time", as_index=True).agg(aggregations)


def align_frames(
    imu: pd.DataFrame,
    semg: pd.DataFrame,
    *,
    interval_ms: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Crop two modalities to their overlap and reindex to a shared timeline."""

    imu_time = "First_Timestamp_ms" if "First_Timestamp_ms" in imu.columns else "Timestamp"
    if "Timestamp" not in semg.columns:
        raise ValueError("sEMG frame has no Timestamp column")
    imu_work = _prepare(imu, imu_time)
    semg_work = _prepare(semg, "Timestamp")
    if imu_work.empty or semg_work.empty:
        raise ValueError("One modality has no valid timestamps")

    start = max(imu_work.index.min(), semg_work.index.min())
    end = min(imu_work.index.max(), semg_work.index.max())
    if start > end:
        raise ValueError("The recordings do not overlap")
    timeline = pd.date_range(start=start, end=end, freq=f"{interval_ms}ms")
    aligned_imu = imu_work.reindex(timeline)
    aligned_semg = semg_work.reindex(timeline)
    timestamp_text = timeline.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    aligned_imu.insert(0, "First_Timestamp_ms", timestamp_text)
    aligned_semg.insert(0, "Timestamp", timestamp_text)
    aligned_imu.index = pd.RangeIndex(len(aligned_imu))
    aligned_semg.index = pd.RangeIndex(len(aligned_semg))
    metrics = {
        "start": str(start),
        "end": str(end),
        "rows": len(timeline),
        "imu_missing_values": int(aligned_imu.isna().sum().sum()),
        "semg_missing_values": int(aligned_semg.isna().sum().sum()),
    }
    return aligned_imu, aligned_semg, metrics


def align_pair(pair: SessionPair, output_root: str | Path) -> dict[str, object]:
    """Align one pair and write it into a stable, anonymisation-ready layout."""

    output_root = Path(output_root)
    imu = pd.read_csv(pair.imu_path)
    semg = pd.read_csv(pair.semg_path)
    aligned_imu, aligned_semg, metrics = align_frames(imu, semg)
    destination = output_root / pair.subject / pair.movement / pair.session
    destination.mkdir(parents=True, exist_ok=True)
    aligned_imu.to_csv(destination / "IMU.csv", index=False)
    aligned_semg.to_csv(destination / "sEMG.csv", index=False)
    return {"subject": pair.subject, "movement": pair.movement, "session": pair.session, **metrics}
