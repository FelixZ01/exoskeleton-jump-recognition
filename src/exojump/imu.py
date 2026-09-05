"""Decode packed IMU acquisition exports into tabular joint-angle data."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .constants import IMU_CHANNELS


def parse_device_timestamp(value: str) -> int:
    """Convert ``hours:minutes:seconds:milliseconds`` to elapsed milliseconds."""

    parts = value.removeprefix("Time--").split(":")
    if len(parts) != 4:
        raise ValueError(f"Unexpected device timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_payload(payload: str, sensor_groups: Sequence[int] = (4, 2, 1, 3)) -> tuple[int, list[float]]:
    """Parse one packed row and select four three-axis sensor groups.

    The recovered converted files imply the default group order ``(4, 2, 1, 3)``
    from an eight-group payload. The mapping must be checked against hardware notes
    before the channels are given anatomical names.
    """

    parts = [part.strip() for part in str(payload).split(",")]
    if len(parts) < 2:
        raise ValueError("IMU payload has no angle values")
    elapsed_ms = parse_device_timestamp(parts[0])
    values = [float(value) for value in parts[1:]]
    if len(values) % 3:
        raise ValueError(f"Expected angle triplets, found {len(values)} values")
    groups = [values[index : index + 3] for index in range(0, len(values), 3)]
    if max(sensor_groups) >= len(groups):
        raise ValueError(f"Requested group {max(sensor_groups)}, but payload has {len(groups)} groups")
    selected = [number for group_index in sensor_groups for number in groups[group_index]]
    return elapsed_ms, selected


def convert_raw_imu(
    input_path: str | Path,
    output_path: str | Path,
    sensor_groups: Sequence[int] = (4, 2, 1, 3),
) -> dict[str, int]:
    """Convert one raw IMU CSV while preserving device-clock gaps.

    Invalid rows are skipped and counted. A device-clock day rollover is handled by
    adding 24 hours when the clock moves backwards.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted: list[list[object]] = []
    invalid_rows = 0
    anchor_wall_time: datetime | None = None
    anchor_device_ms: int | None = None
    rollover_ms = 0
    previous_device_ms: int | None = None

    with input_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Data" not in reader.fieldnames:
            raise ValueError(f"{input_path} is not a packed IMU export")
        wall_column = "First_Timestamp_ms" if "First_Timestamp_ms" in reader.fieldnames else "Timestamp"

        for row in reader:
            try:
                device_ms, angles = parse_payload(row["Data"], sensor_groups)
                if previous_device_ms is not None and device_ms < previous_device_ms - 12 * 3600 * 1000:
                    rollover_ms += 24 * 3600 * 1000
                absolute_device_ms = device_ms + rollover_ms
                previous_device_ms = device_ms

                if anchor_wall_time is None:
                    anchor_text = (row.get(wall_column) or "").strip()
                    if not anchor_text:
                        raise ValueError("First valid IMU row has no wall-clock timestamp")
                    anchor_wall_time = pd.to_datetime(anchor_text).to_pydatetime()
                    anchor_device_ms = absolute_device_ms

                offset = absolute_device_ms - int(anchor_device_ms)
                wall_time = anchor_wall_time + timedelta(milliseconds=offset)
                converted.append(
                    [wall_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], row["Data"].split(",", 1)[0].removeprefix("Time--"), *angles]
                )
            except (TypeError, ValueError):
                invalid_rows += 1

    frame = pd.DataFrame(converted, columns=["First_Timestamp_ms", "Timestamp", *IMU_CHANNELS])
    frame.to_csv(output_path, index=False)
    return {"converted_rows": len(frame), "invalid_rows": invalid_rows}


def convert_tree(
    input_root: str | Path,
    output_root: str | Path,
    sensor_groups: Sequence[int] = (4, 2, 1, 3),
) -> list[dict[str, object]]:
    """Convert every packed ``IMU_*.csv`` below a directory."""

    input_root = Path(input_root)
    output_root = Path(output_root)
    results: list[dict[str, object]] = []
    for source in sorted(input_root.rglob("IMU_*.csv")):
        destination = output_root / source.relative_to(input_root)
        try:
            counts = convert_raw_imu(source, destination, sensor_groups)
            results.append({"file": str(source.relative_to(input_root)), "status": "ok", **counts})
        except Exception as exc:  # one malformed capture should not abort the batch
            results.append({"file": str(source.relative_to(input_root)), "status": "error", "error": str(exc)})
    return results
