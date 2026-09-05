"""Audit active IMU payload slots without exposing participant identifiers."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


def audit_imu_slots(input_root: str | Path, sample_rows: int = 500) -> dict[str, object]:
    """Count payload slots and active slots in each packed IMU recording."""

    input_root = Path(input_root)
    pattern_counts: Counter[tuple[int, tuple[int, ...]]] = Counter()
    payload_counts: Counter[int] = Counter()
    files_scanned = 0
    invalid_files = 0
    for path in sorted(input_root.rglob("IMU_*.csv")):
        active: defaultdict[int, bool] = defaultdict(bool)
        payload_slots: int | None = None
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "Data" not in reader.fieldnames:
                    continue
                for row_index, row in enumerate(reader):
                    if row_index >= sample_rows:
                        break
                    parts = [part.strip() for part in str(row.get("Data", "")).split(",")[1:]]
                    if not parts or len(parts) % 3:
                        continue
                    payload_slots = len(parts) // 3
                    for slot in range(payload_slots):
                        values = [float(value) for value in parts[slot * 3 : (slot + 1) * 3]]
                        active[slot] |= any(value != 0 for value in values)
            if payload_slots is None:
                invalid_files += 1
                continue
            files_scanned += 1
            active_slots = tuple(slot for slot in range(payload_slots) if active[slot])
            payload_counts[payload_slots] += 1
            pattern_counts[(payload_slots, active_slots)] += 1
        except (OSError, TypeError, ValueError):
            invalid_files += 1

    patterns = [
        {
            "payload_slots": payload_slots,
            "active_slots_zero_based": list(active_slots),
            "active_slot_count": len(active_slots),
            "recordings": count,
        }
        for (payload_slots, active_slots), count in sorted(pattern_counts.items())
    ]
    return {
        "files_scanned": files_scanned,
        "invalid_files": invalid_files,
        "payload_slot_counts": {str(key): value for key, value in sorted(payload_counts.items())},
        "activity_patterns": patterns,
        "sample_rows_per_file": sample_rows,
    }
