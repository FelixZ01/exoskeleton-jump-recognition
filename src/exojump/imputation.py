"""Missing-value handling for the 12 IMU orientation channels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import IMU_CHANNELS


def impute_angles(
    frame: pd.DataFrame,
    *,
    limit: int | None = None,
    fill_edges: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Linearly interpolate internal gaps and return quality counts.

    Edge gaps remain missing by default because extrapolation has weaker support.
    Set ``fill_edges`` only when that assumption is acceptable for the analysis.
    """

    result = frame.copy()
    channels = [column for column in IMU_CHANNELS if column in result.columns]
    if not channels:
        raise ValueError("No recognised IMU angle columns were found")
    result[channels] = result[channels].apply(pd.to_numeric, errors="coerce")
    before = int(result[channels].isna().sum().sum())
    kwargs: dict[str, object] = {"method": "linear", "limit": limit}
    if fill_edges:
        kwargs["limit_direction"] = "both"
    else:
        kwargs["limit_area"] = "inside"
    result[channels] = result[channels].interpolate(**kwargs)
    after = int(result[channels].isna().sum().sum())
    return result, {"missing_before": before, "missing_after": after, "filled": before - after}


def impute_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    fill_edges: bool = False,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_csv(input_path)
    result, metrics = impute_angles(frame, limit=limit, fill_edges=fill_edges)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return metrics
