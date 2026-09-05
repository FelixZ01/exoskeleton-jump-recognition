"""Privacy-aware inventory generation for the recovered experiment archive."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from .constants import VIDEO_SUFFIXES


SUBJECT_RE = re.compile(r"(?<!\d)(\d{2}_[A-Za-z]+)(?![A-Za-z])")
SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def _layer(relative: Path) -> str:
    text = relative.as_posix()
    if text.startswith("dataset_all/") or text.startswith("zzf/") or text.startswith("ZZF_20251021/"):
        return "raw_or_device_processed"
    if "/aligned_data_strict/" in f"/{text}" or "/imputed_joint_angles/" in f"/{text}":
        return "processed"
    if "/cnn_optimal_data/" in f"/{text}":
        return "model_candidate"
    if text.startswith("data_process/"):
        return "interim"
    return "documentation_or_exploration"


def _subject_token(relative: Path) -> str:
    match = SUBJECT_RE.search(relative.as_posix())
    return match.group(1).upper() if match else ""


def _session_token(relative: Path) -> str:
    match = SESSION_RE.search(relative.as_posix())
    return match.group(1) if match else ""


def inventory(root: str | Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    extension_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    layer_bytes: Counter[str] = Counter()
    subjects: set[str] = set()
    movements: Counter[str] = Counter()
    videos = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._") or "__MACOSX" in path.parts or path.name == ".DS_Store":
            continue
        relative = path.relative_to(root)
        suffix = path.suffix.lower() or "[no extension]"
        is_video = suffix in VIDEO_SUFFIXES
        subject = _subject_token(relative)
        movement = "tiaogao" if "tiaogao" in relative.as_posix().lower() else "tiaoyuan" if "tiaoyuan" in relative.as_posix().lower() else ""
        layer = _layer(relative)
        if subject:
            subjects.add(subject)
        if movement:
            movements[movement] += 1
        if is_video:
            videos += 1
        extension_counts[suffix] += 1
        layer_counts[layer] += 1
        layer_bytes[layer] += path.stat().st_size
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "suffix": suffix,
                "layer": layer,
                "subject": subject,
                "movement": movement,
                "session": _session_token(relative),
                "excluded_video": int(is_video),
            }
        )

    summary = {
        "root_name": root.name,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "subject_count": len(subjects),
        "subjects_are_deidentified_in_public_outputs": True,
        "video_count": videos,
        "extension_counts": dict(sorted(extension_counts.items())),
        "layer_counts": dict(sorted(layer_counts.items())),
        "layer_bytes": dict(sorted(layer_bytes.items())),
        "movement_file_counts": dict(sorted(movements.items())),
    }
    return rows, summary


def write_inventory(
    root: str | Path,
    private_output: str | Path,
    public_output: str | Path | None = None,
) -> dict[str, object]:
    """Write a detailed private inventory and optional anonymous aggregates."""

    rows, summary = inventory(root)
    private_output = Path(private_output)
    private_output.mkdir(parents=True, exist_ok=True)
    detailed_path = private_output / "file_inventory.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["relative_path"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (private_output / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if public_output is not None:
        public_output = Path(public_output)
        public_output.mkdir(parents=True, exist_ok=True)
        with (public_output / "aggregate_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["layer", "file_count", "total_bytes"])
            for layer in sorted(summary["layer_counts"]):
                writer.writerow([layer, summary["layer_counts"][layer], summary["layer_bytes"][layer]])
        aligned_rows = [
            row
            for row in rows
            if str(row["relative_path"]).startswith("data_process/aligned_data_strict/aligned_IMU/")
            and row["suffix"] == ".csv"
            and row["subject"]
        ]
        subject_codes = {
            subject: f"P{index:02d}"
            for index, subject in enumerate(sorted({str(row["subject"]) for row in aligned_rows}), 1)
        }
        session_counts = Counter(
            (subject_codes[str(row["subject"])], str(row["movement"])) for row in aligned_rows
        )
        with (public_output / "aligned_session_counts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["participant_code", "movement", "aligned_session_count"])
            for (participant, movement), count in sorted(session_counts.items()):
                writer.writerow([participant, movement, count])
        public_summary = {
            **summary,
            "root_name": "local_experiment_archive",
            "aligned_session_count": len(aligned_rows),
        }
        (public_output / "summary.json").write_text(
            json.dumps(public_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary
