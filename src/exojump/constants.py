"""Shared channel and label definitions."""

IMU_CHANNELS = [
    f"R{sensor}_{angle}"
    for sensor in range(1, 5)
    for angle in ("Roll", "Pitch", "Yaw")
]

SEMG_CHANNELS = [f"Channel_{index}" for index in range(1, 9)]

MOVEMENT_LABELS = {"tiaogao": 0, "tiaoyuan": 1}
MOVEMENT_NAMES = {0: "vertical_jump", 1: "long_jump"}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
