import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from exojump.alignment import align_frames
from exojump.dataset import discover_aligned_sessions, participant_split
from exojump.imu import convert_raw_imu, parse_payload
from exojump.imputation import impute_angles
from exojump.metrics import aggregate_session_probabilities, classification_metrics


class ImuConversionTests(unittest.TestCase):
    def test_recovered_sensor_group_mapping(self):
        payload = "Time--2:47:10:169," + ",".join(str(value) for value in range(24))
        elapsed, selected = parse_payload(payload)
        self.assertEqual(elapsed, 10_030_169)
        self.assertEqual(selected, [12, 13, 14, 6, 7, 8, 3, 4, 5, 9, 10, 11])

    def test_conversion_preserves_device_clock_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.csv"
            output = Path(directory) / "normalised.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["First_Timestamp_ms", "Data"])
                values = ",".join(str(value) for value in range(24))
                writer.writerow(["2025-10-22 16:00:00.000", f"Time--1:00:00:000,{values}"])
                writer.writerow(["", f"Time--1:00:00:003,{values}"])
            metrics = convert_raw_imu(source, output)
            converted = pd.read_csv(output)
            self.assertEqual(metrics["converted_rows"], 2)
            self.assertEqual(converted.loc[1, "First_Timestamp_ms"], "2025-10-22 16:00:00.003")


class PreprocessingTests(unittest.TestCase):
    def test_internal_missing_angles_are_interpolated(self):
        frame = pd.DataFrame(
            {
                "First_Timestamp_ms": ["a", "b", "c"],
                "R1_Roll": [1.0, np.nan, 3.0],
            }
        )
        result, metrics = impute_angles(frame)
        self.assertEqual(result.loc[1, "R1_Roll"], 2.0)
        self.assertEqual(metrics["filled"], 1)

    def test_alignment_uses_shared_millisecond_grid(self):
        imu = pd.DataFrame(
            {
                "First_Timestamp_ms": ["2025-01-01 00:00:00.001", "2025-01-01 00:00:00.003"],
                "Timestamp": ["0:0:0:1", "0:0:0:3"],
                "R1_Roll": [1.0, 3.0],
            }
        )
        semg = pd.DataFrame(
            {
                "Timestamp": [
                    "2025-01-01 00:00:00.000",
                    "2025-01-01 00:00:00.001",
                    "2025-01-01 00:00:00.002",
                    "2025-01-01 00:00:00.003",
                ],
                "Channel_1": [0.0, 1.0, 2.0, 3.0],
            }
        )
        aligned_imu, aligned_semg, metrics = align_frames(imu, semg)
        self.assertEqual(len(aligned_imu), 3)
        self.assertEqual(len(aligned_semg), 3)
        self.assertTrue(np.isnan(aligned_imu.loc[1, "R1_Roll"]))
        self.assertEqual(metrics["rows"], 3)

    def test_participant_split_has_no_overlap(self):
        subjects = np.asarray(["P01", "P01", "P02", "P02"])
        train_mask, test_mask, held_out = participant_split(subjects, "P02")
        self.assertEqual(held_out, "P02")
        self.assertTrue(np.all(subjects[train_mask] == "P01"))
        self.assertTrue(np.all(subjects[test_mask] == "P02"))

    def test_recovered_archive_layout_is_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            imu = root / "aligned_IMU/01_ABC_IMU_data/IMU_ABC_tiaogao/2025-01-02_03-04-05/IMU_2025-01-02_03-04-05.csv"
            semg = root / "aligned_sEMG/01_ABC_sEMG_data/sEMG_ABC_tiaogao/2025-01-02_03-04-05/processed_data_2025-01-02_03-04-05.csv"
            imu.parent.mkdir(parents=True)
            semg.parent.mkdir(parents=True)
            imu.write_text("R1_Roll\n1\n", encoding="utf-8")
            semg.write_text("Channel_1\n1\n", encoding="utf-8")
            sessions = discover_aligned_sessions(root)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0][:3], ("01_ABC", "tiaogao", "2025-01-02_03-04-05"))


class MetricTests(unittest.TestCase):
    def test_binary_metrics(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)

    def test_session_aggregation(self):
        probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]])
        labels = np.array([0, 0, 1])
        subjects = np.array(["P01", "P01", "P02"])
        sessions = np.array(["A", "A", "B"])
        actual, predicted, keys = aggregate_session_probabilities(probabilities, labels, subjects, sessions)
        np.testing.assert_array_equal(actual, np.array([0, 1]))
        np.testing.assert_array_equal(predicted, np.array([0, 1]))
        self.assertEqual(keys, ["P01/A", "P02/B"])


if __name__ == "__main__":
    unittest.main()
