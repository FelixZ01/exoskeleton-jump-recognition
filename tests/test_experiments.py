import importlib.util
import unittest

import numpy as np

from exojump.benchmark import window_summary_features
from exojump.metrics import grouped_bootstrap_intervals
from exojump.training import prepare_fold


class ExperimentUtilityTests(unittest.TestCase):
    def test_window_features_have_eight_values_per_channel(self):
        values = np.arange(5 * 20 * 3, dtype=np.float32).reshape(5, 20, 3)
        features = window_summary_features(values)
        self.assertEqual(features.shape, (5, 24))
        self.assertTrue(np.isfinite(features).all())

    def test_grouped_bootstrap_returns_bounded_intervals(self):
        y_true = np.asarray([0, 1, 0, 1, 0, 1])
        y_pred = np.asarray([0, 1, 1, 1, 0, 0])
        groups = np.asarray(["P01", "P01", "P02", "P02", "P03", "P03"])
        intervals = grouped_bootstrap_intervals(
            y_true, y_pred, groups, resamples=50, seed=3
        )
        for values in intervals.values():
            self.assertLessEqual(values["lower"], values["estimate"])
            self.assertGreaterEqual(values["upper"], values["estimate"])
            self.assertEqual(values["resamples"], 50)

    def test_fold_preparation_keeps_participants_separate(self):
        rng = np.random.default_rng(4)
        subjects = np.repeat(np.asarray(["P01", "P02", "P03"]), 4)
        arrays = {
            "X_imu": rng.normal(size=(12, 16, 3)).astype(np.float32),
            "X_semg": rng.normal(size=(12, 16, 2)).astype(np.float32),
            "y": np.tile(np.asarray([0, 1]), 6),
            "subject": subjects,
            "session": np.asarray([f"S{index:02d}" for index in range(12)]),
        }
        fold = prepare_fold(
            arrays,
            test_subject="P03",
            validation_subject="P02",
            normalisation="train_global",
        )
        self.assertTrue(np.all(subjects[fold["train_mask"]] == "P01"))
        self.assertTrue(np.all(subjects[fold["validation_mask"]] == "P02"))
        self.assertTrue(np.all(subjects[fold["test_mask"]] == "P03"))
        self.assertAlmostEqual(float(fold["x_imu_train"].mean()), 0.0, places=5)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class NeuralArchitectureTests(unittest.TestCase):
    def test_all_architecture_and_modality_combinations(self):
        import torch

        from exojump.model import ARCHITECTURES, MODALITIES, build_model

        imu = torch.randn(3, 32, 12)
        semg = torch.randn(3, 32, 8)
        for architecture in ARCHITECTURES:
            for modality in MODALITIES:
                with self.subTest(architecture=architecture, modality=modality):
                    model = build_model(architecture=architecture, modality=modality)
                    output = model(imu, semg)
                    self.assertEqual(tuple(output.shape), (3, 2))
                    self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
