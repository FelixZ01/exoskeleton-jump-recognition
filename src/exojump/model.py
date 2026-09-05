"""Dual-branch 1D CNN for sEMG/IMU fusion."""

from __future__ import annotations


def build_model(imu_channels: int = 12, semg_channels: int = 8, classes: int = 2):
    """Construct the model lazily so non-training utilities do not require PyTorch."""

    import torch
    from torch import nn

    class SignalBranch(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(channels, 32, kernel_size=9, padding=4),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, values):
            return self.layers(values.transpose(1, 2)).squeeze(-1)

    class DualBranchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.imu = SignalBranch(imu_channels)
            self.semg = SignalBranch(semg_channels)
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, classes),
            )

        def forward(self, imu, semg):
            return self.classifier(torch.cat((self.imu(imu), self.semg(semg)), dim=1))

    return DualBranchCNN()
