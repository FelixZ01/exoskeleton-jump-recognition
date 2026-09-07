"""Neural architectures for multimodal jump time-series classification."""

from __future__ import annotations


ARCHITECTURES = ("cnn", "bilstm", "tcn", "transformer")
MODALITIES = ("imu", "semg", "fusion")


def build_model(
    imu_channels: int = 12,
    semg_channels: int = 8,
    classes: int = 2,
    *,
    architecture: str = "cnn",
    modality: str = "fusion",
    hidden_size: int = 64,
    dropout: float = 0.3,
):
    """Build a CNN, BiLSTM, TCN, or compact Transformer lazily.

    Every model accepts ``forward(imu, semg)``. Keeping one forward signature lets
    all architectures use the same participant splits, normalisation, training loop,
    and evaluation code. ``modality`` controls which inputs the model may use.
    """

    import torch
    from torch import nn

    architecture = architecture.lower()
    modality = modality.lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")

    def select_inputs(imu, semg):
        if modality == "imu":
            return imu
        if modality == "semg":
            return semg
        return torch.cat((imu, semg), dim=2)

    selected_channels = {
        "imu": imu_channels,
        "semg": semg_channels,
        "fusion": imu_channels + semg_channels,
    }[modality]

    class SignalBranch(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(channels, 32, kernel_size=9, padding=4),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, hidden_size, kernel_size=5, padding=2),
                nn.GroupNorm(8, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout / 3),
                nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, values):
            return self.layers(values.transpose(1, 2)).squeeze(-1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            if modality == "fusion":
                self.imu_branch = SignalBranch(imu_channels)
                self.semg_branch = SignalBranch(semg_channels)
                representation_size = hidden_size * 2
            else:
                self.branch = SignalBranch(selected_channels)
                representation_size = hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(representation_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, classes),
            )

        def forward(self, imu, semg):
            if modality == "fusion":
                representation = torch.cat(
                    (self.imu_branch(imu), self.semg_branch(semg)), dim=1
                )
            else:
                representation = self.branch(select_inputs(imu, semg))
            return self.classifier(representation)

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            # Downsampling makes 2,000-sample windows tractable while retaining order.
            self.downsample = nn.AvgPool1d(kernel_size=4, stride=4)
            self.recurrent = nn.LSTM(
                input_size=selected_channels,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, classes),
            )

        def forward(self, imu, semg):
            values = select_inputs(imu, semg).transpose(1, 2)
            values = self.downsample(values).transpose(1, 2)
            _, (hidden, _) = self.recurrent(values)
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(representation)

    class ResidualTCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            padding = 2 * dilation
            self.layers = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=5,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=5,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
            )

        def forward(self, values):
            return values + self.layers(values)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Conv1d(selected_channels, hidden_size, kernel_size=1)
            self.temporal = nn.Sequential(
                ResidualTCNBlock(hidden_size, dilation=1),
                ResidualTCNBlock(hidden_size, dilation=2),
                ResidualTCNBlock(hidden_size, dilation=4),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, classes),
            )

        def forward(self, imu, semg):
            values = select_inputs(imu, semg).transpose(1, 2)
            return self.classifier(self.temporal(self.input_projection(values)))

    class TimeSeriesTransformer(nn.Module):
        """Small Transformer encoder regularised for the six-person pilot."""

        def __init__(self):
            super().__init__()
            # Reduce 2,000 samples to 250 tokens to control attention cost.
            self.tokeniser = nn.Conv1d(
                selected_channels,
                hidden_size,
                kernel_size=9,
                stride=8,
                padding=4,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=max(dropout, 0.4),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(max(dropout, 0.4)),
                nn.Linear(hidden_size, classes),
            )

        @staticmethod
        def positional_encoding(length: int, width: int, device, dtype):
            positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
            scales = torch.exp(
                torch.arange(0, width, 2, device=device, dtype=dtype)
                * (-torch.log(torch.tensor(10_000.0, device=device, dtype=dtype)) / width)
            )
            encoding = torch.zeros(length, width, device=device, dtype=dtype)
            encoding[:, 0::2] = torch.sin(positions * scales)
            encoding[:, 1::2] = torch.cos(positions * scales)
            return encoding

        def forward(self, imu, semg):
            values = select_inputs(imu, semg).transpose(1, 2)
            tokens = self.tokeniser(values).transpose(1, 2)
            tokens = tokens + self.positional_encoding(
                tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype
            )
            representation = self.encoder(tokens).mean(dim=1)
            return self.classifier(representation)

    builders = {
        "cnn": CNN,
        "bilstm": BiLSTM,
        "tcn": TCN,
        "transformer": TimeSeriesTransformer,
    }
    return builders[architecture]()
