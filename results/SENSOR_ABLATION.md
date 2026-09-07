# Leave-one-IMU-node-out retraining ablation

The best IMU-only TCN was retrained after removing each three-axis IMU node in
turn. All experiments used the same 224 sessions, participant-held-out protocol,
three random seeds, and early stopping. The full four-node TCN baseline achieved
90.9% balanced accuracy.

| Removed node | Remaining channels | Balanced accuracy | Macro-F1 | Accuracy change |
|---|---:|---:|---:|---:|
| None | 12 | 90.9% ± 3.5% | 90.5% ± 3.1% | reference |
| R1 | 9 | 91.1% ± 0.7% | 91.2% ± 0.7% | +0.2 pp |
| R2 | 9 | 91.1% ± 2.8% | 90.4% ± 3.3% | +0.2 pp |
| R3 | 9 | 81.5% ± 1.1% | 81.4% ± 0.7% | −9.3 pp |
| R4 | 9 | 62.8% ± 2.7% | 55.6% ± 2.6% | **−28.1 pp** |

## Interpretation

R4 carried the most discriminative information, followed by R3. Removing R1 or
R2 did not reduce mean performance, suggesting redundancy or participant-
specific noise within this pilot dataset. A product-oriented reduced sensor
configuration should retain R4 and R3, but physical placement recommendations
must wait until the historical R1–R4 identifiers are mapped conclusively to
anatomical locations. The six-person sample is too small to justify permanent
hardware removal without a larger prospective validation study.

The next experiment therefore retrains compact R3+R4, R4-only, and R3-only
TCNs rather than inferring a reduced design from occlusion alone.
Its completed results are reported in [Compact IMU sensor configurations](COMPACT_SENSOR_CONFIGURATIONS.md).
