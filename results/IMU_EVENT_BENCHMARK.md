# IMU-only event detection benchmark

The formal Google Colab T4 experiment used all 224 eligible aligned sessions
from six participants and 647 two-second windows. Jump events were located from
IMU angular dynamics only; plantar pressure and sEMG were excluded from both
event localisation and classification.

| Model | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| CNN, IMU-only event windows | 90.0% ± 1.3% | 89.3% ± 1.6% |
| TCN, IMU-only event windows | **90.9% ± 3.5%** | **90.5% ± 3.1%** |
| CNN, pressure-centred IMU reference | 87.3% ± 3.0% | 86.8% ± 3.1% |

The IMU-event CNN exceeded its pressure-centred reference by 2.7 percentage
points in mean balanced accuracy. The TCN achieved the strongest result in this
experiment. This supports the feasibility of removing the pressure-derived
event anchor, but the six-person sample remains a pilot and is not evidence of
clinical or population-level generalisation.

Protocol: complete-participant holdout, recording-level aggregation, three
random seeds (42, 7, 123), up to 30 epochs, and early stopping.
