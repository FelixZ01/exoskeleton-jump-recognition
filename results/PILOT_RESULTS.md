# Six-participant pilot results

These results are an honest engineering baseline, not a clinical or product-validation claim. All 224 aligned recordings from six participants were used. Each outer fold held out one complete participant; no participant appeared in both training and test data.

## Session-level feature baseline

Regularised logistic regression was evaluated on robust per-recording summary features.

| Input | Balanced accuracy | Macro F1 | Confusion matrix |
|---|---:|---:|---|
| IMU | 56.6% | 56.1% | `[[56, 61], [37, 70]]` |
| sEMG | **67.8%** | **67.8%** | `[[82, 35], [37, 70]]` |
| IMU + sEMG | 63.5% | 62.7% | `[[61, 56], [27, 80]]` |

The sEMG-only baseline performed best overall. Simple feature concatenation did not improve it, indicating that fusion requires better temporal alignment, event segmentation, or a model designed to suppress participant and sensor-configuration differences.

## Dual-branch 1D CNN

The CNN used 1-second non-overlapping windows, per-window normalisation, class-weighted loss, a separate validation participant for early stopping, and recording-level averaging of window probabilities. Across the six outer participant folds, mean recording-level balanced accuracy was **65.6% ± 14.2 percentage points**. Mean macro F1 was approximately **58.9%** after treating an unpredicted class as F1 = 0.

Performance varied substantially between held-out participants (balanced accuracy 50.0%–86.1%). The CNN did not consistently outperform the simpler sEMG baseline. This is evidence of learnable signal, but also of domain shift and overfitting risk.

## Pressure-event-centred dual-branch CNN

The aggregate plantar-pressure signal was smoothed and its low-pressure flight phase was used to centre 2-second windows. Three offsets (−250, 0, and +250 ms) produced 643 windows from 222 eligible recordings; two recordings were excluded because they did not yield a complete quality-controlled event window. The pressure signal located the event but was not passed to the dual-branch IMU+sEMG classifier.

Across three complete nested participant-held-out runs (seeds 42, 7, and 123), mean recording-level balanced accuracy was **88.8% ± 1.2 percentage points across seeds**, and mean macro F1 was **88.5% ± 1.3 percentage points**. Participant-level mean balanced accuracy ranged from 77.2% to 93.3%.

This is a substantial and repeatable improvement over arbitrary 1-second windows, showing that event definition was a larger bottleneck than network complexity. It remains an offline trial-classification result: continuous real-time recognition and product deployment have not been validated. A pressure signal or a separately validated event detector is required at inference time.

## Interpretation

- Six participants are enough for a pilot software demonstration, not a strong generalisation claim.
- P01 used a different effective IMU configuration from later participants, creating a hardware-domain shift.
- Arbitrary sliding windows contained preparation or recovery rather than the jump event itself; pressure-centred windows substantially improved the pilot result.
- Sensor placement, sEMG muscle mapping, and device-side filtering remain incompletely documented.

## Highest-priority next experiments

1. Compare a consistent four-IMU subset across every participant with sEMG-only and fusion models.
2. Verify whether sEMG is already device-filtered; if not, add band-pass, power-line notch, rectification, and envelope extraction.
3. Extend the current three-seed result to at least five seeds and add participant/session-level bootstrap confidence intervals.
4. Compare pressure-event detection against IMU-only event detection to quantify dependence on the insole signal.
5. Compare logistic regression, SVM, random forest, 1D ResNet/TCN, and gated or attention-based fusion.
6. Link jump height/distance and product settings to each recording before attempting parameter recommendation or regression.
