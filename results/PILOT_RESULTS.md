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

## Interpretation

- Six participants are enough for a pilot software demonstration, not a strong generalisation claim.
- P01 used a different effective IMU configuration from later participants, creating a hardware-domain shift.
- Arbitrary sliding windows can contain preparation or recovery rather than the jump event itself.
- Sensor placement, sEMG muscle mapping, and device-side filtering remain incompletely documented.

## Highest-priority next experiments

1. Detect take-off/landing from foot-pressure and IMU energy, then train on event-centred windows.
2. Compare a consistent four-IMU subset across every participant with sEMG-only and fusion models.
3. Verify whether sEMG is already device-filtered; if not, add band-pass, power-line notch, rectification, and envelope extraction.
4. Run at least five random seeds for each leave-one-participant-out fold and report participant-level confidence intervals.
5. Compare logistic regression, SVM, random forest, 1D ResNet/TCN, and gated or attention-based fusion.
6. Link jump height/distance and product settings to each recording before attempting parameter recommendation or regression.
