# Full Colab multi-model benchmark

Run date: 2026-09-07  
Runtime: Google Colab T4 GPU  
Dataset: 643 pressure-event-centred windows from 222 sessions and six anonymised participants  
Protocol: nested participant-held-out evaluation; seeds 42, 7, and 123; up to 30 epochs; early-stopping patience 5; 2,000 participant-clustered bootstrap resamples

All requested experiment processes completed with return code 0. The complete runtime output was archived as `exojump_full_results.zip`. Only aggregate, de-identified results are reported here.

## Fusion-model comparison

| Model | Balanced accuracy | Macro-F1 | 95% CI for pooled balanced accuracy |
|---|---:|---:|---:|
| Logistic regression | 67.6% | 67.0% | 49.4%–85.6% |
| SVM | 65.9% | 65.9% | 49.0%–82.8% |
| Random Forest | 72.6% | 72.7% | 59.4%–86.1% |
| MiniROCKET | 82.3% | 82.4% | 68.6%–92.4% |
| 1D CNN | **87.5% ± 1.6%** | **87.0% ± 1.3%** | 79.5%–94.6% |
| BiLSTM | 49.7% ± 2.3% | 45.2% ± 2.2% | 28.3%–70.8% |
| TCN | 84.5% ± 3.7% | 83.4% ± 4.6% | 69.6%–97.8% |

CNN achieved the best mean performance and was more stable across seeds than TCN. MiniROCKET was the strongest non-neural baseline, reaching 82.3% balanced accuracy without end-to-end neural optimisation. The present BiLSTM configuration did not learn a robust participant-independent decision boundary and should not be presented as a competitive result without architecture or optimisation changes.

## CNN modality ablation

| Input | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| IMU only | **87.3% ± 3.0%** | **86.8% ± 3.1%** |
| sEMG only | 53.5% ± 0.5% | 43.3% ± 1.8% |
| IMU + sEMG fusion | **87.5% ± 1.6%** | **87.0% ± 1.3%** |

Fusion improved mean balanced accuracy by only about 0.1 percentage points over IMU alone, although its across-seed variability was lower. The current result therefore supports an IMU-dominant classifier, not a strong claim that multimodal fusion materially improves accuracy. The weak sEMG result warrants verification of channel semantics, sensor placement, device filtering, synchronisation, and participant-domain shift.

## Seed-42 pooled session diagnostics

| Model/input | Balanced accuracy | Confusion matrix |
|---|---:|---|
| CNN fusion | 88.9% | `[[98, 18], [7, 99]]` |
| TCN fusion | 85.1% | `[[87, 29], [5, 101]]` |
| BiLSTM fusion | 52.1% | `[[88, 28], [76, 30]]` |
| CNN IMU only | 87.6% | `[[97, 19], [9, 97]]` |
| CNN sEMG only | 56.2% | `[[44, 72], [27, 79]]` |

These confusion matrices illustrate one seed only; the headline comparisons above use the mean and standard deviation across all three seeds.

## Interpretation and limitations

- Entire participants were held out, reducing identity leakage.
- Metrics were aggregated at session level and uncertainty was clustered by participant.
- Six participants support a technical pilot, not a population-level or clinical validation claim.
- Pressure-centred windows make this offline trial classification; continuous recognition and pressure-free deployment remain unvalidated.
- The result can support a CV or supervisor discussion when described as a reproducible six-participant pilot, but should not be described as proven product tuning or real-time performance.
- The next highest-value experiment is an IMU-only event detector and classifier evaluated without access to plantar pressure at inference time, followed by sEMG preprocessing and sensor-mapping verification.
