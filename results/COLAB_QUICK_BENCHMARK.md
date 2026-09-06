# Colab quick multi-model benchmark

Run date: 2026-09-06  
Runtime: Google Colab T4 GPU  
Dataset: 643 pressure-event-centred windows from 222 sessions and six anonymised participants  
Protocol: nested participant-held-out evaluation, one seed (`42`), three training epochs, early-stopping patience of two, and 200 participant-clustered bootstrap resamples

This run is a pipeline smoke test and preliminary model comparison. It is not the reportable final benchmark because it uses only one seed and three epochs. The final experiment should use at least three seeds and 30 epochs.

## Fusion-model comparison

| Model | Balanced accuracy | Macro-F1 | Pooled session balanced accuracy | Pooled session macro-F1 | 95% CI for pooled balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 67.6% | 67.0% | 67.6% | 67.0% | 49.4%–86.6% |
| SVM | 65.9% | 65.9% | 65.9% | 65.9% | 49.0%–84.0% |
| 1D CNN | **88.3%** | 87.8% | 88.1% | 88.2% | 81.3%–95.5% |
| BiLSTM | 50.3% | 44.9% | 51.7% | 49.4% | 36.6%–78.2% |
| TCN | 88.2% | **88.4%** | **89.1%** | **89.1%** | 81.3%–95.5% |

The CNN and TCN produced nearly identical participant-held-out performance and clearly exceeded the summary-feature baselines in this quick run. The BiLSTM result should not be treated as evidence that recurrent models are unsuitable: three epochs are insufficient for a reliable architecture comparison.

## CNN modality ablation

| Input modality | Balanced accuracy across folds | Macro-F1 across folds | Pooled session balanced accuracy | Pooled session macro-F1 | Confusion matrix |
|---|---:|---:|---:|---:|---|
| IMU only | **88.1%** | **87.5%** | **88.5%** | **88.3%** | `[[96, 20], [6, 100]]` |
| sEMG only | 48.5% | 33.9% | 52.7% | 42.5% | `[[14, 102], [7, 99]]` |

In this run, the event-centred IMU stream carried most of the discriminative signal. The weak sEMG-only result may reflect sensor placement, filtering, channel semantics, temporal alignment, or participant-to-participant domain shift. It should trigger data-quality and preprocessing checks rather than a claim that sEMG is intrinsically uninformative.

## Reproducibility notes

- The test participant was never present in the training or validation data.
- Metrics were aggregated at session level, not only at window level.
- Confidence intervals used participant-clustered resampling.
- CNN: 35,202 parameters; approximately 3.64 seconds total training time for the quick six-fold run on T4.
- BiLSTM: 44,290 parameters; approximately 2.13 seconds.
- TCN: 125,506 parameters; approximately 7.85 seconds.
- The complete Colab output archive was generated as `exojump_quick_results.zip` in the temporary runtime.

## Next run

Use `QUICK_MODE = False` to run three seeds, 30 epochs, Logistic/SVM/Random Forest/MiniROCKET, CNN/BiLSTM/TCN, modality ablation, and 2,000 participant-clustered bootstrap resamples. Conclusions intended for a CV, supervisor discussion, or paper should be based on that full run rather than this smoke test.
