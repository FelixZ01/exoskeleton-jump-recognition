# Compact time-series Transformer benchmark

## Question and protocol

This experiment tests whether a compact attention model adds useful architectural diversity to the exoskeleton jump-recognition benchmark. It uses the complete anonymised IMU-event dataset: 647 windows from 224 trials and six participants. Evaluation remains nested participant-held-out, so the test participant never contributes training or validation windows.

Each configuration was evaluated over all six held-out participants with seeds 42, 7, and 123 (18 trained folds per configuration). The model uses a stride-8 convolutional tokenizer, two Transformer encoder layers, four attention heads, hidden width 64, dropout 0.4, AdamW weight decay `1e-4`, and early stopping with patience 5. Reported values are mean ± standard deviation across seeds.

## Results

| Transformer input | Balanced accuracy | Macro-F1 | Parameters | Mean best epoch | Mean epochs run |
|---|---:|---:|---:|---:|---:|
| R3+R4 compact IMU | **89.47% ± 1.74** | **89.29% ± 2.17** | 70,722 | 7.72 | 12.72 |
| Full IMU | 87.41% ± 8.59 | 86.56% ± 10.49 | 74,178 | 5.28 | 10.28 |
| Full IMU + sEMG | 84.61% ± 3.63 | 82.39% ± 3.98 | 78,786 | 4.67 | 9.67 |

The compact R3+R4 Transformer was the strongest attention configuration. It was 0.91 percentage points above the R3+R4 TCN mean (88.56%), although its seed variation was much larger (1.74 versus 0.23 percentage points). It remained 1.42 percentage points below the best full-sensor TCN result (90.89%). Therefore, this experiment supports the Transformer as a competitive comparison model, but does not establish it as superior to the TCN.

Adding sEMG did not help this Transformer: fusion was 2.81 percentage points below full IMU balanced accuracy and required 4,608 additional parameters. This agrees with the broader project finding that the present sEMG/pressure stream is not yet a reliable source of extra discriminative information after alignment and preprocessing.

## Overfitting assessment

All 54 trained folds stopped early before the 30-epoch limit. Mean final training loss was very low while final validation loss remained higher:

| Input | Early-stopped folds | Final train loss | Final validation loss | Best validation loss | Final validation − train loss |
|---|---:|---:|---:|---:|---:|
| R3+R4 compact IMU | 18/18 | 0.0306 | 0.3385 | 0.1800 | 0.3079 |
| Full IMU | 18/18 | 0.0075 | 0.4277 | 0.2086 | 0.4202 |
| Full IMU + sEMG | 18/18 | 0.0124 | 0.5763 | 0.2898 | 0.5639 |

These learning curves show a real overfitting tendency, especially for fusion. The experiment does not hide that limitation: validation-driven early stopping restores the best checkpoint instead of evaluating the final epoch. The smaller R3+R4 input produced both the best Transformer accuracy and the smallest train–validation gap, suggesting that removing weak or redundant channels acts as useful regularisation.

## Interpretation

The defensible conclusion is that a compact Transformer can classify the two jump types competitively, while the dilated TCN remains the strongest and more stable full-sensor model. R3+R4 is a promising compact hardware configuration across both architectures. Because there are only six participants, these results are pilot evidence; a larger external participant cohort is required before making deployment or physiological generalisation claims.

