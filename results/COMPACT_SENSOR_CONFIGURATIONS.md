# Compact IMU sensor configurations

This experiment directly retrained the best IMU-only TCN with compact sensor
subsets. It used all 224 eligible sessions, participant-held-out evaluation,
three random seeds, and early stopping.

| Retained nodes | IMU channels | Balanced accuracy | Macro-F1 | Change vs four nodes |
|---|---:|---:|---:|---:|
| R1–R4 | 12 | 90.9% ± 3.5% | 90.5% ± 3.1% | reference |
| R3 + R4 | 6 | **88.6% ± 0.2%** | **88.7% ± 0.2%** | −2.3 pp |
| R4 only | 3 | 84.5% ± 1.0% | 84.1% ± 1.2% | −6.4 pp |
| R3 only | 3 | 46.0% ± 2.2% | 31.7% ± 3.9% | −44.9 pp |

## Product interpretation

The R3+R4 pair retained 97.4% of the full model's mean balanced accuracy while
halving the modelled IMU channel count. R4 alone remained useful, whereas R3
alone was below the 50% binary balanced-accuracy reference. Together with the
leave-one-node-out results, this indicates that R4 is the essential node and R3
adds complementary motion information. R1 and R2 may be redundant or noisy in
this pilot dataset.

The R3+R4 configuration is therefore the strongest compact candidate for a
future product-tuning study. This is a model-level finding, not yet a hardware
removal recommendation: anatomical node mapping must be confirmed and the
result replicated in a larger prospective sample.
