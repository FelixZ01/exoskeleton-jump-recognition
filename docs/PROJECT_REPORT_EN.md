# Multimodal Jump Recognition for a Civilian Exoskeleton — End-to-End Project Report

## 1. Project summary

This project investigates the recognition of vertical and long jumps for a civilian exoskeleton using surface electromyography (sEMG), inertial measurement units (IMUs), and plantar-pressure signals. The work spans human-participant data acquisition, heterogeneous sensor-data management, millisecond-level synchronisation, missing-data treatment, jump-event localisation, multimodal deep learning, and participant-independent evaluation.

The formal dataset contains six participants, two movement classes, and 227 paired IMU–sEMG acquisition sessions. A total of 224 sessions were retained after strict temporal alignment. Pressure-event localisation and window-level quality control produced 643 event-centred windows from 222 eligible sessions. The central evaluation question was whether a model could recognise jump type for a participant who had not appeared in training, rather than merely memorising participant-specific or session-specific patterns.

The current system is an exploratory research prototype. Its completed evidence now includes pressure- and IMU-derived event localisation, classical and deep-learning baselines, CNN/TCN/BiLSTM/Transformer comparisons, repeated-seed evaluation, leave-one-sensor-out retraining, and compact-sensor experiments. It demonstrates multimodal physiological-signal processing, leakage-aware machine learning, and reproducible research engineering; it is not yet a clinically validated or deployment-ready real-time product.

## 2. Motivation and objectives

An exoskeleton may require different control strategies and parameter settings for different movements. Reliable movement recognition could support control-mode selection, engineering calibration, and future human–machine coordination research. The available modalities provide complementary information:

- IMUs describe segment orientation and movement dynamics;
- sEMG captures neuromuscular activation during movement;
- plantar pressure represents support, take-off, flight, and landing phases;
- motion capture and body measurements were used to calculate and verify participant-specific jump height, providing an objective performance reference for sensor-data analysis.

The project objectives were to:

1. organise heterogeneous recordings collected across devices, participants, and repeated trials;
2. build a traceable pipeline for IMU decoding, timestamp reconstruction, and IMU–sEMG alignment;
3. classify vertical versus long jumps under strict participant-independent evaluation;
4. compare the contributions of IMU, sEMG, plantar pressure, and multimodal fusion;
5. establish a technical basis for movement-aware exoskeleton calibration.

## 3. Experimental design

### 3.1 Participants and movement tasks

- Formal participants: six, anonymised as P01–P06 in all public outputs;
- Movement classes: vertical jump and long jump;
- Paired acquisition sessions: 227;
- Strictly aligned IMU–sEMG sessions: 224;
- Sessions retained for pressure-event-centred modelling: 222.

Both movement classes are present for every participant. P03–P06 each have 20 sessions per class; P01 and P02 have fewer sessions because some recordings were incomplete. Class-weighted training and participant-level splitting were used to manage this imbalance.

### 3.2 Sensor configuration

**IMUs.** Eight physical IMU nodes were arranged bilaterally, with four per side at the waist, anterior distal thigh just above the knee, anterior lower leg, and dorsum of the foot. Formal acquisition packets contained eight three-value sensor slots. A data audit found valid values in all eight slots for the 27 P01 recordings, whereas four slots provided stable non-zero values in the remaining 200 recordings and the other slots were zero-filled. The historical processing pipeline retained the right-side four-node chain. Each node contributes roll, pitch, and yaw, giving 12 orientation channels. In the acquisition naming convention, `R` denotes right and numbering follows the body from top to bottom: **R1 right waist/hip, R2 right anterior distal thigh, R3 right anterior lower leg, and R4 right dorsum of foot**.

**sEMG.** Eight sEMG channels were acquired on the right side. `Channel_1`–`Channel_6` correspond to six electrodes distributed around the thigh, `Channel_7` to the anterolateral lower leg, and `Channel_8` to the medial ankle. Exact muscle names remain unavailable, so the analysis does not attach unverified physiological labels to these positions.

**Plantar pressure.** The same device-processed CSV tables also contain 16 plantar-pressure channels (`FP_CH41–FP_CH48` and `FP_CH51–FP_CH58`) and an aggregate `sum_foot` or `count_foot` signal. Although stored beside sEMG, pressure is treated as a separate modality throughout the pipeline.

**Movement-performance data.** Motion-capture recordings and body measurements were used to calculate and verify each participant's jump height; long-jump trials also included jump-distance measurements. These continuous outcomes describe individual movement performance and provide a reference for analysing relationships with IMU, sEMG, and plantar-pressure signals. The present model classifies vertical versus long jumps, so stature, body dimensions, and measured jump height were not used as classification inputs, preventing identification through participant-specific characteristics.

### 3.3 Data scale

After excluding two experimental videos, the inventory contains approximately 2.30 GB across 4,607 non-video files, including 2,483 CSV files, 948 logs, and 1,160 analysis figures. These files include raw exports, device-processed tables, intermediate aligned data, imputed data, and early model-candidate folders. Because the archive also contains packaged duplicates, this figure describes the inventoried non-video files rather than a deduplicated volume of independent modelling data. The pipeline preserves the processing layers rather than overwriting original recordings.

## 4. Data-processing pipeline

```text
Multidevice acquisition
        ↓
Data inventory and provenance layers
        ↓
IMU payload decoding and timestamp reconstruction
        ↓
Session matching by participant, movement, and acquisition time
        ↓
Millisecond-grid IMU–sEMG alignment
        ↓
Dropped-sample detection and short-gap interpolation
        ↓
Plantar-pressure flight-phase localisation
        ↓
Event-centred IMU+sEMG windows
        ↓
Participant-disjoint train/validation/test splits
        ↓
Classical baselines and CNN/BiLSTM/TCN/Transformer evaluation
        ↓
Modality, sensor-node, and compact-configuration ablations
```

### 4.1 IMU decoding and timestamp reconstruction

The raw IMU export stores a device clock and multiple angle values in a packed `Data` field. The parser decodes the device timestamp and anchors it to the first valid wall-clock timestamp. Subsequent timestamps are reconstructed from actual device-clock differences, preserving genuine gaps instead of masking dropped samples with a synthetic continuous row index.

Matching raw and converted rows indicates that the four retained groups use zero-based payload indices `[4, 2, 1, 3]`, mapped to R1–R4 roll, pitch, and yaw. The numeric mapping is verified. Together with the photograph and legacy proximal-to-distal biomechanical constraints, it supports interpretation of R1–R4 as waist/hip, distal thigh, anterior lower leg, and dorsum of foot, respectively.

### 4.2 Multimodal temporal alignment

IMU and sEMG were recorded by different systems with non-identical timestamp formats and valid time ranges. Files are matched using participant, movement, and recording time. Each pair is cropped to its common valid interval and reindexed on a one-millisecond timeline so that corresponding rows represent the same instant.

Historical reports describe 227 attempted pairs, most of which aligned directly and a small number of which required repair or failed. The final strict-alignment directory contains 224 paired sessions.

### 4.3 Missing-data treatment

When the IMU device clock reveals a missing interval, the pipeline explicitly inserts missing rows before interpolating short gaps in the orientation channels. File-level counts are retained so that imputed values are not mistaken for observed measurements. During dataset construction, windows exceeding the joint IMU–sEMG missingness threshold are rejected.

### 4.4 Plantar-pressure event localisation

Initial one-second sliding windows covered preparation, take-off, flight, landing, and post-movement periods. Because jump timing varied between recordings, many windows did not contain the discriminative movement event.

The improved pipeline uses plantar pressure to locate the event:

1. read `sum_foot`, falling back to `count_foot` or the sum of the 16 pressure channels;
2. suppress short impulses using a centred rolling median;
3. exclude boundary regions at the start and end of each recording;
4. identify the minimum-pressure region as the flight-phase centre;
5. extract a two-second IMU+sEMG window around that centre;
6. generate light temporal augmentation at −250, 0, and +250 ms offsets;
7. average probabilities across windows from the same trial before session-level scoring.

Of the 224 aligned sessions, 222 produced complete event windows that satisfied the missingness criteria, yielding 643 model windows. Pressure was used to locate the event but was not passed into the dual-branch classifier; classification remained based on IMU and sEMG.

## 5. Modelling

### 5.1 Classical feature baseline

Classical baselines were included to determine whether deep learning added value. Robust session-level features—mean, standard deviation, root mean square, quartiles, median, range, and mean absolute first difference—were extracted per channel. Class-weighted, regularised logistic regression was evaluated separately for IMU, sEMG, pressure, and IMU+sEMG features. Imputation and standardisation were fitted only on training participants.

### 5.2 Dual-branch 1D CNN

The deep model contains separate temporal branches for the two signal types:

- the IMU branch receives 12 orientation channels;
- the sEMG branch receives eight signal channels;
- each branch uses two 1D convolutions, GroupNorm, ReLU, pooling, and dropout;
- global average-pooled branch representations are concatenated;
- a fully connected classifier predicts vertical or long jump.

GroupNorm was chosen instead of BatchNorm to avoid dependence on batch-level running statistics in a small-sample, cross-participant setting. Training also uses per-window normalisation, class-weighted cross-entropy, a participant-disjoint validation set, early stopping, and fixed random seeds.

### 5.3 Extended time-series model suite

The maintained benchmark applies the same participant splits and trial-level aggregation to logistic regression, RBF-SVM, random forest, MiniROCKET, CNN, BiLSTM, residual dilated TCN, and a compact time-series Transformer. The Transformer uses a stride-8 convolutional tokenizer, two encoder layers, four attention heads, dropout 0.4, AdamW weight decay, and validation-participant early stopping. This compact design adds an attention-based comparison while limiting the capacity and attention cost that would otherwise be inappropriate for a six-participant pilot.

An IMU-derived event detector was also implemented so that movement localisation no longer depends on plantar pressure. Formal retraining then evaluated full IMU input, leave-one-IMU-node-out variants, R3+R4, R4-only, and R3-only configurations. These are retraining experiments—not test-time masking—so the results can support sensor-design decisions more directly.

## 6. Evaluation design

A nested participant-held-out design was used:

- one complete participant forms the outer test fold;
- one of the remaining five participants forms the validation set;
- the other four participants form the training set;
- all six outer folds are evaluated, so every participant is tested once;
- the event-centred CNN experiment is repeated with seeds 42, 7, and 123;
- predictions from multiple windows belonging to the same trial are averaged before scoring.

The primary metrics are balanced accuracy—the mean recall across the two classes—and macro F1. Random window splitting was deliberately avoided because neighbouring windows from the same participant or trial would cause identity and temporal leakage.

## 7. Results

### 7.1 Baselines and arbitrary windows

| Method | Evaluation unit | Balanced accuracy | Macro F1 |
|---|---|---:|---:|
| IMU logistic regression | 224 sessions | 56.6% | 56.1% |
| sEMG logistic regression | 224 sessions | **67.8%** | **67.8%** |
| Plantar-pressure logistic regression | 224 sessions | 65.3% | 65.2% |
| IMU+sEMG logistic regression | 224 sessions | 63.5% | 62.7% |
| Dual-branch CNN, arbitrary 1 s windows | 1,025 windows | 65.6% ± 14.2% across folds | 58.9% |

sEMG was the strongest classical single modality. Simple IMU+sEMG feature concatenation did not outperform sEMG alone, demonstrating that multimodal input does not guarantee useful fusion when events are not temporally localised.

### 7.2 Pressure-event-centred CNN

| Seed | Mean balanced accuracy across six folds | Mean macro F1 |
|---:|---:|---:|
| 42 | 89.8% | 89.5% |
| 7 | 87.6% | 87.0% |
| 123 | 89.1% | 89.0% |
| **Three-run mean** | **88.8% ± 1.2%** | **88.5% ± 1.3%** |

Mean participant-level balanced accuracy across the three runs ranged from 77.2% to 93.3%. Event localisation raised mean balanced accuracy from 65.6% for arbitrary windows to 88.8%, indicating that event definition was a more important bottleneck than increasing model complexity.

### 7.3 IMU-only event localisation and model comparison

The IMU event detector retained all 224 aligned sessions and produced 647 windows without using pressure to identify the jump centre. Under the same nested participant-held-out protocol, the IMU-only CNN reached 90.00% ± 1.30% balanced accuracy and the full-IMU TCN reached **90.89% ± 3.50%**, the strongest full-sensor result. In the wider benchmark, CNN fusion reached 87.5% ± 1.6%, TCN fusion 84.5% ± 3.7%, MiniROCKET 82.3%, random forest 72.6%, logistic regression 67.6%, RBF-SVM 65.9%, and BiLSTM 49.7% balanced accuracy.

### 7.4 Sensor-node and compact-configuration experiments

Leave-one-node-out TCN retraining showed that removing R1 or R2 caused no meaningful loss, while removing R3 reduced balanced accuracy by 9.34 percentage points and removing R4 reduced it by 28.13 points relative to the 90.89% full-IMU baseline. R4 was therefore the most influential retained node in this dataset.

The R3+R4 compact TCN retained 88.56% ± 0.23% balanced accuracy, only 2.33 points below the full model. R4 alone reached 84.47% ± 0.96%; R3 alone reached 46.00% ± 2.18%. R3+R4 represents the right anterior-lower-leg and right dorsum-of-foot pair and is the strongest two-node configuration.

### 7.5 Compact Transformer and overfitting diagnosis

| Transformer input | Balanced accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| R3+R4 compact IMU | **89.47% ± 1.74%** | **89.29% ± 2.17%** | 70,722 |
| Full IMU | 87.41% ± 8.59% | 86.56% ± 10.49% | 74,178 |
| Full IMU+sEMG | 84.61% ± 3.63% | 82.39% ± 3.98% | 78,786 |

R3+R4 was also the strongest Transformer input, but the Transformer did not surpass the full-IMU TCN and was less stable across seeds. All 54 Transformer folds stopped early. Very low final training losses and substantially higher validation losses show a genuine overfitting tendency, especially for fusion. Early stopping restored the best validation checkpoint, and the result is reported as controlled pilot evidence rather than evidence that attention is inherently superior.

## 8. Conclusions

The project supports nine main conclusions:

1. Reliable multimodal modelling depends first on defensible device synchronisation.
2. Participant-held-out evaluation substantially reduces the inflated results produced by random or resubstitution testing.
3. sEMG contains meaningful discriminative information for the two jump types.
4. Naive multimodal fusion can underperform a strong single-modality baseline.
5. Pressure-based event localisation provides a large, repeatable improvement in IMU+sEMG classification.
6. The current results establish feasibility for offline trial classification, not continuous real-time recognition, automatic exoskeleton tuning, or clinical performance.
7. IMU-derived event localisation can remove the pressure dependency while retaining all 224 aligned trials.
8. Right dorsum-of-foot R4 carries the strongest node-level information, while the right lower-leg R3 plus right-foot R4 pair preserves most of the full-sensor performance.
9. A compact Transformer is competitive, particularly with R3+R4, but TCN is the stronger and more stable full-sensor choice; explicit overfitting diagnostics are essential in this small cohort.

## 9. Technical contributions and demonstrated skills

The project demonstrates the ability to:

- participate in multimodal human-movement data acquisition;
- inventory and organise approximately 2.30 GB across 4,607 non-video experimental files;
- process IMU, sEMG, pressure, motion-capture, and performance data;
- reconstruct and verify timestamps across independent acquisition devices;
- implement millisecond alignment, dropped-sample detection, interpolation, and quality control;
- design plantar-pressure-based movement-event localisation;
- develop classical baselines, MiniROCKET, CNN, BiLSTM, TCN, and a compact time-series Transformer;
- use participant-held-out validation, early stopping, class weighting, and repeated-seed evaluation;
- design IMU-only event localisation, leave-one-sensor-out retraining, compact-sensor comparisons, and learning-curve overfitting diagnostics;
- identify and correct label-definition and data-leakage problems in an early prototype;
- refactor scattered research scripts into a documented, tested, and reproducible GitHub project.

The present product claim is deliberately limited: the system provides a movement-recognition basis for later exoskeleton calibration. It does not yet perform validated automatic parameter optimisation.

## 10. Limitations

- The study includes only six participants.
- Effective IMU availability differed between acquisition batches.
- sEMG channel-to-position mapping is confirmed, but exact channel-to-muscle mapping remains unknown.
- Device-side sEMG filtering has not been fully documented.
- Two aligned sessions did not produce complete quality-controlled event windows.
- The task is offline classification of pre-recorded trials, not continuous streaming recognition.
- The current classification study does not yet model the continuous relationship between sensor features and measured jump height or distance.
- Cross-day, cross-device, and external-cohort validation have not been performed.

## 11. Next research steps

1. Use a small calibration experiment to verify IMU axis orientation and confirm the exact muscles underlying the known sEMG positions.
2. Verify device-side sEMG processing and add filtering only if required.
3. Confirm the IMU-versus-pressure event agreement with manually annotated take-off and landing times.
4. Recruit a larger external cohort and evaluate cross-day and cross-device generalisation.
5. Repeat the selected final TCN and compact Transformer with at least five seeds and prospective participants.
6. Investigate why aligned sEMG does not improve the current deep models before attempting more complex fusion.
7. Extend the dataset to jump-height and jump-distance regression and analyse their relationship with exoskeleton settings.
8. Extend classification to movement-quality scoring, performance prediction, and parameter recommendation.
9. Evaluate the right-lower-leg R3 plus right-foot R4 reduced-hardware design in a larger cohort.

## 12. Project outcomes and research significance

### Engineering outcomes

The project delivers an end-to-end multimodal workflow from experimental acquisition to model evaluation. After video exclusion, approximately 2.30 GB across 4,607 non-video experimental files were inventoried and organised into traceable data layers. IMU decoding, timestamp reconstruction, modality alignment, missing-data treatment, event localisation, dataset construction, and model validation were refactored into reusable software. Configuration files, command-line interfaces, anonymised manifests, automated tests, and experiment records support reproducibility and future extension.

### Methodological outcome

Strict participant-disjoint evaluation exposed the information loss caused by arbitrary sliding windows. Pressure- and IMU-derived event detectors were consequently introduced. The IMU-centred full-sensor TCN achieved 90.89% mean balanced accuracy, while R3+R4 retained 88.56% with TCN and reached 89.47% with the compact Transformer. Leave-one-node-out retraining and explicit learning-curve analysis added sensor-importance and overfitting evidence beyond a single headline accuracy.

### Practical relevance

The workflow provides a data and validation foundation for movement-mode recognition and engineering calibration of civilian exoskeletons. Its contribution extends beyond classification performance: it establishes an auditable framework for multidevice synchronisation, signal-quality control, and participant-independent testing. The same framework could support gait recognition, rehabilitation-movement analysis, movement-quality assessment, and human–machine intent recognition.

### Research significance

The project highlights three central challenges in small-sample multimodal human-signal research: domain shift caused by sensor-configuration changes, leakage caused by random window splitting, and the effect of event definition on model generalisation. These findings motivate further work on cross-participant and cross-device domain generalisation, interpretable multimodal fusion, continuous-stream event detection, and predictive links between recognised movements, exoskeleton control settings, and individual user differences.

## 13. Reproducibility links

- Data dictionary: [English](DATA_DICTIONARY.md) | [中文](DATA_DICTIONARY_ZH.md)
- Data audit: [English](DATA_AUDIT.md) | [中文](DATA_AUDIT_ZH.md)
- Processing pipeline: [English](PIPELINE.md) | [中文](PIPELINE_ZH.md)
- Model-validity review: [English](MODEL_VALIDITY.md) | [中文](MODEL_VALIDITY_ZH.md)
- Sensor-placement photograph and evidence: [English](SENSOR_PLACEMENT.md) | [中文](SENSOR_PLACEMENT_ZH.md)
- Pilot results: [English](../results/PILOT_RESULTS.md) | [中文](../results/PILOT_RESULTS_ZH.md)
- Full model benchmark: [English](../results/COLAB_FULL_BENCHMARK.md) | [中文](../results/COLAB_FULL_BENCHMARK_ZH.md)
- IMU-event benchmark: [English](../results/IMU_EVENT_BENCHMARK.md) | [中文](../results/IMU_EVENT_BENCHMARK_ZH.md)
- Sensor ablation: [English](../results/SENSOR_ABLATION.md) | [中文](../results/SENSOR_ABLATION_ZH.md)
- Compact-sensor benchmark: [English](../results/COMPACT_SENSOR_CONFIGURATIONS.md) | [中文](../results/COMPACT_SENSOR_CONFIGURATIONS_ZH.md)
- Transformer benchmark and overfitting analysis: [English](../results/TRANSFORMER_BENCHMARK.md) | [中文](../results/TRANSFORMER_BENCHMARK_ZH.md)
- Machine-readable metrics: [`../results/transformer_metrics.json`](../results/transformer_metrics.json)
- Core package: [`../src/exojump/`](../src/exojump/)
- Command-line tools: [`../scripts/`](../scripts/)
