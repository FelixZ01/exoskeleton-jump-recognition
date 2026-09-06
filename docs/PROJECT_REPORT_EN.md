# Multimodal Jump Recognition for a Civilian Exoskeleton — End-to-End Project Report

## 1. Project summary

This project investigates the recognition of vertical and long jumps for a civilian exoskeleton using surface electromyography (sEMG), inertial measurement units (IMUs), and plantar-pressure signals. The work spans human-participant data acquisition, heterogeneous sensor-data management, millisecond-level synchronisation, missing-data treatment, jump-event localisation, multimodal deep learning, and participant-independent evaluation.

The formal dataset contains six participants, two movement classes, and 227 paired IMU–sEMG acquisition sessions. A total of 224 sessions were retained after strict temporal alignment. Pressure-event localisation and window-level quality control produced 643 event-centred windows from 222 eligible sessions. The central evaluation question was whether a model could recognise jump type for a participant who had not appeared in training, rather than merely memorising participant-specific or session-specific patterns.

The current system is an exploratory research prototype. It demonstrates multimodal physiological-signal processing, leakage-aware machine learning, and reproducible research engineering; it is not yet a clinically validated or deployment-ready real-time product.

## 2. Motivation and objectives

An exoskeleton may require different control strategies and parameter settings for different movements. Reliable movement recognition could support control-mode selection, engineering calibration, and future human–machine coordination research. The available modalities provide complementary information:

- IMUs describe segment orientation and movement dynamics;
- sEMG captures neuromuscular activation during movement;
- plantar pressure represents support, take-off, flight, and landing phases;
- motion capture and jump-performance measurements may support later movement-quality or regression studies.

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

**IMUs.** Eight physical IMU nodes were placed on the right lower limb: six around the thigh, one on the lower leg, and one at the ankle. Formal acquisition packets contained eight three-value sensor slots. A data audit found valid values in all eight slots for the 27 P01 recordings, whereas four slots provided stable non-zero values in the remaining 200 recordings and the other slots were zero-filled. The historical processing pipeline therefore retained four stable nodes, denoted R1–R4. Each retained node contributes roll, pitch, and yaw, giving 12 orientation channels. The exact mapping from R1–R4 to physical placement has not yet been recovered and is not inferred without evidence.

**sEMG.** Eight sEMG channels were placed along the right side of the body and right lower limb, following a waist-to-leg line that covered the waist, thigh, lower leg, and dorsum of the foot. The source material does not establish the precise muscle name or order for `Channel_1` through `Channel_8`; the analysis therefore retains neutral channel identifiers.

**Plantar pressure.** The same device-processed CSV tables also contain 16 plantar-pressure channels (`FP_CH41–FP_CH48` and `FP_CH51–FP_CH58`) and an aggregate `sum_foot` or `count_foot` signal. Although stored beside sEMG, pressure is treated as a separate modality throughout the pipeline.

**Additional records.** The archive includes motion-capture files, jump-height and jump-distance measurements, and body measurements. These records have not yet been mapped reliably to every session and were therefore excluded from the current classification results.

### 3.3 Data scale

The recovered archive contains approximately 2.39 GB and 4,609 visible files, including 2,483 CSV files, 948 logs, and 1,160 analysis figures. It includes raw exports, device-processed tables, intermediate aligned data, imputed data, and early model-candidate folders. The pipeline preserves these layers rather than overwriting original recordings.

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
Classical baselines and dual-branch 1D CNN evaluation
```

### 4.1 IMU decoding and timestamp reconstruction

The raw IMU export stores a device clock and multiple angle values in a packed `Data` field. The parser decodes the device timestamp and anchors it to the first valid wall-clock timestamp. Subsequent timestamps are reconstructed from actual device-clock differences, preserving genuine gaps instead of masking dropped samples with a synthetic continuous row index.

Matching raw and converted rows indicates that the four retained groups use zero-based payload indices `[4, 2, 1, 3]`, mapped to R1–R4 roll, pitch, and yaw. This numeric mapping is verified; the anatomical placement mapping remains unresolved.

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

## 8. Conclusions

The project supports six main conclusions:

1. Reliable multimodal modelling depends first on defensible device synchronisation.
2. Participant-held-out evaluation substantially reduces the inflated results produced by random or resubstitution testing.
3. sEMG contains meaningful discriminative information for the two jump types.
4. Naive multimodal fusion can underperform a strong single-modality baseline.
5. Pressure-based event localisation provides a large, repeatable improvement in IMU+sEMG classification.
6. The current results establish feasibility for offline trial classification, not continuous real-time recognition, automatic exoskeleton tuning, or clinical performance.

## 9. Technical contributions and demonstrated skills

The project demonstrates the ability to:

- participate in multimodal human-movement data acquisition;
- organise approximately 2.39 GB and 4,609 heterogeneous experimental files;
- process IMU, sEMG, pressure, motion-capture, and performance data;
- reconstruct and verify timestamps across independent acquisition devices;
- implement millisecond alignment, dropped-sample detection, interpolation, and quality control;
- design plantar-pressure-based movement-event localisation;
- develop classical baselines and a dual-branch deep neural network;
- use participant-held-out validation, early stopping, class weighting, and repeated-seed evaluation;
- identify and correct label-definition and data-leakage problems in an early prototype;
- refactor scattered research scripts into a documented, tested, and reproducible GitHub project.

The present product claim is deliberately limited: the system provides a movement-recognition basis for later exoskeleton calibration. It does not yet perform validated automatic parameter optimisation.

## 10. Limitations

- The study includes only six participants.
- Effective IMU availability differed between acquisition batches.
- The R1–R4 physical-node mapping has not been recovered.
- Exact sEMG channel-to-muscle mapping remains unknown.
- Device-side sEMG filtering has not been fully documented.
- Two aligned sessions did not produce complete quality-controlled event windows.
- The task is offline classification of pre-recorded trials, not continuous streaming recognition.
- Motion capture, jump performance, and exoskeleton parameters are not yet reliably linked to every session.
- Cross-day, cross-device, and external-cohort validation have not been performed.

## 11. Next research steps

1. Recover the physical R1–R4 mapping and sEMG channel-to-muscle order.
2. Verify device-side sEMG processing and add filtering only if required.
3. Extend evaluation to at least five random seeds and participant/session-level bootstrap intervals.
4. Develop an IMU-only event detector and compare it with pressure-based localisation.
5. Compare SVM, random forest, 1D ResNet, TCN, BiLSTM, and attention-based fusion.
6. Quantify the contribution of each modality and fusion strategy through ablation studies.
7. Link jump height, distance, and exoskeleton settings to individual sessions.
8. Extend classification to movement-quality scoring, performance prediction, and parameter recommendation.
9. Recruit more participants and evaluate cross-day and cross-device generalisation.

## 12. Project outcomes and research significance

### Engineering outcomes

The project delivers an end-to-end multimodal workflow from experimental acquisition to model evaluation. Approximately 2.39 GB of experimental files were organised into traceable data layers, while IMU decoding, timestamp reconstruction, modality alignment, missing-data treatment, event localisation, dataset construction, and model validation were refactored into reusable software. Configuration files, command-line interfaces, anonymised manifests, automated tests, and experiment records support reproducibility and future extension.

### Methodological outcome

Strict participant-disjoint evaluation exposed the information loss caused by arbitrary sliding windows. A plantar-pressure-based flight-phase detector was consequently introduced to construct event-centred samples. Under nested leave-one-participant-out evaluation across six participants and three random seeds, the event-centred dual-branch CNN achieved 88.8% mean balanced accuracy and 88.5% mean macro F1, substantially improving on arbitrary one-second windows.

### Practical relevance

The workflow provides a data and validation foundation for movement-mode recognition and engineering calibration of civilian exoskeletons. Its contribution extends beyond classification performance: it establishes an auditable framework for multidevice synchronisation, signal-quality control, and participant-independent testing. The same framework could support gait recognition, rehabilitation-movement analysis, movement-quality assessment, and human–machine intent recognition.

### Research significance

The project highlights three central challenges in small-sample multimodal human-signal research: domain shift caused by sensor-configuration changes, leakage caused by random window splitting, and the effect of event definition on model generalisation. These findings motivate further work on cross-participant and cross-device domain generalisation, interpretable multimodal fusion, continuous-stream event detection, and predictive links between recognised movements, exoskeleton control settings, and individual user differences.

## 13. Reproducibility links

- Data dictionary: [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)
- Data audit: [`DATA_AUDIT.md`](DATA_AUDIT.md)
- Processing pipeline: [`PIPELINE.md`](PIPELINE.md)
- Model-validity review: [`MODEL_VALIDITY.md`](MODEL_VALIDITY.md)
- Pilot results: [`../results/PILOT_RESULTS.md`](../results/PILOT_RESULTS.md)
- Machine-readable metrics: [`../results/pilot_metrics.json`](../results/pilot_metrics.json)
- Core package: [`../src/exojump/`](../src/exojump/)
- Command-line tools: [`../scripts/`](../scripts/)
