# Technical Work and Individual Contributions

## 1. Project role

My work on this civilian exoskeleton project focused on converting multimodal human-movement recordings into reliable data for modelling and product-oriented analysis. I was responsible for, or closely involved in, multimodal data collection, experimental file organisation, cross-device timestamp alignment, missing-data treatment, signal-quality analysis, machine-learning and deep-learning experiments, and interpretation of the resulting model performance.

The project investigated the classification of vertical and long jumps using inertial measurement unit (IMU), surface electromyography (sEMG), plantar-pressure, and motion-capture data. I helped turn fragmented, device-specific experimental outputs into a traceable processing and evaluation workflow. The main modelling objective was generalisation to unseen participants, with the longer-term aim of supporting movement recognition and engineering parameter tuning for a civilian exoskeleton.

## 2. Workload at a glance

| Workstream | Main work completed or supported | Scale and output |
|---|---|---|
| Human-participant data collection | Assisted participants with the exoskeleton experiment, collected vertical- and long-jump motion-capture data, and used body measurements to verify jump height | 6 participants and 2 movement classes |
| Sensor-data management | Organised IMU, sEMG, plantar-pressure, motion-capture, jump-performance, and experimental log files | Approximately 2.30 GB across 4,607 files after video exclusion |
| Data inventory and pairing | Grouped files by participant, movement, and acquisition time; matched IMU and sEMG recordings | 227 IMU–sEMG session pairs |
| Time synchronisation | Inspected timestamps from separate devices, reconstructed a common millisecond timeline, and handled differences in valid start and end times | 224 fully aligned sessions |
| Missing and anomalous data | Detected dropped IMU samples, inserted missing rows, interpolated short gaps, and retained processing records | 224 aligned sessions processed |
| Movement-event localisation | Used plantar-pressure changes to locate the flight phase and constructed quality-controlled event-centred windows | 222 eligible sessions and 643 model windows |
| Machine-learning baselines | Engineered statistical features and compared logistic-regression baselines using IMU, sEMG, pressure, and fused inputs | 67.8% balanced accuracy for the sEMG baseline |
| Deep-learning modelling | Developed a dual-branch 1D CNN for IMU and sEMG, with early stopping, class weighting, and prediction aggregation | 88.8% mean balanced accuracy across three seeds |
| Model benchmarking | Implemented and compared CNN, TCN, BiLSTM, compact Transformer, MiniROCKET, random forest, SVM, and logistic regression under a common protocol | Best full-IMU TCN balanced accuracy: 90.89% |
| Sensor ablation | Retrained TCN after removing each right-side IMU node and evaluated lower-leg R3 plus foot R4, R4-only, and R3-only inputs | R3+R4 retained 88.56%; right-foot R4 was the most influential node |
| Attention-model analysis | Built a regularised time-series Transformer and analysed learning curves, seed variation, parameter counts, and early stopping | R3+R4 Transformer: 89.47%; all folds checked for overfitting |
| Generalisation assessment | Separated training, validation, and test data by participant to prevent identity leakage | Six-fold nested leave-one-participant-out evaluation |
| Reproducible engineering | Organised legacy Python code into a documented project with configuration, command-line tools, tests, and anonymised manifests | Reproducible GitHub repository structure |

## 3. Human-participant experiments and multimodal acquisition

### 3.1 Experimental work

- Participated in collecting civilian-exoskeleton jump data from six participants.
- Supported repeated trials of vertical and long jumps, experimental record keeping, and file archiving.
- Collected and organised IMU, sEMG, plantar-pressure, and motion-capture recordings; combined motion capture with body measurements to calculate and verify participant-specific jump height; and organised jump-distance and experimental-log records.
- Checked device connectivity, file generation, and recording validity during acquisition to reduce unusable trials caused by interruptions or missing outputs.

### 3.2 Sensor and channel review

- Confirmed an eight-node bilateral IMU arrangement with four nodes per side: waist, anterior distal thigh just above the knee, anterior lower leg, and dorsum of the foot.
- Identified the 12 orientation channels from four consistently usable IMU nodes retained for modelling, with roll, pitch, and yaw from each node.
- Confirmed the right-side sEMG channel-position map: Channels 1–6 around the thigh, Channel 7 on the anterolateral lower leg, and Channel 8 on the medial ankle, while preserving uncertainty about exact muscle names.
- Identified and separately managed 16 plantar-pressure channels stored in the same acquisition tables as the sEMG signals.
- Audited changes in effective IMU availability between participants and acquisition batches, identifying a potential source of hardware-related domain shift.

## 4. Data organisation and quality control

### 4.1 Experimental data inventory

- Separated raw exports, device-processed files, aligned data, interpolated data, figures, and logs into traceable processing stages.
- After excluding two experimental videos, audited approximately 2.30 GB across 4,607 non-video files, including 2,483 CSV files, 948 log files, and 1,160 analysis figures.
- Reported the inventoried archive size separately from the deduplicated modelling-data volume, excluding videos and packaged copies from the modelling-data total.
- Established file relationships using participant, movement class, acquisition time, and processing stage.
- Preserved raw files and historical outputs instead of overwriting them, allowing the processing history to be reviewed.

### 4.2 Structural and anomaly checks

- Used Python to inspect CSV schemas, recording lengths, missingness, time ranges, and channel validity in batches.
- Identified inactive IMU slots, inconsistent recording durations, irregular sampling intervals, and sensor-configuration differences across acquisition batches.
- Flagged records that could not be used reliably rather than concealing anomalies in the reported results.
- Produced anonymised aggregate manifests and a data dictionary so that the dataset structure, fields, and processing status could be independently reviewed.

## 5. Cross-device synchronisation and missing-data treatment

### 5.1 IMU parsing and timestamp reconstruction

- Parsed device time and multiple orientation measurements packed into the raw IMU serial payload.
- Reconstructed a millisecond-scale timeline using the first valid wall-clock value as an anchor while preserving the device's observed sampling intervals.
- Compared raw payload rows with converted records to verify the extraction order of the four usable IMU groups.
- Preserved evidence of dropped samples rather than generating an artificially continuous timeline from row numbers.

### 5.2 sEMG–IMU alignment

- Manually investigated the temporal relationship between IMU and sEMG files and established rules for matching recordings from separate devices.
- Converted the manual matching logic into a Python workflow that paired files by participant, movement, and acquisition time.
- Cropped recordings to their common valid interval and resampled them onto a shared one-millisecond time grid.
- Produced 224 fully aligned IMU–sEMG sessions in which each row represents multimodal observations from the same point in time.

### 5.3 Dropped samples and interpolation

- Detected missing IMU samples from gaps in the device clock.
- Inserted explicit missing rows so that observed and subsequently imputed values remained distinguishable.
- Applied linear interpolation to short gaps and retained file-level processing reports.
- Recalculated joint IMU–sEMG missingness when constructing model windows and rejected windows that failed the quality threshold.

## 6. Signal analysis and movement-event localisation

- Conducted time-series visualisation, signal-quality checks, and exploratory cross-modal analysis for IMU, sEMG, and plantar-pressure signals.
- Used statistical features, correlation analysis, and principal component analysis to examine signal behaviour and redundancy.
- Determined that arbitrary sliding windows often captured preparation or post-landing activity rather than the jump itself, weakening the learned movement representation.
- Developed a plantar-pressure-based event-localisation method that smoothed the aggregate pressure signal and treated the low-pressure flight phase as the jump centre.
- Extracted two-second IMU+sEMG windows around the detected event and used −250, 0, and +250 ms offsets for limited temporal augmentation.
- Generated 643 windows from 222 quality-controlled sessions out of the 224 aligned recordings.

## 7. Machine-learning and deep-learning development

### 7.1 Conventional machine-learning baselines

- Extracted robust summary features including mean, standard deviation, root mean square, quartiles, median, range, and mean absolute first difference.
- Built an interpretable logistic-regression baseline with regularisation and class weighting.
- Evaluated IMU, sEMG, plantar-pressure, and IMU+sEMG inputs separately to assess their individual contributions.
- Fitted missing-value imputation and feature scaling on training participants only, preventing information from the test participants from entering the training pipeline.
- Obtained 67.8% balanced accuracy and 67.8% macro-F1 with the sEMG-only baseline.

### 7.2 Dual-branch 1D CNN

- Implemented a dual-branch one-dimensional convolutional neural network in PyTorch for joint IMU and sEMG modelling.
- Used one branch for 12 IMU orientation channels and a second branch for eight sEMG channels.
- Combined one-dimensional convolution, GroupNorm, ReLU activation, pooling, and dropout within each branch before feature fusion and binary classification.
- Applied class-weighted cross-entropy, per-window normalisation, early stopping on an independent validation participant, and fixed random seeds to reduce overfitting.
- Averaged probabilities across augmented windows from the same trial and reported final predictions at the recording level rather than treating correlated windows as independent samples.

### 7.3 Multi-model, sensor-ablation, and Transformer experiments

- Implemented a shared PyTorch interface for CNN, BiLSTM, residual dilated TCN, and compact time-series Transformer architectures, alongside classical and MiniROCKET baselines.
- Built an IMU-derived jump-event detector that retained all 224 aligned trials and produced 647 event-centred windows without requiring plantar pressure at inference.
- Ran full nested participant-held-out comparisons across six test participants and three random seeds, with recording-level aggregation and participant-clustered uncertainty estimates.
- Conducted leave-one-IMU-node-out retraining rather than relying on test-time masking, identifying right dorsum-of-foot R4 as the most influential retained node.
- Evaluated right lower-leg R3 plus right-foot R4, R4-only, and R3-only hardware-reduction candidates; R3+R4 preserved most of the full-sensor performance.
- Designed a small-sample-aware Transformer with strided tokenisation, two encoder layers, dropout 0.4, AdamW weight decay, and early stopping.
- Diagnosed rather than concealed Transformer overfitting by comparing training and validation losses, best and completed epochs, parameter counts, and variation across seeds.

## 8. Experimental design and validity controls

- Defined generalisation to previously unseen participants as the primary evaluation objective.
- Used a nested leave-one-participant-out design: one participant for outer-loop testing, a second participant for validation, and the remaining participants for training.
- Completed all six outer folds so that every participant served once as the fully held-out test participant.
- Repeated the complete evaluation using random seeds 42, 7, and 123 to measure sensitivity to model initialisation.
- Reported balanced accuracy and macro-F1 to reflect performance across both movement classes.
- Identified and corrected the lack of strict train–test separation in an early prototype, and excluded the overlap-affected result from the validated findings.
- Maintained a clear distinction between offline trial classification, continuous real-time recognition, and automatic product-parameter optimisation.

## 9. Results and technical diagnosis

- The arbitrary one-second-window CNN achieved 65.6% mean balanced accuracy, with substantial variation between held-out participants.
- Error analysis indicated that unreliable coverage of the core jump event, rather than insufficient network depth, was the main performance bottleneck.
- After introducing plantar-pressure event localisation, the event-centred CNN achieved the following results across three random seeds and six participant-held-out folds:
  - **88.8% ± 1.2 percentage points** mean balanced accuracy;
  - **88.5% ± 1.3 percentage points** mean macro-F1.
- Balanced accuracy improved from 65.6% with arbitrary windows to 88.8% with event-centred windows, demonstrating the importance of reliable event definition.
- The IMU-event full-sensor TCN achieved **90.89% ± 3.50%** balanced accuracy; the corresponding CNN achieved **90.00% ± 1.30%**.
- Removing R4 reduced TCN balanced accuracy by 28.13 percentage points, while the R3+R4 compact TCN retained **88.56% ± 0.23%**.
- The R3+R4 Transformer reached **89.47% ± 1.74%** balanced accuracy, but its larger seed variation and early-stopping behaviour did not justify replacing the TCN as the main model.
- Verified that all 54 Transformer folds stopped early and documented the train–validation loss gaps as evidence of residual overfitting risk.
- Documented limitations including the small cohort, sensor-configuration variation, unresolved sEMG muscle labels, and uncertainty about device-side filtering, clearly separating offline validation from deployment-level performance.

## 10. Reproducible engineering and documentation

- Reorganised fragmented Python scripts by acquisition, preprocessing, analysis, plotting, motion capture, and modelling purpose.
- Preserved legacy code for provenance while establishing a clearer main pipeline for future use.
- Replaced hard-coded paths in the maintained workflow with configuration files and command-line arguments.
- Modularised IMU conversion, modality alignment, interpolation, pressure- and IMU-event dataset construction, classical/deep benchmarking, sensor ablation, CNN/TCN/BiLSTM/Transformer training, and cross-validation.
- Added tests for critical preprocessing steps, anonymised data inventories, machine-readable experiment metrics, and reproducible commands.
- Produced documentation covering data auditing, field definitions, pipeline stages, model validity, privacy boundaries, and the complete project workflow in both Chinese and English.
- Applied anonymisation and Git exclusion rules to human-participant data, publishing code and aggregate results without releasing raw participant recordings.

## 11. Support for product development and future research

- Delivered an offline prototype for distinguishing vertical and long jumps during civilian-exoskeleton experiments.
- Provided an evidence base for movement-mode recognition and subsequent engineering parameter tuning.
- Used data-quality auditing to identify sensor failures, configuration changes, and synchronisation errors that can inform improvements to experimental and device workflows.
- Established a foundation for movement-quality scoring, jump-height or distance prediction, participant-specific modelling, and future parameter recommendation.
- Defined follow-up experiments involving cross-session and cross-device validation, IMU-only event detection, and modality-ablation studies.

## 12. Skills demonstrated

### Data and programming

- Python programming for scientific data processing and workflow automation;
- NumPy, pandas, SciPy, and related scientific-computing tools;
- Batch cleaning, matching, and quality auditing of complex CSV and experimental log data;
- Millisecond-scale time-series reconstruction, resampling, interpolation, and multimodal synchronisation;
- Git/GitHub project organisation, version control, and technical documentation.

### Machine learning and artificial intelligence

- Statistical feature engineering and logistic-regression baselines;
- Deep-learning development with PyTorch;
- 1D CNNs, residual dilated TCNs, BiLSTMs, compact time-series Transformers, MiniROCKET, multi-branch architectures, and multimodal feature fusion;
- Class-imbalance handling, early stopping, regularisation, and repeated random-seed experiments;
- Participant-level cross-validation, data-leakage prevention, and generalisation assessment;
- Leave-one-sensor-out retraining, compact-sensor design, learning-curve diagnosis, and overfitting analysis;
- Comparative experimentation, error analysis, limitation identification, and model interpretation.

### Biosignals and experimental research

- Processing of IMU, sEMG, plantar-pressure, and motion-capture data;
- Human-movement data collection and on-site quality checks;
- Sensor-channel auditing, dropped-sample detection, and signal-quality analysis;
- Jump-event detection, movement-phase localisation, and event-centred modelling;
- Translation of biosignal analysis into evidence relevant to exoskeleton product tuning.

## 13. Verifiable project outputs

- Full English report: [`PROJECT_REPORT_EN.md`](PROJECT_REPORT_EN.md)
- Full Chinese report: [`PROJECT_REPORT_ZH.md`](PROJECT_REPORT_ZH.md)
- Data audit: [English](DATA_AUDIT.md) | [中文](DATA_AUDIT_ZH.md)
- Data dictionary: [English](DATA_DICTIONARY.md) | [中文](DATA_DICTIONARY_ZH.md)
- Processing pipeline: [English](PIPELINE.md) | [中文](PIPELINE_ZH.md)
- Model-validity review: [English](MODEL_VALIDITY.md) | [中文](MODEL_VALIDITY_ZH.md)
- Sensor-placement photograph and evidence: [English](SENSOR_PLACEMENT.md) | [中文](SENSOR_PLACEMENT_ZH.md)
- Six-participant pilot results: [English](../results/PILOT_RESULTS.md) | [中文](../results/PILOT_RESULTS_ZH.md)
- Full model comparison: [`../results/COLAB_FULL_BENCHMARK.md`](../results/COLAB_FULL_BENCHMARK.md)
- IMU-event results: [`../results/IMU_EVENT_BENCHMARK.md`](../results/IMU_EVENT_BENCHMARK.md)
- Sensor and compact-configuration results: [`../results/SENSOR_ABLATION.md`](../results/SENSOR_ABLATION.md), [`../results/COMPACT_SENSOR_CONFIGURATIONS.md`](../results/COMPACT_SENSOR_CONFIGURATIONS.md)
- Transformer benchmark and overfitting analysis: [`../results/TRANSFORMER_BENCHMARK.md`](../results/TRANSFORMER_BENCHMARK.md)
- Core implementation: [`../src/exojump/`](../src/exojump/)
- Executable scripts: [`../scripts/`](../scripts/)

> Scope note: the current evidence supports offline classification of vertical and long jumps in pre-segmented experimental trials. Continuous real-time recognition and automatic parameter optimisation have not yet been validated and are therefore not claimed as completed outcomes.
