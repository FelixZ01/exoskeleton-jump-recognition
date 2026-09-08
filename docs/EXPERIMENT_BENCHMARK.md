# Multi-model experiment protocol

## Purpose

This protocol extends the validated dual-branch CNN experiment without changing its core scientific question: can a model distinguish vertical and long jumps for a participant it has never seen during training?

The additional models are controls, not decorations. They test whether the reported improvement is specific to one architecture, whether recurrence is useful, and whether a compact non-neural model can match the deep-learning result.

## Implemented models

| Family | Model | Input |
|---|---|---|
| Statistical baseline | Logistic regression | Eight summary features per channel |
| Kernel baseline | RBF-SVM | Eight summary features per channel |
| Tree ensemble | Random forest | Eight summary features per channel |
| Time-series transform | MiniROCKET | Raw multivariate event windows |
| Convolutional network | Dual/single-branch 1D CNN | Raw event windows |
| Recurrent network | Bidirectional LSTM | Downsampled raw event windows |
| Temporal convolution | Residual dilated TCN | Raw event windows |
| Attention network | Compact time-series Transformer encoder | Strided raw event-window tokens |

Every neural architecture supports `imu`, `semg`, and `fusion` modes. The same saved NPZ, participant identities, session identities, window augmentation, and recording-level probability aggregation are used throughout.

The Transformer is deliberately compact for this six-participant pilot. A strided convolution reduces each 2,000-sample window to 250 tokens before two encoder layers (four heads, hidden width 64, feed-forward width 128). Dropout 0.4, AdamW weight decay `1e-4`, participant-level validation, early stopping, and three repeated seeds reduce—but cannot eliminate—the overfitting risk. It is treated as a comparison model, not assumed to be superior merely because it uses attention.

## Validity safeguards

- Complete participants, rather than windows, are held out for testing.
- Neural training uses another complete participant for validation and early stopping.
- Preprocessing statistics are not transferred from test participants into training in `train_global` mode.
- Multiple windows from one trial are aggregated before scoring.
- Fixed model settings are used across outer test participants.
- Results include balanced accuracy, macro-F1, confusion matrices, model size, training time, per-participant folds, and participant-clustered 95% bootstrap intervals.
- Repeated seeds measure neural optimisation variability; they do not create additional independent participants.

## Experiment workflow

### 1. Create the event-centred dataset

```bash
python scripts/build_dataset.py \
  --aligned-root /path/to/imputed_joint_angles \
  --output data/processed/jump_event_windows.npz \
  --window-size 2000 \
  --window-mode pressure_event
```

### 2. Classical and strong transform baselines

```bash
python scripts/benchmark_classical_models.py \
  --dataset data/processed/jump_event_windows.npz \
  --output outputs/classical_benchmark.json \
  --models logistic,svm,random_forest,minirocket \
  --modalities fusion
```

`MiniROCKET` is optional because it requires `sktime`. Install all modelling dependencies with:

```bash
pip install -e '.[analysis,deep-learning,time-series]'
```

### 3. Architecture comparison

```bash
python scripts/benchmark_deep_models.py \
  --dataset data/processed/jump_event_windows.npz \
  --output outputs/deep_architectures \
  --architectures cnn,bilstm,tcn,transformer \
  --modalities fusion \
  --seeds 42,7,123 \
  --device auto
```

For a focused formal Transformer comparison on full IMU, IMU+sEMG fusion, and the compact R3+R4 sensor configuration:

```bash
python scripts/run_transformer_experiment.py \
  --dataset data/processed/jump_imu_event_windows.npz \
  --output outputs/transformer_formal \
  --device auto
```

The command writes per-fold learning histories, checkpoints, aggregate metrics, `transformer_summary.json`, a `RUN_COMPLETE` marker, and a downloadable ZIP archive. Mean best epoch versus completed epochs is retained as an additional overfitting/early-stopping diagnostic.

### 4. Modality ablation

The fusion CNN is already produced above, so only the two single modalities need to be added:

```bash
python scripts/benchmark_deep_models.py \
  --dataset data/processed/jump_event_windows.npz \
  --output outputs/cnn_modality_ablation \
  --architectures cnn \
  --modalities imu,semg \
  --seeds 42,7,123 \
  --device auto
```

### 5. Occlusion interpretation

Run this for each held-out fold of the selected final model:

```bash
python scripts/evaluate_occlusion.py \
  --dataset data/processed/jump_event_windows.npz \
  --model-dir outputs/deep_architectures/tcn/fusion/seed_42/fold_P01 \
  --output outputs/occlusion/tcn_fusion_seed42_P01.json \
  --device auto
```

Zero occlusion is performed after normalisation. A positive performance drop indicates that the removed channel or time region supported classification; it is an association with the model prediction, not a causal physiological conclusion.

### 6. Generate a comparison table

```bash
python scripts/summarize_benchmarks.py \
  --classical outputs/classical_benchmark.json \
  --deep outputs/deep_architectures/deep_benchmark.json \
  --output outputs/MODEL_COMPARISON.md
```

## Cloud execution environment

The reference Colab workflow is provided in [`../notebooks/ExoJump_Cloud_Benchmark.ipynb`](../notebooks/ExoJump_Cloud_Benchmark.ipynb). It installs the public repository, validates the schema of an anonymised event-window dataset, runs the benchmark suite, and exports aggregate results. The formal experiment used three random seeds, up to 30 epochs, validation-participant early stopping, and 2,000 participant-clustered bootstrap resamples.

Public outputs contain only aggregate, de-identified metrics. Participant codes and within-participant trial identifiers are used solely to enforce leakage-resistant splits and recording-level aggregation.
