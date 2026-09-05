/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# Multimodal Exoskeleton Jump Recognition

Reproducible research code for recognizing **vertical jump** and **long jump** movements from synchronised surface electromyography (sEMG) and inertial measurement unit (IMU) signals collected during civilian exoskeleton experiments.

> Status: research prototype. The repository contains code, aggregate metadata, and documentation. Human-participant recordings are intentionally excluded from Git.

## Project overview

The recovered experiment archive contains six identifiable participant folders and two movement classes. A total of 227 IMU/sEMG session pairs were reported during alignment; 224 paired sessions are present in the final aligned directory. The processing records show:

- 1 kHz target sampling for both modalities;
- conversion of packed IMU serial payloads into 12 joint-orientation channels (four sensors × roll/pitch/yaw);
- millisecond-level timestamp reconstruction and sEMG/IMU alignment;
- detection of IMU gaps and insertion of missing rows;
- joint-angle interpolation for 224 IMU recordings;
- quality-based selection of six candidate recordings for an early modelling experiment.

The cleaned pipeline in `src/exojump/` replaces hard-coded local paths with command-line arguments and treats the scientific task as **binary movement classification**, using participant-held-out evaluation to reduce identity leakage.

## Demonstrated engineering work

- Managed multi-device acquisition outputs spanning IMU, sEMG, foot-pressure, motion-capture, and jump-performance measurements.
- Reconstructed a millisecond timeline from device and wall clocks, detected dropped IMU samples, and paired 227 multimodal sessions.
- Produced 224 aligned session pairs and a documented joint-angle missing-data workflow.
- Built signal-quality, cross-modal correlation, PCA, and motion-cycle screening analyses.
- Reframed the modelling task around leakage-aware, participant-held-out evaluation for product-tuning and research use.

## Repository layout

```text
configs/                 Example pipeline configuration
data/manifests/          Public aggregate inventory (no participant initials)
docs/                    Audit, data dictionary, pipeline, and privacy notes
legacy/                  Recovered original scripts, renamed by function
scripts/                 Command-line entry points
src/exojump/             Cleaned and reusable pipeline implementation
tests/                   Lightweight unit tests for critical preprocessing
```

The `legacy/` directory preserves the original work history. Those files may contain hard-coded paths, prototype assumptions, and unavailable vendor dependencies; they are not the recommended entry points.

## Quick start

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[analysis,deep-learning]'
```

Audit a local copy of the dataset without copying participant data into Git:

```bash
python scripts/audit_data.py \
  --source /path/to/exoskeleton \
  --private-output data/private_manifests
```

Convert packed IMU files, align modalities, and interpolate missing joint angles:

```bash
python scripts/convert_imu.py --input data/raw/IMU_data --output data/interim/imu_normalized
python scripts/align_modalities.py --imu data/interim/imu_normalized --semg data/interim/semg_normalized --output data/interim/aligned
python scripts/impute_joint_angles.py --input data/interim/aligned --output data/processed/aligned
```

Build leakage-aware windows and train the dual-branch 1D CNN:

```bash
python scripts/build_dataset.py --aligned-root data/processed/aligned --output data/processed/jump_windows.npz
python scripts/train_dual_branch_cnn.py --dataset data/processed/jump_windows.npz --output outputs/baseline
```

## Scientific limitations

The recovered checkpoint and result figures are not presented as validated performance. The legacy prototype used one selected recording per synthetic class, computed prototypes with an untrained encoder, and evaluated on the same records used to create those prototypes. This can produce a trivially high apparent accuracy and does not measure generalisation to unseen people.

For defensible results, use all eligible sessions, label by movement type, hold out entire participants, report balanced accuracy/F1/confusion matrices, and repeat evaluation across participant folds. See [docs/MODEL_VALIDITY.md](docs/MODEL_VALIDITY.md).

## Legacy exploratory analysis

![Legacy multimodal analysis](docs/assets/multimodal_analysis_legacy.png)

This figure is retained as evidence of exploratory work on one aligned recording. Its correlation, PCA, and SNR values are illustrative and must not be interpreted as population-level or validated model results.

## Data access and ethics

Raw sEMG, IMU, motion-capture, body measurements, logs, videos, and participant initials are not included. Before any data release, confirm participant consent, institutional ownership, ethics requirements, and de-identification. See [docs/PRIVACY.md](docs/PRIVACY.md).

## 中文说明

本仓库整理了民用外骨骼实验中的 sEMG 与 IMU 多模态数据处理代码，目标任务为跳高/跳远动作识别。原始人体实验数据默认不上传 GitHub；仓库保留可复现代码、匿名汇总清单和方法说明。旧代码完整保存在 `legacy/`，主流程位于 `src/exojump/`。
