# Recovered data audit

Audit date: 2026-09-05. Source files were read without modifying the original directory. Videos, macOS metadata, and archive duplicates were not copied into the repository.

## Inventory

| Item | Observed value |
|---|---:|
| Visible files after metadata exclusion | 4,609 |
| Total bytes including two videos | 2,392,259,505 |
| Non-video research files | 4,607 |
| Non-video research-file bytes | 2,301,861,285 (approximately 2.30 GB) |
| CSV files | 2,483 |
| PNG files | 1,160 |
| Log files | 948 |
| Videos skipped | 2 files, 90,398,220 bytes |
| Participant folders detected | 6 |
| Confirmed right-leg IMU nodes | 8 (thigh 6, lower leg 1, ankle 1) |
| Movement classes | 2 |
| Final aligned IMU/sEMG pairs present | 224 |

The directory also contains two ZIP archives, including a duplicate packaged copy of `dataset_all`, so even the non-video archive size must not be interpreted as a deduplicated data volume. The 2.30 GB figure describes the non-video files that were inventoried, not the amount of unique model input.

## IMU configuration audit

The six-person formal dataset contains 227 packed IMU recordings with eight payload slots. In 27 recordings from P01, all eight slots contained non-zero values; in the other 200 recordings, four slots contained stable non-zero values while the remaining slots were zero-filled. The historical conversion therefore retained four nodes as R1–R4. A separate set of 20 earlier pilot recordings used a six-slot payload and is not counted as an additional participant in the formal dataset.

## Data layers

### Source and device-processed exports

- `dataset_all/IMU_data`: 227 packed IMU CSV recordings.
- `dataset_all/sEMG_data`: 454 sEMG CSV files (raw and device-processed forms for 227 sessions), plus plots/logs.
- `dataset_all/jump_height_data`: jump-height measurements and associated files.
- `dataset_all/jump_distance_data`: six participant-level text files.
- `dataset_all/body_data`: body-measurement text file.
- `ZZF_20251021`: motion-capture calibration/trial CSV exports.
- `zzf`: earlier personal acquisition runs.

“Raw” is therefore modality-specific: `raw_data_*.csv` is the lowest-level sEMG export found, while `processed_data_*.csv` is already device/software processed but precedes cross-modal alignment.

### Interim processing

- `data_process/IMU_data_copy`: unpacked 12-channel IMU tables.
- `data_process/sEMG_data_copy`: sEMG tables with full millisecond timestamps.
- `data_process/aligned_data_strict`: 224 aligned IMU files plus 224 aligned sEMG files.

### Processed/model-candidate data

- `data_process/imputed_joint_angles`: 224 IMU and 224 corresponding sEMG files.
- `data_process/cnn_optimal_data`: six selected sessions with IMU, sEMG, metadata, and quality features.

## Report reconciliation

The historical alignment report records 227 attempted pairs: 223 “perfect”, two repaired, and two failed. The final strict-alignment directory contains 224 pairs, one fewer than the 225 nominal successes. This discrepancy should be resolved before a paper by comparing the report against the file-level private manifest.

The imputation report says 448 files were scanned, with 224 processed and 224 skipped. Inspection shows that the 448 files comprise 224 IMU and 224 sEMG tables; the sEMG files were skipped because they contain no joint-angle gaps. Thus “224 successfully imputed” is the meaningful IMU count.

## Class and participant balance

The aligned session counts are stored in `data/manifests/aligned_session_counts.csv`. Four participants (P03–P06) have 20 recordings per movement. Two folders are less complete: P01 has 18 vertical-jump and 7 long-jump sessions; P02 has 19 vertical-jump and 20 long-jump sessions.

The six “optimal” model candidates are strongly imbalanced: five vertical-jump recordings and one long-jump recording. They are suitable for qualitative inspection, not for training or evaluating a two-class classifier by themselves.

## Code recovery

Twenty Python files were recovered from a separate PyCharm project and copied into `legacy/`. No Python files were present in the 2.2 GB experiment data directory. The cleaned pipeline documents where behaviour was inferred from converted data rather than directly recovered from source code.
