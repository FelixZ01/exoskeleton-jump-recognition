# Public signal examples

This directory contains four small, identity-free examples extracted from the
model-ready event windows. They are included only to demonstrate the schema and
signal format for reviewers of the project.

- `vertical_jump_imu_example.csv`: 12 IMU orientation channels
- `long_jump_imu_example.csv`: 12 IMU orientation channels
- `vertical_jump_semg_example.csv`: eight auxiliary wearable channels
- `long_jump_semg_example.csv`: eight auxiliary wearable channels

Each file contains one 2-second window sampled at 1 kHz. Participant and trial
identifiers, absolute timestamps, raw recordings, and videos are not included.
These examples are not a redistributable research dataset and should not be
used to reproduce the reported benchmark.

Regenerate locally with:

```bash
python scripts/export_public_samples.py \
  --dataset data/processed/jump_event_windows.npz \
  --output data/samples
```
