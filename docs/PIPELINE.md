# Processing pipeline

```text
Acquisition exports
  ├── packed IMU serial CSV
  ├── raw and processed sEMG CSV
  ├── force/foot-pressure channels
  └── motion-capture and jump-distance/height records
          │
          ▼
1. IMU decoding and timestamp reconstruction
          │
          ▼
2. Session matching by participant, movement, and recording timestamp
          │
          ▼
3. Millisecond-grid alignment of IMU and sEMG
          │
          ▼
4. Joint-angle gap interpolation + quality flags
          │
          ▼
5. Fixed-length windows, labelled as vertical jump / long jump
          │
          ▼
6. Participant-held-out model training and evaluation
```

## Recovered processing history

The original scripts used a 1 ms global interval. The confirmed right-leg layout contained eight physical IMU nodes: six around the thigh, one on the lower leg, and one at the ankle. Formal acquisition payloads provided eight three-value slots. Historical processing retained four stable sensor groups and reordered them into R1–R4 roll/pitch/yaw; matching packed and converted rows implies the zero-based order `[4, 2, 1, 3]`. The physical R1–R4 position mapping remains unresolved and must not be invented.

The strict alignment prototype cropped the earlier modality, detected gaps from the IMU device clock, inserted rows for missing milliseconds, and forced equal lengths using the sEMG timeline. The cleaned implementation makes the common timeline explicit and records missingness rather than silently treating padded values as observed measurements.

## Recommended analysis unit

A session is identified by participant, movement (`tiaogao` or `tiaoyuan`), and recording timestamp. Windows from one participant must not appear in both training and test data. Splitting random windows would leak participant-specific motion and sensor-placement patterns. The cleaned training code uses separate training, validation, and test participants, class-weighted loss, per-window normalisation, early stopping, and recording-level probability aggregation.
