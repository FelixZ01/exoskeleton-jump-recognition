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
5. Plantar-pressure or IMU event detection and fixed-length jump windows
          │
          ▼
6. Participant-held-out model training and evaluation
          │
          ▼
7. Multi-model, modality, and sensor-configuration evaluation
```

## Recovered processing history

The original scripts used a 1 ms global interval. The confirmed layout contained eight physical IMU nodes arranged bilaterally, four per side: waist, anterior distal thigh just above the knee, anterior lower leg, and dorsum of the foot. Formal acquisition payloads provided eight three-value slots. Historical processing retained four stable sensor groups and reordered them into R1–R4 roll/pitch/yaw; matching packed and converted rows implies the zero-based order `[4, 2, 1, 3]`. The retained side and physical R1–R4 order remain unresolved and must not be invented.

The strict alignment prototype cropped the earlier modality, detected gaps from the IMU device clock, inserted rows for missing milliseconds, and forced equal lengths using the sEMG timeline. The cleaned implementation makes the common timeline explicit and records missingness rather than silently treating padded values as observed measurements.

The device-processed sEMG table also contains 16 plantar-pressure channels and an aggregate `sum_foot`/`count_foot` value. The cleaned dataset builder can smooth this aggregate signal, locate the low-pressure flight phase, and centre model windows on the event. An IMU-only event locator is also implemented to reduce pressure dependence at inference. Pressure is used as an event anchor unless explicitly included as a model input; it is not treated as an sEMG channel.

## Recommended analysis unit

A session is identified by participant, movement (`tiaogao` or `tiaoyuan`), and recording timestamp. Windows from one participant must not appear in both training and test data. Splitting random windows would leak participant-specific motion and sensor-placement patterns. The cleaned training code uses separate training, validation, and test participants, class-weighted loss, per-window normalisation, early stopping, recording-level probability aggregation, and repeated seeds. Formal evaluation covers classical models, MiniROCKET, CNN, BiLSTM, TCN, and a compact Transformer, together with modality, leave-one-IMU-node-out, and compact-sensor comparisons.
