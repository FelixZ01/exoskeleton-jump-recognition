# Data dictionary

## Normalised IMU table

| Column | Meaning |
|---|---|
| `First_Timestamp_ms` | Reconstructed wall-clock timestamp at millisecond precision |
| `Timestamp` | Device elapsed clock (`hours:minutes:seconds:milliseconds`) |
| `R1_Roll`, `R1_Pitch`, `R1_Yaw` | Three orientation channels for retained sensor group R1 |
| `R2_Roll`, `R2_Pitch`, `R2_Yaw` | Three orientation channels for retained sensor group R2 |
| `R3_Roll`, `R3_Pitch`, `R3_Yaw` | Three orientation channels for retained sensor group R3 |
| `R4_Roll`, `R4_Pitch`, `R4_Yaw` | Three orientation channels for retained sensor group R4 |

The available files do not establish the anatomical placement or physical units of R1–R4. They should be described as sensor orientation channels until hardware documentation confirms joint locations and whether values are degrees.

## Device-processed sEMG table

| Column group | Meaning |
|---|---|
| `Timestamp` | Recording timestamp; standardised to wall-clock milliseconds in interim data |
| `Channel_1` … `Channel_8` | Eight sEMG signal channels |
| `FP_CH41` … `FP_CH48` | First bank of pressure/force channels |
| `FP_CH51` … `FP_CH58` | Second bank of pressure/force channels |
| `count_foot` / `sum_foot` | Derived aggregate pressure/force value; naming varies between exports |

Channel-to-muscle and force-sensor placement metadata were not found. Avoid assigning muscle names or biomechanical interpretations without the sensor-placement protocol.

## Session metadata

| Field | Meaning |
|---|---|
| participant | Private identifier; public outputs use P01–P06 only |
| movement | `tiaogao` = vertical jump; `tiaoyuan` = long jump |
| session | Recording start timestamp in the source folder |
| modality | IMU, sEMG/force, motion capture, jump height, or jump distance |

## Quality-selection metadata

| Field | Meaning |
|---|---|
| `dataset_rank` | Rank produced by the historical heuristic |
| `total_score` | Composite heuristic score, not a prediction probability |
| `num_cycles` | Number of detected motion cycles used for scoring |
| `avg_intensity` | Mean motion-intensity feature |
| `avg_correlation` | Mean absolute cross-modal correlation feature |
| `avg_synchronization` | Peak-timing synchronisation feature |
