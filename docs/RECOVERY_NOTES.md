# Recovery notes and unresolved questions

## Confirmed from files

- Target sampling interval was configured as 1 ms.
- Six participant identifiers appear in the available dataset.
- Both vertical-jump and long-jump recordings exist for every detected participant.
- Converted IMU files contain four retained three-axis orientation groups.
- Aligned IMU and sEMG outputs share identical wall-clock timestamps row by row.
- The historical reports and output directories demonstrate completed alignment and imputation runs.

## Inferred and requiring confirmation

- The IMU payload-to-R1–R4 mapping was inferred from matching packed and converted rows as group order `[4, 2, 1, 3]` (zero based).
- R1–R4 anatomical placement and angle units were not documented.
- The exact muscle placement of sEMG channels 1–8 was not documented.
- Two missing/failed source sessions and the one-file discrepancy between report and output require manual reconciliation.
- Earlier career notes mentioned eight people, while this archive contains six participant folders. Do not claim eight complete participants for this dataset without locating the missing recordings and consent records.

## Recommended next recovery targets

1. Sensor placement diagram or experimental protocol.
2. Participant consent/ethics and data-ownership records.
3. Hardware packet format defining all eight IMU groups.
4. Missing two participant datasets, if they exist.
5. Experiment notebook linking jump-height/distance measurements to session timestamps.
6. Any original training notebook that produced the November figures; no matching source was found in the supplied PyCharm directory.
