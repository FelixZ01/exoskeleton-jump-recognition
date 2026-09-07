# Recovery notes and unresolved questions

## Confirmed from files

- Target sampling interval was configured as 1 ms.
- Six participant identifiers appear in the available dataset.
- Both vertical-jump and long-jump recordings exist for every detected participant.
- The placement used eight IMU nodes bilaterally: four per side at the waist, anterior distal thigh just above the knee, anterior lower leg, and dorsum of the foot.
- The right-side sEMG position map is confirmed: Channels 1–6 around the thigh, Channel 7 on the anterolateral lower leg, and Channel 8 on the medial ankle.
- The sEMG acquisition tables also contain 16 plantar-pressure channels and an aggregate foot-pressure signal.
- Converted IMU files contain four retained three-axis orientation groups.
- Aligned IMU and sEMG outputs share identical wall-clock timestamps row by row.
- The historical reports and output directories demonstrate completed alignment and imputation runs.

## Inferred and requiring confirmation

- The IMU payload-to-R1–R4 mapping was inferred from matching packed and converted rows as group order `[4, 2, 1, 3]` (zero based).
- The precise mapping from retained R1–R4 groups to the eight physical positions and the angle units was not documented.
- The exact muscle names underlying the confirmed sEMG channel positions were not documented.
- Two missing/failed source sessions and the one-file discrepancy between report and output require manual reconciliation.
- The available study dataset contains six participants and should be reported as a six-person pilot.

## Recommended next recovery targets

1. Device-ID or labelled cable/placement record resolving the retained side and R1–R4 node order; the available photograph establishes the overall layout but not the IDs.
2. Participant consent/ethics and data-ownership records.
3. Hardware packet format defining all eight IMU groups.
4. Experiment notebook linking jump-height/distance measurements to session timestamps.
5. Any original training notebook that produced the November figures; no matching source was found in the supplied PyCharm directory.
