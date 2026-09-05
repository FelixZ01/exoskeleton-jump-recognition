# Model validity review

## What the recovered prototype did

The legacy model script loaded six quality-selected IMU recordings. It assigned the folder rank (`motion_dataset_01` … `06`) as the class label, so every recording became its own class rather than representing the intended two movements. It then:

- instantiated a convolutional encoder but did not optimise its weights;
- calculated one prototype from each single-record class;
- predicted the same six records used to create the prototypes;
- reported this resubstitution accuracy as model performance.

This result is useful as a software proof of concept only. It is not evidence of jump-type recognition accuracy or generalisation.

## Minimum defensible evaluation

- Target labels: vertical jump vs long jump.
- Unit of independence: participant, not window.
- Split: leave-one-participant-out cross-validation, or at minimum a fully held-out participant.
- Fit preprocessing statistics using training participants only.
- Report: per-fold sample counts, balanced accuracy, macro F1, sensitivity/specificity, and confusion matrix.
- Compare: IMU-only, sEMG-only, early/late fusion, and a simple non-neural baseline.
- Add uncertainty: bootstrap confidence intervals at the participant/session level.
- Avoid selecting test recordings using the same quality score or outcome used during training.

With six participants, results should be described as a pilot study. More participants and repeated sessions would be required for a strong product or publication claim.
