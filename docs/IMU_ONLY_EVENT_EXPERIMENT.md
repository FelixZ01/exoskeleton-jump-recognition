# IMU-only event detection experiment

## Research question

Can the system recognise standing long jump versus vertical jump without using
plantar pressure to locate the movement window?

The original benchmark used the minimum plantar-pressure signal as the event
anchor. That is useful offline, but it makes the preprocessing stage dependent
on a pressure insole. This follow-up experiment replaces that anchor with an
IMU-only detector and keeps the participant-held-out evaluation protocol.

## Event detector

For each recording, the pipeline:

1. interpolates short gaps in the 12 orientation channels;
2. unwraps angular signals across the ±180-degree boundary;
3. calculates sample-to-sample angular velocity;
4. robustly scales every channel using its median absolute deviation;
5. combines channels as angular-velocity energy and smooths the signal; and
6. selects the dominant energy peak away from recording boundaries.

No pressure or sEMG channel participates in event localisation. Three 2-second
windows centred at −250, 0 and +250 ms around the detected event are retained
when valid.

## Local validation

Across 222 matched recordings, the IMU anchor had a median absolute offset of
162.5 ms relative to the pressure-derived reference. 70.3% of anchors were
within 250 ms and 90.1% were within 500 ms. These values are a detector
agreement check rather than classification performance.

## Formal benchmark results

The cloud experiment trains CNN and TCN classifiers using only IMU inputs, with
three random seeds, up to 30 epochs, early stopping, recording-level prediction
aggregation, complete-participant holdout, and participant-clustered bootstrap
confidence intervals. The key comparison is against the previous pressure-
centred, IMU-only CNN result (balanced accuracy 87.3% ± 3.0%).

Run the complete experiment with:

```bash
python scripts/run_imu_event_experiment.py \
  --dataset data/processed/jump_imu_event_windows.npz \
  --output outputs/imu_event \
  --device cuda
```

Completion is indicated by `outputs/imu_event/RUN_COMPLETE` and the line
`RUN_COMPLETE:` in the console. The output archive is `imu_event_results.zip`.

## Formal result

Using all 224 aligned sessions, the IMU-only event detector produced 647
windows. CNN reached 90.0% ± 1.3% balanced accuracy and TCN reached 90.9% ±
3.5%. The corresponding macro-F1 scores were 89.3% ± 1.6% and 90.5% ± 3.1%.
See the [full result table](../results/IMU_EVENT_BENCHMARK.md).
