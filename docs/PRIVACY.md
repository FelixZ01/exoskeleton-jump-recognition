# Privacy, consent, and release checklist

This project contains human biomechanical and physiological recordings. Even initials or folder names can be identifying when combined with collection dates, body measurements, videos, and institutional context.

Before publishing data or a trained checkpoint:

1. Confirm that the institute and project supervisor permit external release.
2. Confirm that participant consent covers public data sharing and machine-learning reuse.
3. Remove names, initials, exact collection times, local paths, logs, videos, and device identifiers.
4. Replace participant identifiers with stable random codes stored in a private mapping.
5. Consider timestamp offsets rather than real dates/times.
6. Review body measurements and rare movement patterns for re-identification risk.
7. Document exclusions, preprocessing, licence, and an access/removal contact.

The repository `.gitignore` blocks common participant-data formats and model checkpoints by default. Do not bypass it without completing this checklist.

## Public apparatus photograph

`docs/assets/sensor_setup_front.jpg` is the sole public experiment photograph. It was added at the repository owner's explicit direction for apparatus documentation, contains no visible face, and was copied without EXIF metadata. It should not be used to infer participant identity. Its inclusion does not authorise publication of any other participant image, video, raw sensor file, or identifying record.
