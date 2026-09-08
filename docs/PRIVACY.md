# Data privacy

I collected and processed human biomechanical and physiological signals in this project. I keep the original recordings, participant information, videos, body measurements, device logs, and acquisition timestamps outside the public repository.

The public version contains the processing and modelling code, anonymised data examples, aggregate manifests, experiment settings, and group-level results. I applied the following de-identification steps before publication:

- replaced participant identifiers with P01–P06;
- replaced acquisition timestamps with within-participant trial codes;
- removed local paths, device identifiers, logs, and personal metadata;
- excluded raw sensor files, trained checkpoints, videos, and body measurements;
- reported model performance only as aggregate metrics.

The `.gitignore` rules also prevent common participant-data formats and model checkpoints from entering the public codebase.

## Public apparatus photograph

I included one front-view photograph to document the sensor setup. The image does not show the participant's face, and I removed its EXIF metadata before publication. No other participant photographs or videos are included.
