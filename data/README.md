# Local data layout

Human-participant data are excluded from version control. The formal pilot dataset contains six participants. Device-processed sEMG tables also include 16 plantar-pressure channels and an aggregate `sum_foot`/`count_foot` signal; these are separate modalities even though they share a CSV file.

A local working copy may use:

```text
data/
├── raw/                  Immutable exports from acquisition devices
├── interim/              Parsed, timestamp-normalised, and aligned files
├── processed/            Imputed signals and model-ready windows
├── private_manifests/    File-level inventory containing local paths/IDs
└── manifests/            Aggregate, anonymised metadata safe to version
```

Raw files are kept unchanged. Derived files are written to `interim/` or `processed/`, with the corresponding command and configuration recorded for reproducibility.
