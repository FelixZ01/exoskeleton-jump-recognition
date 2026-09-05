# Local data layout

Human-participant data are excluded from version control. A local working copy may use:

```text
data/
├── raw/                  Immutable exports from acquisition devices
├── interim/              Parsed, timestamp-normalised, and aligned files
├── processed/            Imputed signals and model-ready windows
├── private_manifests/    File-level inventory containing local paths/IDs
└── manifests/            Aggregate, anonymised metadata safe to version
```

Do not rename or overwrite raw files. Generate derived files in `interim/` or `processed/` and record the command/configuration used.
