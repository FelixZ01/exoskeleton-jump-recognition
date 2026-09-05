# Recovered code map

The original PyCharm project was copied without changing its logic. Files were renamed only to make their purpose visible.

| Recovered file | Purpose | Status |
|---|---|---|
| `legacy/acquisition/imu_serial_capture.py` | IMU serial acquisition at a nominal 1 kHz | Requires hardware/COM configuration |
| `legacy/acquisition/semg_lightvista_capture.py` | sEMG/force-pressure acquisition | Requires vendor modules and LightVista |
| `legacy/preprocessing/alignment_strict.py` | Gap detection and millisecond alignment | Main historical alignment script |
| `legacy/preprocessing/alignment_audit.py` | Alignment verification/report generation | Historical audit; patched to create timestamped backups |
| `legacy/preprocessing/joint_angle_imputation.py` | Joint-angle interpolation and report | Main historical imputation script |
| `legacy/preprocessing/joint_angle_imputation_experimental.py` | Experimental extended imputation | Prototype |
| `legacy/selection/motion_quality_selection.py` | Motion-cycle scoring and candidate selection | Selected six candidate sessions in current archive |
| `legacy/modeling/imu_prototypical_unvalidated.py` | Early IMU prototype classifier | Not a valid generalisation evaluation |
| `legacy/analysis/` | Schema, timestamp, quality, and multimodal analysis | Exploratory scripts |
| `legacy/mocap/` | Motion-capture exploration | Exploratory script |
| `legacy/plotting/` | Session plots | Exploratory scripts |

All legacy files retain hard-coded local paths so the provenance is visible. Use `scripts/` and `src/exojump/` for new runs.
