# Sensor-placement evidence from the experiment photograph

![Front view of the wearable-sensor setup](assets/sensor_setup_front.jpg)

## Directly supported observations

During data collection, I used the black straps to secure the black IMU modules and the beige elastic wraps to secure the sEMG electrode regions. The white spherical objects beside the limbs are motion-capture markers rather than IMUs. The participant carried the exoskeleton on the back, outside the field of view in this front photograph.

Eight black IMU modules are visible in a bilateral arrangement, with four modules on each side:

| Anatomical position | Visible modules | Photograph-supported description |
|---|---:|---|
| Waist | 2 | One module on each side of the waist belt |
| Anterior distal thigh | 2 | One module on each thigh just above the knee |
| Anterior lower leg | 2 | One module on the front of each lower leg |
| Dorsum of foot | 2 | One module on the top of each foot |
| **Total** | **8** | Four positions per side, arranged bilaterally |

The photograph and acquisition recollection support a four-per-side configuration. They therefore supersede the earlier unsupported description of “six around the thigh, one on the lower leg, and one at the ankle.”

## Mapping boundaries

The image contains no readable hardware ID or packet-slot label, so the photograph alone cannot connect a physical module to raw IMU slot 1–8. Combining the historical conversion order, hip–knee–ankle constraints in the legacy interpolation code, the right-side sEMG setup, and node-ablation behaviour supports the working reconstruction **R1 waist/hip, R2 anterior distal thigh, R3 anterior lower leg, and R4 dorsum of foot**. The four groups most likely represent the right-side chain; confidence in the side and identifier mapping is moderate.

The acquisition record indicates that the full arrangement consisted of left and right four-node waist-to-foot chains, while later modelling used four stable nodes. R4 made the largest contribution in the sensor-ablation experiment, consistent with the pronounced orientation change expected at the foot during jumping. Because device IDs and labelled cable records were not preserved, the working reconstruction supports interpretation of the experiment but is not a direct hardware-ID verification.

## sEMG placement and interpretation boundary

The eight sEMG channels were recorded on the right side. The confirmed channel-to-position map is:

| sEMG channels | Position |
|---|---|
| `Channel_1`–`Channel_6` | Six electrodes distributed circumferentially around the right thigh |
| `Channel_7` | Right anterolateral lower leg |
| `Channel_8` | Right medial ankle |

Beige wraps show the broad fixation regions, but individual muscle names are not available. Position mapping is therefore confirmed, while channel-to-muscle mapping remains unresolved. The photograph also cannot determine the precise placement of individual plantar-pressure channels.

The public image has no visible face and was copied without EXIF metadata. It is included as apparatus documentation, not as participant-identification data.
