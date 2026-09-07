# Sensor-placement evidence from the experiment photograph

![Front view of the wearable-sensor setup](assets/sensor_setup_front.jpg)

## Directly supported observations

The repository owner confirmed that the black modules fixed by black straps are IMU nodes, while the beige elastic wraps secure the sEMG electrode regions. White spherical objects visible beside the limbs are motion-capture markers rather than IMUs. The exoskeleton was carried on the participant's back and is not visible in this front view.

Eight black IMU modules are visible in a bilateral arrangement, with four modules on each side:

| Anatomical position | Visible modules | Photograph-supported description |
|---|---:|---|
| Waist | 2 | One module on each side of the waist belt |
| Anterior distal thigh | 2 | One module on each thigh just above the knee |
| Anterior lower leg | 2 | One module on the front of each lower leg |
| Dorsum of foot | 2 | One module on the top of each foot |
| **Total** | **8** | Four positions per side, arranged bilaterally |

The photograph and acquisition recollection support a four-per-side configuration. They therefore supersede the earlier unsupported description of “six around the thigh, one on the lower leg, and one at the ankle.”

## What cannot be recovered from the photograph

The image contains no readable hardware ID or packet-slot label. It cannot establish which physical module corresponds to raw IMU slot 1–8. The four stable modelling groups R1–R4 are confirmed as four retained IMU orientation groups because each contains roll, pitch, and yaw. A four-per-side layout makes a unilateral retained chain plausible, but the data and photograph alone cannot establish whether all four came from one side, which side was retained, or the R1-to-R4 anatomical order.

The acquisition recollection indicates that the full arrangement consisted of left and right four-node waist-to-foot chains, while later modelling retained four stable nodes. The existing sensor-ablation result—R4 being most informative—is compatible with, but does not prove, a distal leg or foot position. The exact retained side and R1–R4 order must not be reported without device IDs, cable records, or a labelled placement protocol.

## sEMG placement and interpretation boundary

The eight sEMG channels were recorded on the right side. The confirmed channel-to-position map is:

| sEMG channels | Position |
|---|---|
| `Channel_1`–`Channel_6` | Six electrodes distributed circumferentially around the right thigh |
| `Channel_7` | Right anterolateral lower leg |
| `Channel_8` | Right medial ankle |

Beige wraps show the broad fixation regions, but individual muscle names are not available. Position mapping is therefore confirmed, while channel-to-muscle mapping remains unresolved. The photograph also cannot determine the precise placement of individual plantar-pressure channels.

The public image has no visible face and was copied without EXIF metadata. It is included as apparatus documentation, not as participant-identification data.
