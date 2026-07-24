# B21.12 visual interpretation of the eight official Fresh2 failures

This document records a visual, pixel-structure interpretation of `persistent_failure.png`. It is descriptive only. The official B21.11 raw-PSNR result remains 92/100, and the ground-truth-assisted 180-degree alignment does not define a deployable runtime correction.

## High-level decomposition

The eight official failures are visually heterogeneous:

- three are dominated by the exact 180-degree phase-retrieval ambiguity;
- two preserve facial geometry but contain broad low-spatial-frequency chromatic/illumination errors;
- two are structured twin/ghost mixtures with duplicated or spatially warped facial content;
- one combines unusually complex saturated content with severe color-channel-like ghosting and an alternate-run structural collapse.

This supports a class-level interpretation rather than one generic “bad reconstruction” category.

## Case-by-case labels

| row | image | primary visual category | secondary category | interpretation |
|---:|---:|---|---|---|
| 15 | 65553 | low-frequency chromatic/illumination bias | geometry preserved; measurement-loss near-tie | Both candidates preserve the face, pose, edges, and most local texture. The main error is a broad purple/blue/green color field over skin, hair, and background, with altered skin tone and illumination. Arm 2 is visually closer and has higher PSNR, but the exact losses are almost tied, so the frozen margin rule retains arm 1. This is not a good25 selector failure because both candidates remain below 25 dB. |
| 33 | 67293 | structured twin-mixture / swirl ghosting | global geometric warp | Sunglasses, eyes, mouth, hair, and facial outline are repeatedly smeared along curved trajectories. Local patches remain face-like, but their global spatial arrangement is wrong. A 180-degree rotation slightly helps but does not undo the mixed/warped phase basin. |
| 37 | 65003 | chromatic multi-exposure / channel-like ghosting | alternate-run structural collapse; high-complexity content | Arm 1 still contains the correct face and ornate headwear, but multiple colored copies are offset over each other. Arm 2 loses coherent facial structure and becomes a highly saturated texture collage. The dense multicolor accessories appear substantially harder for the face prior than ordinary FFHQ portraits. |
| 51 | 65365 | twin-image mixture / double exposure | severe cyan-magenta photometric corruption | Arm 1 is partly 180-degree inverted and contains circular or repeated overlays of the hat, glasses, and face. After rotation it becomes recognizable but remains strongly mixed. Arm 2 is upright but has duplicated facial contours and strong cyan/magenta color contamination. Both trajectories reach the correct semantic subject but not a single clean spatial solution. |
| 58 | 66889 | low-frequency chromatic/illumination bias | stable bad basin across restarts | Both candidates are nearly identical and preserve the face, hands, pose, and background geometry. The dominant error is a broad left-to-right purple/blue versus green/yellow cast, with incorrect skin tone and local contrast. Low candidate disagreement indicates a stable photometric failure rather than stochastic basin diversity. |
| 71 | 62908 | pure 180-degree/twin ambiguity | selected candidate is the better ambiguity representative | Both raw candidates are inverted. Rotating the selected arm 2 gives 28.90 dB, while rotated arm 1 reaches only 23.30 dB. The exact-loss selector selected the candidate that is genuinely better once the known ambiguity is resolved. |
| 82 | 66715 | pure 180-degree/twin ambiguity | both restarts converge to the same inverted solution | The two candidates are almost identical, both upside down, and both become high-quality reconstructions after rotation (29.61 and 29.78 dB). This is a measurement-symmetry outcome, not a semantic or candidate-generation failure modulo the equivalence class. |
| 83 | 68539 | pure 180-degree/twin ambiguity in the selected arm | alternate trajectory diffuse structural collapse | Selected arm 2 is a clean inverted reconstruction and reaches 31.90 dB after rotation. Arm 1 is a diffuse, multi-exposure collapse with no coherent face. This case simultaneously illustrates the value of the second restart and correct exact-loss selection. |

## Recommended manual CSV labels

| row | manual_primary_category | manual_secondary_category |
|---:|---|---|
| 15 | low-frequency chromatic/illumination bias | geometry preserved; exact-loss near-tie |
| 33 | structured twin-mixture / swirl ghosting | global geometric warp |
| 37 | chromatic multi-exposure / channel-like ghosting | alternate-run structural collapse; high-complexity content |
| 51 | twin-image mixture / double exposure | severe cyan-magenta photometric corruption |
| 58 | low-frequency chromatic/illumination bias | stable bad basin across restarts |
| 71 | rot180/twin ambiguity | selected candidate is better after alignment |
| 82 | rot180/twin ambiguity | both restarts share the inverted basin |
| 83 | rot180/twin ambiguity | alternate-run diffuse structural collapse |

## Class-level interpretation

The five failures persistent after the identity/rot180 oracle split into three different mechanisms:

1. **Photometric/chromatic failure with correct geometry**: rows 15 and 58.
2. **Structured wrong phase basin / twin mixture**: rows 33 and 51.
3. **High-complexity or low-density content with unstable prior behavior**: row 37.

This means that “five genuine candidate-generation failures” should not be read as five examples of the same attractor. A future method would likely need different interventions for chromatic consistency, twin-mixture rejection, and low-density complex imagery. The final B21.11 panel must not be used to tune those interventions.
