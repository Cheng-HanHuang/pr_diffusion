# B21.12 failure and selector atlas decision

## Status

Completed successfully on the frozen B21.11 outputs. No GPU solver run or policy change occurred.

## Integrity

- atlas rows: `27`
- Fresh2 rescues: `12`
- protected Fresh1 successes: `7`
- official Fresh2 failures: `8`
- per-case panels: `27`
- group sheets: `3`
- all required artifacts: present

## Offline 180-degree ambiguity result

Among the eight official raw-PSNR failures:

- selected candidate becomes good25 after ground-truth-assisted 180-degree alignment: `3`
- only the unselected candidate becomes good25 after alignment: `0`
- neither candidate reaches good25 under identity or 180-degree rotation: `5`

Thus the official B21.11 result remains `92/100`, while an offline ambiguity-aware interpretation separates three rotation/twin-ambiguity cases from five failures persistent under both candidate orientations.

## Visual taxonomy

Manual visual inspection of `persistent_failure.png` further separates the eight cases:

- pure selected-candidate 180-degree ambiguity: rows `71`, `82`, `83`;
- low-frequency chromatic/illumination bias with largely correct geometry: rows `15`, `58`;
- structured twin-mixture, double-exposure, or swirl ghosting: rows `33`, `51`;
- high-complexity saturated content with chromatic multi-exposure and alternate-run structural collapse: row `37`.

The detailed case-by-case interpretation is recorded in `docs/b21/b21_12_visual_failure_interpretation.md`.

## Interpretation

The exact-loss selector was not responsible for any official good25 miss. Three failures are attributable to the known phase-retrieval 180-degree ambiguity under the raw-PSNR convention. Five remain candidate-generation failures even after granting an oracle choice over both candidates and both orientations, but those five are not homogeneous: they include photometric/chromatic failures, structured wrong-basin mixtures, and one high-complexity/low-density collapse.

The ground-truth-assisted orientation audit and visual taxonomy are descriptive only. They are not deployable selectors and do not authorize changing the frozen runtime policy, threshold, restart count, or official benchmark score.

## Decision

- Mark B21.12 complete.
- Preserve the official B21.11 Fresh2 result as `92/100`.
- Report both the official raw-PSNR failure count (`8`) and the offline ambiguity decomposition (`3` rotation-resolvable, `5` persistent).
- Preserve the class-level visual taxonomy for manuscript failure analysis.
- Proceed next to a preregistered fixed-baseline comparison; do not resume Fresh2 method tuning on the final panel.
