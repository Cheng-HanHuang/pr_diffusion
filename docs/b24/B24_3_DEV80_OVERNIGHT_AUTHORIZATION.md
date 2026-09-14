# B24.3 DEV80 overnight authorization

## Authorization

The user/planner explicitly authorized the **DEV80 overnight stage** after reviewing the completed EPP321 Pilot16 refinement.

This stage is development-only. It does not expose any of the 305 frozen confirmation images.

## Scientific endpoint

The final scientific target is not merely to beat an older EPP variant. The project must establish whether the frozen NP method improves recovery relative to established baselines at fixed or similar compute.

For this stage:

- NP4 and NP_EPP_321 are exactly matched at 8,800 total same-model UNet forward evaluations per image.
- Wall time is an engineering diagnostic, not the scientific matching variable.
- Raw iteration counts are **not** equated across NP, DAPS and SITCOM. The run preserves the native work/timing ledger needed for a later method-specific FLOP or forward/operator-equivalent accounting.

## Frozen DEV80 panel

The already-frozen method-role manifest supplies exactly 80 development images:

- A: 20
- B: 20
- C: 20
- D: 20

The original A/B/C/D label remains only a **screening stratum**. Fresh baseline outcomes are recomputed on the new development measurement and no image is dropped when its difficulty changes.

Pilot16 is a subset of DEV80. Its existing locked measurements and completed NP results are reused. The other 64 development images receive exactly one locked measurement from their already-frozen `dev_measurement_seed`.

## Fresh external baselines

All 80 development images receive fresh, same-measurement reruns of:

- DAPS-4: four independent native DAPS trajectories;
- pinned SITCOM-4: four independent native pinned-public SITCOM trajectories.

The four solver seeds per image/method are prospectively derived from fixed domains:

- `B24_DEV80_DAPS_SOLVER_V1`
- `B24_DEV80_SITCOM_SOLVER_V1`

Rep 0 is retained as DAPS-1/SITCOM-1. GT-best-of-four is an external candidate-set **oracle ceiling**, not a deployable selector.

## NP methods

The following methods are evaluated on all 80 development images. Pilot16 results are reused; only the remaining 64 are newly executed:

- `NP4_INDEPENDENT` — 8,800 total UNet evaluations;
- `NP_EPP_321` — 8,800 total UNet evaluations;
- `NP_EPP_321_RANDOM_PRUNE` — 8,800 total UNet evaluations;
- `NP_EPP_321_NO_REALLOCATION` — 6,900 total UNet evaluations.

No checkpoint, score, root-seed, selector, or EPP321 schedule change is authorized in this stage.

## Common evaluation representation

Cross-family PSNR is reported on a common raw-orientation 8-bit RGB representation:

- DAPS/SITCOM use their canonical saved RGB PNGs;
- NP terminal tensors are clamped, mapped to `[0,1]`, quantized to 8-bit RGB, and evaluated against the same quantized ground truth.

Native NP float-tensor metrics are retained as diagnostics.

For NP, the executable terminal is selected only by the already-frozen measurement-side selector. GT-best terminal is reported only as an offline oracle diagnostic.

## Decisive development outputs

The DEV80 summary must include at minimum:

- NP_EPP_321 vs NP4 paired selected/oracle PSNR deltas, wins/losses, Good25 rescues/harms and large basin switches;
- matched random-pruning and no-reallocation ablations;
- DAPS-1, SITCOM-1, DAPS-4 oracle and SITCOM-4 oracle performance;
- fresh A/B/C/D transition counts relative to screening strata;
- EPP321 Good25 rescues on measurements where both fresh DAPS-4 and SITCOM-4 fail;
- per-screening-stratum conditional results;
- an explicit compute/work ledger without claiming cross-family FLOP equivalence prematurely.

## Resource and failure policy

- hard B24 process/group ceiling: **52,452 MiB**;
- admission gate: **10,240 MiB**;
- physical GPU identity must match the frozen UUID map;
- never kill or evict another job;
- one task failure must not waste the rest of an overnight GPU assignment: the worker records that task as failed and continues later tasks;
- the final DEV80 gate remains closed until all 80 images have exact PASS completions.

## Not authorized

This stage does **not** authorize:

- any confirmation measurement or confirmation run;
- C1-only extra development exposure;
- additional method/checkpoint/score sweeps;
- changing DAPS/SITCOM source identities;
- merge/rebase/squash/retarget/force-push/history rewrite.
