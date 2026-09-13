# B24.3 bounded EPP321 Pilot16 refinement

Status: **AUTHORIZED — PILOT16 REFINEMENT ONLY**

This stage follows the completed B24.3 Pilot16.  The original pilot showed that
current EPP can create a qualitatively better basin (notably D/30740) but can
also destroy a good independent-root basin under aggressive 4->2 pruning
(notably D/11775).  The planner/user authorized exactly one bounded refinement
before deciding whether to scale to the remaining 64 development images.

## Frozen change

The only main-method change is a more conservative survival schedule with the
same checkpoints and the same measurement-only trailing-32 LF score:

- transitions 0..71: 4 live roots, k=5;
- checkpoint 72: keep 3;
- transitions 72..147: 3 live roots, k=8;
- checkpoint 148: keep 2;
- transitions 148..299: 2 live roots, k=9;
- checkpoint 300: keep 1;
- projection start: fork 4 hard-phase descendants and continue with k=1.

The preprojection proposal work is exactly

`72*4*5 + 76*3*8 + 152*2*9 = 6000`,

so main EPP321 uses `6000 + 699*4 = 8796` proposal UNet evaluations and 8800
total UNet evaluations including the four root initializations.  This exactly
matches `NP4_INDEPENDENT` and the current EPP arm under the within-NP dominant
compute proxy.

Two ablations are frozen:

1. `NP_EPP_321_RANDOM_PRUNE`: same 4->3->2->1 schedule and 8800 total UNet
   evaluations, but survival is selected by a frozen domain hash independent
   of the measurement.
2. `NP_EPP_321_NO_REALLOCATION`: same measurement-based survival schedule but
   k remains 5 after pruning.  It uses 6900 total UNet evaluations and must be
   reported at its actual lower Work-FRE rather than as compute matched.

Exact machine-readable semantics are in
`configs/b24/b24_3_epp321_refinement.json`.

## Fixed-compute principle

The scientific endpoint is **not** merely to beat the older EPP variant.  The
project ultimately needs better recovery than established baselines/controls
at fixed or similar compute.

Within the NP family, total UNet forward evaluations are a defensible dominant
FLOP proxy because model, resolution, scheduler, and operator are identical.
Wall time is only an engineering diagnostic.

Across method families, do **not** equate one NP UNet evaluation with one DAPS
or SITCOM iteration/forward.  Before final cross-family claims, calibrate and
report method-specific FLOPs or a defensible forward/operator-equivalent work
accounting.  Distinguish DAPS-4-equivalent compute, SITCOM-4-equivalent compute,
and four-terminal candidate matching.

## Execution scope

Authorized:

- reuse the exact already-materialized 16 Pilot16 development measurements;
- run only `NP_EPP_321`, `NP_EPP_321_RANDOM_PRUNE`, and
  `NP_EPP_321_NO_REALLOCATION`;
- use the existing 10,240-MiB admission gate and 52,452-MiB hard ceiling;
- summarize selected/oracle PSNR, Good25 rescues/harms, class-conditional
  behavior, selector gaps, and exact work.

Not authorized:

- any new measurement for this refinement;
- the remaining 64 development images;
- any of the 305 confirmation images;
- C1-only extra development exposure;
- changing checkpoint locations, score definition, root seeds, or terminal
  selector after observing EPP321 results.

After the 16-image refinement completes, return the aggregate evidence before
any larger development run.
