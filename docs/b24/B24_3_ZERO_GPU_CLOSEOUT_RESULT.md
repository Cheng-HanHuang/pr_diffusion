# B24.3 zero-GPU development closeout result

## Status

The authorized zero-GPU development closeout completed successfully on 2026-09-14 from B24 scientific-run head:

`c3f13963cd267094231302bf9bc8d3a6e8c754c9`

Original PAC run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

Verified original archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The original capsule is preserved unchanged. Its 10 internal checksums and archive sidecar were independently verified by the planner.

A reporting-only corrected successor completed successfully at:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`

from correction head:

`19d1054dcd7a73a471df08032619c05875b3d3c1`.

The successor performed no reconstruction, no metric recomputation, no GPU work, no measurement generation, and no confirmation exposure. It corrected only the scope labeling of complementarity statistics.

Scope guards remained intact:

- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `reconstruction_performed=false`;
- `confirmation_exposed=false`;
- `source_capsule_preserved_unchanged=true`.

The binding development decision is accepted and remains:

`STOP_B24_METHOD_REFINEMENT`

This correction does not reopen method development and does not authorize confirmation exposure.

## Historical Fresh2 derivation

Fresh2 was derived exactly from stored DEV80 DAPS reps 0 and 1 using the pre-existing clean-free rule:

`choose rep1 iff exact_loss(rep1) < exact_loss(rep0) - 0.7; otherwise rep0`.

Ground truth was not used for selection.

DEV80 results remain valid:

- `FRESH2_SELECTED`: Good25 `57/80` (`0.7125`), mean PSNR `26.1596 dB`, median `29.9255 dB`;
- `DAPS2_ORACLE`: Good25 `59/80`, mean `26.8807 dB`, median `30.2805 dB`;
- Fresh2 selector gap to the two-trajectory oracle: mean `0.7212 dB`, median `0 dB`;
- Fresh2 accepted rep1 on `29/80` images.

Against DAPS1, Fresh2 had mean paired delta `+2.8692 dB`, Good25 rescues/harms `10/0`, and >=5 dB rescues/harms `14/0`.

Against executable NP4 selection, Fresh2 had mean paired delta `+0.4249 dB`, median `+1.0939 dB`, Good25 rescues/harms `14/12`, >=5 dB rescues/harms `14/13`, and PSNR wins/ties/losses `63/0/17`.

Against the DAPS4 oracle ceiling, Fresh2 remained below by mean `-2.6671 dB` and median `-0.0474 dB`, with Good25 harms/rescues `11/0`.

Fresh2 is historical project methodology, not a newly fit B24 method. These findings are descriptive and do not reopen B24.

## Final reporting correction

The original closeout implementation computed `unique_good25_image_ids` over **all 80 DEV images**. Repository prose had incorrectly described those global uniqueness counts as if they belonged to the 10-case shared-failure subset. This was a reporting-scope error only; it did not affect any reconstruction, PSNR, Good25 classification, Fresh2 comparison, PE3 gate, or stop decision.

### All-DEV80 unique Good25 successes

Among the compared executable methods over all 80 DEV images:

- `DAPS1`: `0`;
- `FRESH2_SELECTED`: `1` (`17146`);
- `SITCOM1`: `0`;
- `NP4_SELECTED`: `2`;
- `EPP321_SELECTED`: `2`;
- `PE3_SCORE_SELECTED`: `0`;
- `PE3_RANDOM_SELECTED`: `0`.

These are **all-DEV80 uniqueness counts** and must not be described as hard-subset uniqueness.

### Shared-failure subset

The subset where both fresh baseline oracle candidate sets fail Good25,

`DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB`,

contains exactly `10` images:

`18549, 27188, 30740, 34587, 38516, 47283, 51940, 56397, 61603, 67273`.

Good25 successes / exclusive successes among the compared executable methods inside this subset are:

- `DAPS1`: `0 / 0`;
- `FRESH2_SELECTED`: `0 / 0`;
- `SITCOM1`: `0 / 0`;
- `NP4_SELECTED`: `2 / 1`;
- `EPP321_SELECTED`: `2 / 2`;
- `PE3_SCORE_SELECTED`: `1 / 0`;
- `PE3_RANDOM_SELECTED`: `0 / 0`.

`PE3_SCORE_SELECTED` succeeds on image `34587`, but NP4 also succeeds there, so it is not exclusive. Fresh2's globally unique image `17146` is outside this subset because both DAPS4 and SITCOM4 succeed there.

## Nested DAPS candidate-set check

Because Fresh2 selects between DAPS reps 0 and 1, the following must hold per image in the common PSNR representation:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`.

The corrected successor checked this inequality on all 80 images and reported:

- pass: `true`;
- tolerance: `1e-6 dB`;
- violation count: `0`;
- maximum `Fresh2 - DAPS2 oracle`: `0.0 dB`;
- maximum `DAPS2 oracle - DAPS4 oracle`: `0.0 dB`.

Therefore Fresh2 cannot rescue a failure of the same DAPS4 candidate set. Its shared-failure-subset success count is necessarily `0` here.

The corrected successor also verified that recomputed shared-failure membership matches `HARD_SUBSET.csv` exactly.

## Frozen NP-method result

The frozen DEV80 evidence supporting the stop decision remains:

| Method | Good25 /80 | Mean delta vs NP4 | Good25 rescues/harms |
| --- | ---: | ---: | ---: |
| NP4 selected | 55 | — | — |
| EPP321 | 50 | -1.423 dB | 5/10 |
| PE3 score | 52 | -0.883 dB | 1/4 |
| PE3 random | 53 | -0.889 dB | 2/4 |

Neither PE3 arm passed the prospectively frozen advancement gate.

## Compute closeout

Dispatch-supported dynamic FLOPs from the development-only audit remain:

- DAPS1: `840267171895544`;
- DAPS2 / Fresh2 two-trajectory equivalent: `1680534343791088`;
- DAPS4 independent equivalent: `3361068687582176`;
- PE3_SCORE: `3413820886220800`;
- SITCOM1: `387934191616000`;
- SITCOM4 independent equivalent: `1551736766464000`.

Ratios to PE3_SCORE in the dispatch-supported metric are approximately DAPS1 `0.2461`, DAPS2/Fresh2 `0.4923`, DAPS4 `0.9845`, SITCOM1 `0.1136`, and SITCOM4 `0.4545`.

The `0.4923x` Fresh2/DAPS2 figure is **dispatch-supported FLOPs only**, not exact total computation or runtime. Unsupported Fourier/custom forward/backward work remains explicitly nonzero and separately inventoried. No exact total-FLOP equivalence is claimed.

## Corrected successor artifacts

Final corrected reporting result:

- `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md`.

Reporting-correction implementation/spec:

- `configs/b24/b24_3_zero_gpu_reporting_correction.json`;
- `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION.md`;
- `scripts/b24/correct_b24_3_zero_gpu_closeout.py`;
- `scripts/b24/test_b24_3_zero_gpu_reporting_correction.py`;
- `scripts/b24/launch_b24_3_zero_gpu_reporting_correction.sh`.

The successor capsule contains `CORRECTION_CHECKS.json` and `CORRECTION_METADATA.json`, plus corrected `COMPLEMENTARITY.json`, `B24_3_DEV_CLOSEOUT.json`, and `B24_3_DEV_CLOSEOUT.md`.

## Final conclusion

B24 closes as a negative development result. NP-native pruning/reallocation and the final protected-explorer variants did not robustly improve over independent NP4 at matched NP UNet work. Fresh2 remains a useful historical comparator with substantial per-image tradeoffs, but neither its descriptive DEV80 result nor its dispatch-supported compute ratio warrants reopening B24.

Any Fresh2 follow-up must begin with a separate prospective scientific question. No additional reconstruction is needed to close B24. Confirmation remains locked.
