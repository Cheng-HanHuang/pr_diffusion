# B24.3 zero-GPU reporting correction

## Planner decision

The planner explicitly accepted:

`STOP_B24_METHOD_REFINEMENT`

and kept confirmation locked.

The frozen stop decision is supported by the DEV80 method-development evidence:

| Method | Good25 /80 | Mean delta vs NP4 | Good25 rescues/harms |
| --- | ---: | ---: | ---: |
| NP4 selected | 55 | — | — |
| EPP321 | 50 | -1.423 dB | 5/10 |
| PE3 score | 52 | -0.883 dB | 1/4 |
| PE3 random | 53 | -0.889 dB | 2/4 |

Neither PE3 arm passed the prospectively frozen advancement gate.

## Verified source capsule

Original run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

Verified archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The planner verified that the archive matches its sidecar and that all 10 internal checksums pass. The original capsule must remain unchanged.

## Reporting error being corrected

The original closeout code computed `unique_good25_image_ids` over all 80 DEV images. Repository prose then described those global uniqueness counts as if they were uniqueness counts inside the 10-case subset where both baseline oracle candidate sets fail.

These are different quantities.

### All-DEV80 unique Good25 successes

Among the compared executable methods over all 80 DEV images:

- `DAPS1`: 0;
- `FRESH2_SELECTED`: 1 (`17146`);
- `SITCOM1`: 0;
- `NP4_SELECTED`: 2;
- `EPP321_SELECTED`: 2;
- `PE3_SCORE_SELECTED`: 0;
- `PE3_RANDOM_SELECTED`: 0.

### Shared-failure subset

Subset definition:

`DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB`.

The subset contains 10 images.

Good25 successes / exclusive successes among the compared executable methods are:

- `DAPS1`: `0 / 0`;
- `FRESH2_SELECTED`: `0 / 0`;
- `SITCOM1`: `0 / 0`;
- `NP4_SELECTED`: `2 / 1`;
- `EPP321_SELECTED`: `2 / 2`;
- `PE3_SCORE_SELECTED`: `1 / 0`;
- `PE3_RANDOM_SELECTED`: `0 / 0`.

PE3 score's success is image `34587`, shared with NP4. Fresh2's globally unique image `17146` lies outside the subset.

## Required logical check

Because Fresh2 chooses between DAPS reps 0 and 1, for every image:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`.

The corrected successor must verify this fail-closed. Therefore Fresh2 cannot rescue a failure of the same DAPS4 candidate set.

## Authorized correction scope

Authorized work is reporting-only and zero-GPU:

1. separate all-DEV80 uniqueness from shared-failure-subset success/exclusivity in machine-readable summaries;
2. correct repository result documentation and PR description;
3. add nested-oracle and subset-count/membership consistency checks;
4. preserve the original capsule and publish a corrected successor.

Not authorized:

- GPU execution;
- reconstruction;
- model/operator re-evaluation;
- metric recomputation;
- measurement generation;
- confirmation exposure;
- any new method/selector/score/checkpoint/schedule tuning;
- reopening the frozen stop verdict.

Any future Fresh2 study requires a separate prospective scientific question.
