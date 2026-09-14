# B24.3 final planner return

## Executor status

B24 development, zero-GPU closeout, and the reporting-only correction are complete. This return is archival/reporting only and does not authorize any new execution.

The planner accepted the binding decision:

`STOP_B24_METHOD_REFINEMENT`

Confirmation remains locked. No confirmation image or confirmation measurement was exposed.

## PR / branch identity

- repository: `Cheng-HanHuang/pr_diffusion`
- draft PR: `#38 — B24: frozen failure cohorts + NP-native branching development`
- branch: `codex/b24-bestof4-failure-sweep`
- base: `codex/b23-execution`
- signed B23.1 base SHA: `27505e6328157ac9296c95dc5e611cbeef80de98`
- PR remains draft, open, and unmerged
- PR #37 was not modified

## Frozen DEV80 stop evidence

| Method | Good25 /80 | Mean delta vs NP4 | Good25 rescues/harms |
| --- | ---: | ---: | ---: |
| NP4 selected | 55 | — | — |
| EPP321 | 50 | -1.423 dB | 5/10 |
| PE3 score | 52 | -0.883 dB | 1/4 |
| PE3 random | 53 | -0.889 dB | 2/4 |

Neither PE3 arm passed the prospectively frozen advancement gate. Method refinement therefore remains stopped.

## Original closeout capsule

Original run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

Original archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z.tar.gz`

Verified original archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The planner verified the original sidecar and all ten internal checksums. The original capsule remains preserved unchanged.

## Corrected successor capsule

Reporting-only successor run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`

Corrected archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z.tar.gz`

Corrected successor archive SHA-256, independently verified from the uploaded archive bytes:

`fdd9bbc1b3d3463e9f3c5211bed0254e9f985e93168f7d2ba742f2bcd58da84d`

The reporting-only successor completed with:

- `B24_3_ZERO_GPU_REPORTING_CORRECTION_TESTS_PASS`;
- `B24_3_ZERO_GPU_REPORTING_CORRECTION_ENV_PASS`;
- `B24_3_ZERO_GPU_REPORTING_CORRECTION_GATE_PASS`;
- `source_capsule_preserved_unchanged=true`;
- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `reconstruction_performed=false`;
- `confirmation_exposed=false`;
- `decision=STOP_B24_METHOD_REFINEMENT`.

No Fresh2 metric was recomputed in the correction stage.

## Corrected complementarity reporting

All-DEV80 unique Good25 successes among the compared executable methods:

- DAPS1: 0
- Fresh2: 1 (`17146`)
- SITCOM1: 0
- NP4: 2
- EPP321: 2
- PE3 score: 0
- PE3 random: 0

The shared-failure subset is defined by `DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB` and contains exactly 10 images.

Good25 successes / exclusive successes in that subset:

- DAPS1: `0 / 0`
- Fresh2: `0 / 0`
- SITCOM1: `0 / 0`
- NP4: `2 / 1`
- EPP321: `2 / 2`
- PE3 score: `1 / 0`
- PE3 random: `0 / 0`

The correction verified on all 80 development images:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`

with zero violations at `1e-6 dB` tolerance. Therefore Fresh2 cannot rescue failure of the containing DAPS4 candidate set.

## Fresh2 interpretation

Historical Fresh2 remains a descriptive comparator only:

- Good25: `57/80`
- mean PSNR: `26.1596 dB`
- median PSNR: `29.9255 dB`
- versus NP4 mean paired delta: `+0.4249 dB`
- versus NP4 median paired delta: `+1.0939 dB`
- versus NP4 Good25 rescues/harms: `14/12`

Fresh2 was not fit on B24 and does not reopen B24. Any future Fresh2 work must begin from a separate prospective scientific question.

## Compute interpretation

The DAPS2/Fresh2 two-trajectory equivalent is approximately `0.4923x` PE3_SCORE in **dispatch-supported dynamic FLOPs only**. This is not an exact total-compute or runtime ratio. Unsupported Fourier/custom forward/backward work is explicitly nonzero and remains separately inventoried.

## Final executor return

B24 can close as a negative development result. No further reconstruction or confirmation execution is required or authorized for B24. The executor returns control to the planner with the stop verdict preserved and confirmation locked.
