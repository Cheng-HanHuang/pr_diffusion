# B24 — start here

B24 is a separate study in `Cheng-HanHuang/pr_diffusion`, isolated from B23. PR #37 must remain untouched.

## Immutable base

Signed B23.1 final head:

`27505e6328157ac9296c95dc5e611cbeef80de98`

B24 branch:

`codex/b24-bestof4-failure-sweep`

Draft PR: #38.

## Completed baseline/cohort evidence

- B24.0 exposure freeze PASS.
- B24.1 serial/concurrent equivalence + memory/throughput smoke PASS.
- B24.2 baseline screen complete at 7424 rows.
- Final screen census: A=6925, B=107, C=307, D=85.
- Primary ABC300 and secondary C1 cohorts frozen.
- Method roles frozen prospectively: DEV80=80 and confirmation=305; Pilot16=4 per screening stratum.

## Completed B24.3 development evidence

Completed stages:

- one-image NP branching calibration;
- Pilot16;
- bounded EPP321 refinement;
- full DEV80 with fresh DAPS-4 + pinned SITCOM-4 and NP4/EPP321;
- final protected-explorer refinement (`NP_PE3_SCORE`, `NP_PE3_RANDOM`);
- development-only cross-family dispatch-supported FLOP audit;
- original zero-GPU DEV80 closeout capsule;
- reporting-only corrected successor capsule.

## Final scientific verdict

`NP_EPP_321` failed to improve robustly over compute-matched NP4 on DEV80.

The final PE3 arms were evaluated under the prospectively frozen advancement gate:

- median paired PSNR delta vs NP4 >= 0 dB;
- Good25 count >= NP4;
- Good25 rescues >= harms;
- >=5 dB rescues >= harms.

Neither arm passed all four conditions. The accepted binding decision is:

`STOP_B24_METHOD_REFINEMENT`

Frozen DEV80 evidence:

| Method | Good25 /80 | Mean delta vs NP4 | Good25 rescues/harms |
| --- | ---: | ---: | ---: |
| NP4 selected | 55 | — | — |
| EPP321 | 50 | -1.423 dB | 5/10 |
| PE3 score | 52 | -0.883 dB | 1/4 |
| PE3 random | 53 | -0.889 dB | 2/4 |

No confirmation image has been exposed.

## Original zero-GPU closeout capsule

Original run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

Verified archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The planner verified the sidecar and all 10 internal checksums. The original capsule is preserved unchanged.

## Final reporting correction

A narrow reporting-scope error was identified after the original closeout: `unique_good25_image_ids` was computed over all DEV80 but described in repository prose as if those counts belonged to the 10-case shared-failure subset.

A reporting-only corrected successor completed from correction head:

`19d1054dcd7a73a471df08032619c05875b3d3c1`

Corrected run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`

The successor reported:

- `B24_3_ZERO_GPU_REPORTING_CORRECTION_TESTS_PASS`;
- `B24_3_ZERO_GPU_REPORTING_CORRECTION_ENV_PASS`;
- `B24_3_ZERO_GPU_REPORTING_CORRECTION_GATE_PASS`;
- `source_capsule_preserved=true`;
- `nested_oracle_inequality_pass=true`;
- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `confirmation_exposed=false`;
- `decision=STOP_B24_METHOD_REFINEMENT`.

It performed no reconstruction, no model/operator re-evaluation, and no Fresh2 metric recomputation.

### Correct all-DEV80 uniqueness

Among compared executable methods across all 80 DEV images:

- `DAPS1`: 0;
- `FRESH2_SELECTED`: 1 (`17146`);
- `SITCOM1`: 0;
- `NP4_SELECTED`: 2;
- `EPP321_SELECTED`: 2;
- `PE3_SCORE_SELECTED`: 0;
- `PE3_RANDOM_SELECTED`: 0.

### Correct shared-failure subset

Definition:

`DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB`.

There are exactly 10 images:

`18549, 27188, 30740, 34587, 38516, 47283, 51940, 56397, 61603, 67273`.

Good25 successes / exclusive successes among compared executable methods:

- `DAPS1`: `0 / 0`;
- `FRESH2_SELECTED`: `0 / 0`;
- `SITCOM1`: `0 / 0`;
- `NP4_SELECTED`: `2 / 1`;
- `EPP321_SELECTED`: `2 / 2`;
- `PE3_SCORE_SELECTED`: `1 / 0`;
- `PE3_RANDOM_SELECTED`: `0 / 0`.

The successor also verified on all 80 images:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`,

with zero violations at `1e-6 dB` tolerance. Therefore Fresh2 cannot rescue a failure of the same DAPS4 candidate set.

## Historical Fresh2 descriptive result

Fresh2 remains historical methodology, not a B24-fit method:

- Good25 `57/80`;
- mean PSNR `26.1596 dB`;
- median PSNR `29.9255 dB`;
- vs NP4 mean paired delta `+0.4249 dB`;
- vs NP4 median paired delta `+1.0939 dB`;
- vs NP4 Good25 rescues/harms `14/12`.

These are descriptive findings only. They do not reopen B24.

The DAPS2/Fresh2 dispatch-supported FLOP ratio to PE3_SCORE is approximately `0.4923x`, but this is not exact total compute or runtime; unsupported Fourier/custom work remains nonzero and separately inventoried.

## Final documentation

- `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_RESULT.md`
- `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION.md`
- `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md`
- `configs/b24/b24_3_zero_gpu_reporting_correction.json`
- `scripts/b24/correct_b24_3_zero_gpu_closeout.py`
- `scripts/b24/test_b24_3_zero_gpu_reporting_correction.py`
- `scripts/b24/launch_b24_3_zero_gpu_reporting_correction.sh`

## Final hard boundary

B24 is closed as a negative development result under the frozen gate.

Not authorized:

- any GPU work;
- any reconstruction or model/operator re-evaluation;
- any new measurement generation;
- any of the 305 confirmation images/measurements;
- C1-only extra exposure;
- any further NP/EPP/PE3/DPS method, selector, score, checkpoint, schedule, threshold, or root-budget tuning;
- changing the frozen PE3 stop verdict;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

Confirmation remains locked. Any future Fresh2 work must start from a separate prospective scientific question.
