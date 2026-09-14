# B24.3 zero-GPU reporting-correction result

## Final status

The reporting-only successor capsule completed successfully on 2026-09-14 from correction head:

`19d1054dcd7a73a471df08032619c05875b3d3c1`

Corrected successor run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`

Corrected archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z.tar.gz`

Corrected archive sidecar:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z.tar.gz.sha256`

Source/original capsule archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The original capsule was verified against its sidecar and all ten internal checksums before the successor was generated. It was preserved unchanged.

## Scope guards

The successor reported:

- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `reconstruction_performed=false`;
- `confirmation_exposed=false`;
- `source_capsule_preserved_unchanged=true`;
- `decision=STOP_B24_METHOD_REFINEMENT`.

No Fresh2 metrics were recomputed. The successor only corrected reporting scopes from the already-published per-image DEV80 table.

## Corrected complementarity scopes

### All DEV80: unique Good25 successes

Among the compared executable methods over all 80 development images:

- `DAPS1`: 0;
- `FRESH2_SELECTED`: 1;
- `SITCOM1`: 0;
- `NP4_SELECTED`: 2;
- `EPP321_SELECTED`: 2;
- `PE3_SCORE_SELECTED`: 0;
- `PE3_RANDOM_SELECTED`: 0.

Fresh2's globally unique Good25 success is image `17146`.

### Shared-failure subset

The shared-failure subset is defined by

`DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB`.

It contains exactly 10 images:

`18549, 27188, 30740, 34587, 38516, 47283, 51940, 56397, 61603, 67273`.

Good25 successes / exclusive successes among the compared executable methods are:

- `DAPS1`: `0 / 0`;
- `FRESH2_SELECTED`: `0 / 0`;
- `SITCOM1`: `0 / 0`;
- `NP4_SELECTED`: `2 / 1`;
- `EPP321_SELECTED`: `2 / 2`;
- `PE3_SCORE_SELECTED`: `1 / 0`;
- `PE3_RANDOM_SELECTED`: `0 / 0`.

`PE3_SCORE_SELECTED` succeeds on image `34587`, shared with NP4, and is therefore not exclusive. Fresh2 has no success in this subset.

## Nested DAPS candidate-set invariant

The successor checked on every DEV80 image:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`.

Result:

- pass: `true`;
- tolerance: `1e-6 dB`;
- violation count: `0`;
- maximum `Fresh2 - DAPS2 oracle`: `0.0 dB`;
- maximum `DAPS2 oracle - DAPS4 oracle`: `0.0 dB`.

This makes the shared-failure Fresh2 count of zero structurally consistent: Fresh2 selects from DAPS reps 0/1 and therefore cannot rescue failure of the containing DAPS4 candidate set.

## Frozen method-development decision

The reporting correction does not alter the prospectively frozen DEV80 result:

| Method | Good25 /80 | Mean delta vs NP4 | Good25 rescues/harms |
| --- | ---: | ---: | ---: |
| NP4 selected | 55 | — | — |
| EPP321 | 50 | -1.423 dB | 5/10 |
| PE3 score | 52 | -0.883 dB | 1/4 |
| PE3 random | 53 | -0.889 dB | 2/4 |

Neither PE3 arm passes the advancement gate. The accepted binding decision remains:

`STOP_B24_METHOD_REFINEMENT`

## Fresh2 interpretation

Historical Fresh2 remains a valid descriptive comparator:

- Good25 `57/80`;
- mean PSNR `26.1596 dB`;
- median PSNR `29.9255 dB`;
- versus NP4: mean paired delta `+0.4249 dB`, median `+1.0939 dB`, Good25 rescues/harms `14/12`.

The substantial rescue/harm tradeoff prevents treating this as a post-hoc B24 advancement result. Fresh2 was historical methodology and was not fit on B24.

The DAPS2/Fresh2 dispatch-supported FLOP ratio to PE3_SCORE is approximately `0.4923x`. This is **not** an exact total-compute or runtime ratio; unsupported Fourier/custom forward/backward work remains nonzero and separately inventoried.

## Final B24 conclusion

B24 closes as a negative development result. NP-native pruning/reallocation and protected-explorer variants did not robustly outperform independent NP4 under the prospectively frozen DEV80 gate. The reporting correction is complete, confirmation remains locked, and no further reconstruction is needed for B24 closeout.

Any future Fresh2 investigation must begin as a separate prospective scientific question rather than reopening B24.
