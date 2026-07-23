# B21.11 prospective Fresh2 final benchmark

## Status

Completed prospective estimation on the frozen 100-image panel. The benchmark is valid and the fixed Fresh2 policy satisfies every preregistered descriptive support check.

## Frozen policy

For each locked phase-retrieval measurement:

1. run two independent full DAPS trajectories;
2. use `ann400`, `diff5`, LF disabled, HIO disabled;
3. start with trajectory 1;
4. select trajectory 2 iff its exact operator loss is smaller by more than `0.7`;
5. return the selected reconstruction.

No third trajectory, conditional fallback, detector, or parameter retuning was used.

## Integrity

- independent official-validation images: `100`
- locked measurements: `100`
- trajectory outputs: `200`
- panel manifest SHA-256: `5cfdbf69ac0e19e80e6e8dc00e953a3e69804dc11d16870e511cc1a33045d808`
- all panel, measurement, seed, finite-output, offline-PSNR, and no-dropped-row gates: **passed**

## Primary result

| policy | good25 | bad25 |
|---|---:|---:|
| Fresh1 | `80/100` | `20/100` |
| Fresh2 selected | `92/100` | `8/100` |
| Fresh2 oracle-any-good | `92/100` | `8/100` |

- selected good25 rate: `0.92`
- deterministic image-level bootstrap 95% interval: `[0.86, 0.97]`
- Fresh2 rescues over Fresh1: `12`
- Fresh2 harms over Fresh1: `0`
- selected-oracle gap: `0`
- trajectory-2 accepted: `21/100`

The exact-loss margin selector captured every success available from the two candidates on this panel. All eight remaining failures are therefore candidate-generation failures: neither ordinary trajectory reached good25.

## Selected PSNR

- minimum: `6.3597`
- 5th percentile: `15.8603`
- 10th percentile: `28.0880`
- median: `30.8993`
- mean: `29.2985`
- 90th percentile: `32.3009`
- 95th percentile: `32.6253`
- maximum: `33.2215`

The distribution is strongly separated: most successful reconstructions are around 30--33 dB, while the residual failures are catastrophic rather than marginal misses.

## Cost

- trajectory-1 mean / median wall time: `156.14 / 170.5` seconds
- trajectory-2 mean / median wall time: `156.50 / 160.0` seconds
- total Fresh2 mean / median wall time per image: `312.64 / 333.0` seconds
- mean arm-2 / arm-1 wall ratio: `1.0077`
- observed full-run equivalents: `2.0`

## Preregistered descriptive support checks

- Fresh2 selected good25 at least `90/100`: **passed** (`92/100`)
- selected-oracle gap at most `2/100`: **passed** (`0/100`)
- selector harms at most `1/100`: **passed** (`0/100`)
- at least five rescues over Fresh1: **passed** (`12`)

Descriptive deployment-support statement: **True**.

## Decision

- Adopt Fresh2 as the final fixed DAPS policy for the current `sigma=0.05` FFHQ phase-retrieval setting.
- Freeze `ann400`, `diff5`, two independent starts, and exact-loss margin `0.7`.
- Do not fit another detector, change the threshold, add Fresh3, or test conditional LF/HIO fallbacks on this final panel.
- Treat the remaining `8/100` failures as an explicit candidate-generation limitation.
- Close B21 method development. Subsequent work should be fixed-policy evaluation, baseline comparison, failure visualization, and manuscript consolidation rather than policy search.

Artifacts:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401/analysis_theta0.7
```
