# B21.11 prospective Fresh2 final benchmark

Status: implementation ready on `codex/b21-11-fresh2-final-benchmark`; smoke and integrity review are required before the full launch. The runner implements this plan without changing the panel, seeds, selector, or reporting metrics.

## Objective

Estimate the deployment reliability of the frozen Fresh2 policy on a substantially larger disjoint set of official FFHQ validation images. This benchmark evaluates the fixed policy; it does not tune a detector, fallback, threshold, schedule, or restart count.

## Frozen policy

For each measurement:

1. run two independent full DAPS trajectories;
2. use `ann400`, `diff5`, LF disabled, HIO disabled;
3. start with trajectory 1;
4. select trajectory 2 iff its exact operator loss is smaller by more than `0.7`;
5. return the selected reconstruction.

The restart budget is fixed at two. No third trajectory or conditional fallback is allowed.

## Independent benchmark units

Use `100` distinct official FFHQ validation images with IDs in `60000--69999`.

For each image:

- create one newly generated locked noisy measurement at `sigma=0.05`;
- run one frozen pair of independent trajectory seeds;
- produce one final Fresh2 decision.

Thus the primary benchmark has `100` independent image/measurement units, not repeated trajectory cases on a smaller image panel.

## Deterministic panel construction

1. Enumerate available FFHQ image IDs in `60000--69999` under the configured image root.
2. Exclude every image used in the B21.8 and B21.9 panels:

```text
62802 63282 63803 65808 65960 66452 66892 68263 68924 69293
60067 62957 63135 63199 63319 63368 63678 64050 64116 64471
64542 65317 65656 66511 66731 67092 67673 68111 68922 69441
```

3. For each remaining ID, compute the hexadecimal SHA-256 digest of:

```text
b21.11|5401|<five-digit-image-id>
```

4. Sort by digest and take the first `100` IDs.
5. Write and checksum the frozen panel manifest before generating any measurement or solver output.

This construction is deterministic and prevents image-level cherry-picking.

## Frozen randomness

- measurement panel seed/tag: `5401`
- trajectory-1 seed for benchmark row `i`: `22000 + i`
- trajectory-2 seed for benchmark row `i`: `23000 + i`

Rows are indexed by the deterministic panel order from `0` through `99`.

## Required outputs

### Primary reliability

- Fresh1 good25 count;
- Fresh2 selected good25 count;
- Fresh2 oracle-any-good count;
- Fresh2 selected bad25 count;
- exact selected PSNR distribution, including minimum, median, mean, and quantiles;
- bootstrap or exact binomial uncertainty for the Fresh2 good25 rate, with a 95% interval.

### Incremental value and selector behavior

- Fresh2 rescues over Fresh1;
- Fresh2 harms over Fresh1;
- selected-oracle gap;
- trajectory-2 acceptance fraction;
- all selector-discordant rows with losses and offline PSNR.

### Cost

- per-trajectory wall time;
- total Fresh2 wall time per image;
- mean, median, and quantiles;
- observed cost in full-run equivalents relative to one trajectory.

### Stratified diagnostics

Report image-level rows and aggregate results without fitting thresholds. Candidate disagreement and residual features may be recorded for offline analysis, but they must not alter execution or selection.

## Integrity gates

The benchmark is valid only if:

- all `100` panel images are distinct and outside the excluded development/validation set;
- all `100` locked measurements exist and are distinct files;
- all `200` trajectory outputs and exact-loss CSVs are complete and finite;
- all selected PSNR values are recomputed offline against the frozen FFHQ ground truth;
- no policy parameter is changed after launch;
- no failed row is silently dropped.

## Interpretation

This benchmark is primarily an estimation experiment, not another tuning gate. The final report must state the observed Fresh2 reliability and uncertainty even if it is lower than earlier panels.

For a concise deployment-support statement, report whether all of the following hold without retuning:

```text
Fresh2 selected good25 >= 90/100
Fresh2 selected-oracle gap <= 2/100
Fresh2 selector harms <= 1/100
Fresh2 adds at least 5 rescues over Fresh1
```

Failure of this descriptive support rule does not authorize another detector or restart search on the same panel.
