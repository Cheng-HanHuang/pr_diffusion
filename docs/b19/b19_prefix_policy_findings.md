# B19 Prefix Policy Findings

This note summarizes the current B19 branch-B findings for DAPS phase retrieval with locked measurements.

## Leading policy

The current main frozen policy is:

```text
P6_c100_keep2_inst_lhc:
  K = 6 valid DAPS prefixes
  checkpoint = 100 / 200
  keep_k = 2
  score = rank(x0y measurement loss)
        + rank(x0hat measurement loss)
        + rank(correction RMS)
  final selector = exact DAPS operator loss among the two completed candidates
  cost = 4.0 full-DAPS equivalents
```

The conservative backup policy is:

```text
P5_c125_keep2_inst_lc:
  K = 5 valid DAPS prefixes
  checkpoint = 125 / 200
  keep_k = 2
  score = rank(x0y measurement loss) + rank(correction RMS)
  final selector = exact DAPS operator loss among the two completed candidates
  cost = 3.875 full-DAPS equivalents
```

The step-count cost model is:

```text
cost = K * checkpoint / 200 + keep_k * (200 - checkpoint) / 200
```

`keep_k = 2` means that two prefixes are kept alive and completed. The final selected reconstruction is still chosen by clean-free exact operator loss, not by final PSNR.

## B19.13 cost-normalized replay

On the 15-image hard/mixed panel, the first leading policy was `P5_c125_keep2_cost3p875`.

| policy | cost | mean PSNR | min PSNR | bad25 | bad20 | >=29 | >=30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P5_c125_keep2_cost3p875` | 3.875 | 30.747 | 29.311 | 0/15 | 0/15 | 15/15 | 12/15 |
| `F4_full_exact` | 4.000 | 29.304 | 10.352 | 1/15 | 1/15 | 13/15 | 10/15 |

The prefix policy is not simply using more compute. It reallocates a best-of-4-like budget: one additional partial prefix is sampled, but only two prefixes are completed.

## Failure-mode decomposition

The B19 replay separates three failure modes.

1. Initialization failure: the first candidate/prefix pool contains no good basin.
2. Prefix-selection failure: the pool contains a good basin, but the prefix detector discards all good prefixes.
3. Final-selection failure: the completed set contains a good candidate, but exact final measurement loss selects a bad one.

In B19.13:

- `F4_full_exact` fails on `00007` because the first four runs contain no good reconstruction. This is an initialization failure.
- `F5/F6/F8_full_exact` fail on `00034` because exact final loss selects an upside-down rot180 symmetry candidate. This is a final-selection failure.
- `P5_c125_keep2_cost3p875` has zero initialization failures, zero prefix-selection failures, and zero final-selection failures on the hard/mixed panel.

## B19.13C: `00034` final-selection failure

For `00034`, full exact-loss selection can be fooled by a rot180-like reconstruction.

![B19.13C 00034 first 8 candidates](figures/B19_13C_00034_first8_contact_sheet.png)

The key bad candidate is run `4`:

- Unaligned PSNR: about `11.94`.
- Exact operator loss: about `977.08`.
- Best aligned PSNR after rot180: about `31.13`.

Final exact-loss selection picks this bad rot180 candidate for `F5`, `F6`, and `F8`.

However, the prefix detector avoids it. At checkpoint 125 among the first five runs, run `4` has worse prefix score than the good runs. The leading policy keeps runs `2,0`, both unaligned-good, and avoids the symmetry basin.

## B19.13D: flip / symmetry audit

B19.13D audited 240 candidates from the 15-image hard/mixed panel.

| quantity | count |
|---|---:|
| total candidates | 240 |
| unaligned bad25 candidates | 85 |
| symmetry-rescuable bad candidates | 13 |
| rot180-rescuable bad candidates | 13 |

Thus about 15% of unaligned-bad candidates are actually good after 180-degree rotation. These are not ordinary hallucination failures; they are phase-retrieval symmetry-basin failures.

Rot180-rescuable cases appeared in:

- `00007`: 2 candidates
- `00017`: 3 candidates
- `00028`: 1 candidate
- `00032`: 1 candidate
- `00034`: 6 candidates

This symmetry audit is diagnostic only. Official comparison should still use the standard unaligned PSNR unless all baselines are also evaluated with symmetry alignment.

## B19.13E: policy symmetry-class audit

B19.13E classified selected and kept candidates as:

- `unaligned_good`: identity PSNR >= 25
- `symmetry_rescuable`: identity PSNR < 25 but best aligned PSNR >= 25
- `true_bad`: best aligned PSNR < 25

For the leading policy:

| policy | selected unaligned-good | selected symmetry-rescuable | selected true-bad | kept symmetry-rescuable |
|---|---:|---:|---:|---:|
| `P5_c125_keep2_cost3p875` | 15/15 | 0/15 | 0/15 | 0 |

The leading policy saw symmetry-rescuable candidates in its candidate pools for two images, but kept none of them.

In contrast:

- `F4_full_exact` selected one true-bad candidate, on `00007`.
- `F5_full_exact`, `F6_full_exact`, and `F8_full_exact` selected the `00034` symmetry-rescuable rot180 candidate.
- Full `F12` and `F16` eventually selected unaligned-good candidates, but they keep many symmetry and true-bad candidates because they do not prune.

## B19.14/B19.15: earliest reliable checkpoint and symmetry signal

B19.14 searched earlier checkpoint policies using existing raw trajectories. The main conclusion was that checkpoint 50 and 75 are too early for the current features, while checkpoint 100 is the earliest reliable checkpoint.

The most useful policy from this stage was:

```text
P6_c100_keep2_inst_lhc:
  sample 6 prefixes to checkpoint 100
  rank by x0y measurement loss + x0hat measurement loss + correction RMS
  keep 2 prefixes
  finish the kept prefixes
```

B19.15 showed that rot180-rescuable candidates are exactly measurement-ambiguous: the DAPS phase-retrieval operator loss of a candidate and its 180-degree rotation is equal up to numerical precision. However, by checkpoint 100 the prefix features strongly separate the symmetry-rescuable candidates from the unaligned-good candidates in the hard/mixed panel.

## B19.16A/B/C/D: full FFHQ-25 frozen-policy validation

After freezing `P6_c100_keep2_inst_lhc`, we evaluated it on full FFHQ-25.

### B19.16A: same-seed full FFHQ-25 replay

B19.16A used locked `meas3000` measurements and combined the existing hard/mixed raw16 trajectories with raw6 trajectories for the remaining images.

| policy | cost | mean PSNR | min PSNR | bad25 | bad20 | >=29 | >=30 | failure summary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `P5_c125_keep2_inst_lc` | 3.875 | 30.893 | 29.311 | 0/25 | 0/25 | 25/25 | 22/25 | none |
| `P6_c100_keep2_inst_lhc` | 4.000 | 30.838 | 29.276 | 0/25 | 0/25 | 25/25 | 21/25 | none |
| `F4_full_exact` | 4.000 | 30.013 | 10.352 | 1/25 | 1/25 | 23/25 | 20/25 | init failure on `00007` |
| `F6_full_exact` | 6.000 | 30.030 | 11.943 | 1/25 | 1/25 | 23/25 | 21/25 | final-selection failure on `00034` |

### B19.16B: fresh trajectory validation

B19.16B reran all 25 images with fresh DAPS trajectory seed `4100`, while keeping the same locked measurements `meas3000`.

| policy | cost | mean PSNR | min PSNR | bad25 | bad20 | >=29 | >=30 | failure summary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `P5_c125_keep2_inst_lc` | 3.875 | 30.824 | 29.319 | 0/25 | 0/25 | 25/25 | 21/25 | none |
| `P6_c100_keep2_inst_lhc` | 4.000 | 30.845 | 29.278 | 0/25 | 0/25 | 25/25 | 21/25 | none |
| `F4_full_exact` | 4.000 | 29.959 | 9.162 | 1/25 | 1/25 | 24/25 | 20/25 | final-selection failure on `00005` |
| `F6_full_exact` | 6.000 | 29.962 | 9.162 | 1/25 | 1/25 | 24/25 | 20/25 | final-selection failure on `00005` |

### B19.16C: fresh-run `00005` audit

B19.16C showed that the fresh-run `00005` failure is another rot180 symmetry-basin case.

Full exact-loss selection picks run `0`:

```text
run 0:
  final PSNR = 9.16
  exact operator loss = 963.395
  rot180-aligned PSNR = 30.41
```

Good upright candidates exist in the same pool:

```text
run 3:
  final PSNR = 31.23
  exact operator loss = 963.651

run 2:
  final PSNR = 31.25
  exact operator loss = 964.413
```

At checkpoint 100, the frozen prefix score ranks the good upright runs ahead of run `0`:

```text
run 3: score_inst_lhc = 4
run 2: score_inst_lhc = 5
run 0: score_inst_lhc = 9
```

Therefore `P6_c100_keep2_inst_lhc` keeps runs `3,2`, rejects the low-loss rot180 basin, and selects a 31.23 dB reconstruction.

### B19.16D: fresh measurement validation

B19.16D generated fresh locked measurements with seed `4000` and used trajectory seed `4100`.

| policy | cost | mean PSNR | min PSNR | bad25 | bad20 | >=29 | >=30 | failure summary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `P5_c125_keep2_inst_lc` | 3.875 | 30.873 | 29.249 | 0/25 | 0/25 | 25/25 | 22/25 | none |
| `F4_full_exact` | 4.000 | 30.876 | 29.249 | 0/25 | 0/25 | 25/25 | 22/25 | none |
| `P6_c100_keep2_inst_lhc` | 4.000 | 30.820 | 29.209 | 0/25 | 0/25 | 25/25 | 21/25 | none |
| `F6_full_exact` | 6.000 | 30.856 | 29.249 | 0/25 | 0/25 | 25/25 | 22/25 | none |

Fresh measurement seed `4000` appears easier for standard DAPS4S: `F4_full_exact` has no bad25 failures. The frozen prefix policy also remains stable and has no bad25 failures, but does not improve mean PSNR on this seed.

## Current interpretation

The current evidence supports the following claim:

```text
At the same step-count budget as DAPS4S,
we run 6 valid DAPS prefixes to checkpoint 100,
use clean-free measurement/prior-consistency features to keep 2,
and complete only the kept prefixes.
```

This policy avoids the observed DAPS4S/DAPS6S failures on `meas3000`, including final exact-loss selection of rot180 symmetry-basin candidates. On fresh measurement seed `4000`, the standard baselines do not fail, and the prefix policy remains stable.

The method does not solve phase retrieval ambiguity in principle. Fourier magnitude alone cannot distinguish upside-up and upside-down symmetry-related reconstructions. The contribution is a clean-free basin-selection mechanism that uses mid-trajectory consistency to avoid some ambiguous low-loss basins.
