# B20.7--B20.9: Schedule diversity and basin control for hard image `00046`

This note records the B20.7--B20.9 diagnostic sequence for the hard FFHQ image `00046` in the B19/B20 phase-retrieval solver line. The purpose is to distinguish a useful engineering observation from the stronger scientific goal: ultimately we want every reconstruction attempt, not only an oracle over many attempts, to land in a successful basin.

## Experimental setting

- Image: `00046`
- Noise level: phase retrieval, `sigma = 0.05`
- Primary measurement seed used in the schedule diagnostics: `5001`
- Candidate count per schedule unless stated otherwise: `NUM_RUNS = 16`
- Baseline schedule: approximately `ann200_diff5`
- Tested schedules: `ann250_diff5`, `ann300_diff5`, `ann350_diff5`, `ann375_diff5`, `ann400_diff5`, `ann450_diff5`, and partial `ann500_diff5`
- Primary success threshold in these diagnostics: `PSNR >= 25 dB`

Important caveat: earlier B20.8B measurement-seed validation produced identical results for measurement seeds `5001`, `5002`, and `5003`. This strongly suggests the measurement-seed variation was not actually independent in that validation, or the pipeline reused an identical measurement/payload. Therefore the B20.8B table should not be cited as evidence of measurement-seed robustness until the measurement files and loader path are audited.

## Preceding negative results

Before schedule tuning, several plausible approaches did not solve `00046`.

### B20.7A/B: shallow adaptive deepening was not enough

B20.7A/B showed that successful candidates for `00046` often appear late within a seed, while shallow prefixes do not reliably identify seeds that later become successful. In particular, the first 2, 4, 6, or 8 candidates did not provide a robust seed-selection signal.

Conclusion:

```text
Shallow seed-selection / adaptive deepening is not a reliable solution for 00046.
```

### B20.5A: post-hoc final refinement was not enough

Post-hoc refinement at the final image level produced only tiny improvements and did not rescue true basin failures. It only helped borderline threshold cases such as `00154`.

Conclusion:

```text
For real basin failures like 00046, the final image is already in the wrong basin.
Post-hoc measurement refinement is too late.
```

## B20.8: Annealing length changes basin access

The schedule pilot showed a large nonlocal effect: increasing the number of annealing steps changes which basin a candidate enters.

### Single-schedule summary for `00046`, `meas_seed=5001`

```text
ann250_diff5:
  good = 2 / 8
  rescued = 1
  lost = 0

ann300_diff5:
  good = 3 / 8
  rescued = 2
  lost = 0

ann350_diff5:
  good = 3 / 8
  rescued = 2
  lost = 0

ann375_diff5:
  good = 7 / 8
  rescued = 7
  lost = 1   # loses the originally-good run_seed 5500

ann400_diff5:
  good = 6 / 8
  rescued = 5
  lost = 0
  failures: 5400, 5700

ann450_diff5:
  good = 6 / 8
  rescued = 5
  lost = 0
  failures: 5300, 5600
```

The effect is not monotone per seed. Some seeds improve at `ann375` and fail at `ann400`; others fail at `ann375` but improve at `ann400` or `ann450`. The correct interpretation is not simply "more annealing is better." A better description is:

```text
Different annealing lengths expose different basins.
```

## B20.8G/H: Schedule portfolio solves the tested seed bank, but only as an oracle portfolio

The schedule portfolio replay used existing final-candidate CSVs and constructed candidate pools from multiple annealing schedules.

The best asymmetric portfolios were:

```text
mix375_13_350_3 = ann375_diff5 first 13 candidates + ann350_diff5 first 3 candidates
  cost = 5925 candidate-step units
  good = 8 / 8
  bad25 = 0
  min oracle PSNR = 31.208326
  rescued = 7
  lost = 0

mix375_14_350_2 = ann375_diff5 first 14 candidates + ann350_diff5 first 2 candidates
  cost = 5950 candidate-step units
  good = 8 / 8
  bad25 = 0
  min oracle PSNR = 31.208326
  rescued = 7
  lost = 0

mix375_15_350_1 = ann375_diff5 first 15 candidates + ann350_diff5 first 1 candidate
  cost = 5975 candidate-step units
  good = 8 / 8
  bad25 = 0
  min oracle PSNR = 31.208326
  rescued = 7
  lost = 0
```

For comparison:

```text
ann400_diff5 x 16:
  cost = 6400
  good = 6 / 8
  bad25 = 2

ann375_diff5 x 16:
  cost = 6000
  good = 7 / 8
  bad25 = 1
  but loses the originally-good seed 5500
```

Thus schedule diversity gives a stronger oracle pool than simply increasing annealing depth.

However, this is still an oracle/portfolio result. It does not yet guarantee that each individual reconstruction run succeeds, nor does it tell us how to choose the correct schedule for a given initialization without knowing the final PSNR.

## B20.9A: Schedule-response map

B20.9A built a schedule-response table over each `(run_seed, run_index)` candidate under multiple schedules.

### By-seed summary

```text
run_seed  ann200_good  any_good_schedule  mean_best_psnr  max_best_psnr  mean_num_good_schedules
4400      0            4                  20.674101       31.440145      0.3750
4700      0            5                  22.210872       31.475292      0.3750
5200      0            4                  20.936635       31.384764      0.4375
5300      0            3                  20.522861       31.416054      0.2500
5400      0            3                  20.095813       31.409985      0.1875
5500      1            6                  22.166067       31.452591      0.4375
5600      0            6                  22.678087       31.387690      0.5625
5700      0            3                  19.925651       31.378672      0.1875
```

### Best-schedule counts over all candidates

```text
best_schedule  count  mean_best_psnr  good_candidates
ann400_diff5   33     22.232531       11
ann350_diff5   23     19.783611        4
ann375_diff5   21     25.050007       11
ann450_diff5   18     21.411151        5
ann300_diff5   12     18.221887        1
ann200         11     17.628535        0
ann250_diff5   10     19.463743        2
```

The schedule-response map confirms that success is both seed-dependent and candidate-dependent. A candidate that fails under one annealing length can succeed under another, and each schedule catches a different subset of candidates.

## Scientific interpretation

The current evidence supports the following mechanism:

```text
The hard image 00046 is not primarily a final-selector problem.
It is not solved by post-hoc measurement refinement.
It is not solved by shallow seed diversification.
It is strongly affected by the reverse-process annealing path.
Different annealing paths steer candidates into different basins.
```

The schedule portfolio is useful as an engineering tool and as evidence that diversity matters, but it does not yet solve the stronger goal.

## Remaining goals

The stronger research goal is:

```text
Every individual reconstruction attempt should be considered successful,
not merely the best candidate in an oracle schedule portfolio.
```

This splits naturally into two next questions.

### 1. Choose the correct annealing path for each initialization

Given one initialization / trajectory candidate, can we detect which annealing path should be used?

Useful next diagnostic:

```text
For matched run_seed/run_index pairs across schedules,
compare early trajectory features for schedule-sensitive successes and failures.
```

Candidate examples:

```text
5400, run 4:
  ann450 succeeds (~31.41), while shorter schedules fail.

5400, run 10:
  ann350 succeeds (~31.25), while ann375/ann400/ann450 fail.

5500, run 0:
  ann350 succeeds (~31.32), while ann375 does not.

5300, run 6:
  ann300/ann375 succeed (~31.14--31.21), while ann400/ann450 do not.
```

These paired cases should be used to study whether early residuals, correction norms, jump norms, or x0hat/x0y disagreement indicate which schedule should be chosen.

### 2. Guide initialization toward a better basin

The deeper goal is to increase the probability that each initialization enters a good basin directly. The current failures look basin-level rather than small final errors. Therefore, the next intervention should likely target the early trajectory or initialization distribution.

Candidate directions:

```text
low-frequency / coarse-to-fine warm start
measurement-informed initialization
early reverse-process correction
schedule switching based on early trajectory diagnostics
```

A practical next step is to run raw-trajectory comparisons on a small set of matched schedule-sensitive candidates, then use those diagnostics to design a corrective intervention.

## Current status

B20.7--B20.9 give a strong building block:

```text
Schedule diversity can turn the tested 00046 seed bank from mostly failing to 8/8 oracle success.
```

But the next work should move from oracle portfolios to controlled basin entry:

```text
Detect the right schedule for a candidate, or modify the early trajectory so that each candidate enters a successful basin more often.
```
