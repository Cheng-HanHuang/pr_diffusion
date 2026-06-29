# B19.20 FFHQ100 detector failure decomposition

Date: 2026-06-28  
Branch: `b19_solver_integration`  
Panel: `B19_20_ffhq100_seed20260627_from00000to00999_exclude_ffhq25`  
Evaluation: 100 fresh FFHQ images, measurement seeds `5001--5010`, trajectory run seed `4400`, phase retrieval noise `0.05`.

## Purpose

B19.20 was designed as a detector-level finalization panel.  Earlier detector policies were tested mostly on the original FFHQ-25 panel and on new measurement seeds.  To avoid overfitting to that small image set, we created a fresh reproducible FFHQ100 manifest from `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/00000`, sampled from image IDs `00000--00999` while excluding the old FFHQ-25 panel.

The goal was to decide whether the detector-layer baseline is now sufficiently mature, and whether the project should move to Track 2: solver-level intervention.

## Failure taxonomy

For each selected bad reconstruction, we decompose the source using the analyzer flags:

- `oracle_init_failure_no_good_firstK`: no candidate among the first `K` final reconstructions has PSNR at least 25.  This is an oracle-level / initialization / basin-sampling failure.  A detector cannot recover from this failure because the good basin was not sampled.
- `prefix_selector_failure`: a good candidate exists among the first `K`, but the prefix detector does not keep it.  This is a detector-level failure.
- `final_exact_selector_failure`: a good candidate is kept, but the final exact operator-loss selector chooses a bad candidate.  This is a final-selector / phase-ambiguity failure.

This split is the main result of B19.20.

## Policy-level results

The table below summarizes the main policies over `1000` cases.

| policy                           |   cost_full_equiv |   cases |   mean_selected_psnr |   min_selected_psnr |   bad25 |   bad20 |   oracle_init_failure_no_good_firstK |   prefix_selector_failure |   final_exact_selector_failure |
|:---------------------------------|------------------:|--------:|---------------------:|--------------------:|--------:|--------:|-------------------------------------:|--------------------------:|-------------------------------:|
| F6_full_exact                    |             6     |    1000 |              29.7879 |             6.82754 |      52 |      34 |                                   30 |                         0 |                             22 |
| P6_c100_keep2_lhc_last           |             4     |    1000 |              29.6217 |             8.58308 |      60 |      46 |                                   30 |                        26 |                              4 |
| P5_c125_keep2_lc_last            |             3.875 |    1000 |              29.5464 |             8.14682 |      64 |      43 |                                   50 |                         2 |                             12 |
| F4_full_exact                    |             4     |    1000 |              29.5073 |             6.82754 |      67 |      39 |                                   50 |                         0 |                             17 |
| P5_c125_keep1_loss_only_last     |             3.5   |    1000 |              29.5702 |             8.15063 |      70 |      40 |                                   50 |                        20 |                              0 |
| F5_full_exact                    |             5     |    1000 |              29.3313 |             6.82754 |      74 |      56 |                                   50 |                         0 |                             24 |
| P5_c100-125_keep1_loss_only_mean |             3.5   |    1000 |              29.4628 |             8.14682 |      77 |      50 |                                   50 |                        27 |                              0 |
| P5_c100_keep2_loss_only_last     |             3.5   |    1000 |              29.301  |             8.15346 |      80 |      50 |                                   50 |                        10 |                             20 |
| P5_c75-100_keep2_loss_only_min   |             3.5   |    1000 |              29.2558 |             8.15346 |      83 |      68 |                                   50 |                        13 |                             20 |
| P5_c50-75_keep2_loss_only_min    |             3.125 |    1000 |              29.3278 |             8.15346 |      86 |      50 |                                   50 |                        23 |                             13 |
| P5_c50-75_keep1_loss_only_min    |             2.5   |    1000 |              28.4091 |             6.94027 |     133 |     104 |                                   50 |                        83 |                              0 |
| P5_c75_keep1_loss_only_last      |             2.5   |    1000 |              28.4091 |             6.94027 |     133 |     104 |                                   50 |                        83 |                              0 |

## Current detector baseline

The current compute-saving detector baseline remains

```text
P5_c50-75_keep2_loss_only_min
```

Definition:

```text
K = 5 initial DAPS prefixes
checkpoint set = {50, 75}
decision checkpoint = 75
prefix score = min over checkpoints of rank(x0y measurement loss)
keep_k = 2
final selector = exact DAPS operator loss among kept candidates
cost = 5*(75/200) + 2*(125/200) = 3.125 full-run equivalents
```

On FFHQ100 x 10 measurement seeds, it obtains:

```text
cases = 1000
mean selected PSNR = 29.3278
min selected PSNR = 8.1535
bad25 = 86
bad20 = 50

failure decomposition:
  oracle/init failures       = 50
  prefix-selector failures   = 23
  final-exact failures       = 13
```

Thus this policy is still the best simple low-cost detector baseline, but it is not a complete solver.  Most importantly, many failures are not detector failures.

## Baseline bad images by source

For `P5_c50-75_keep2_loss_only_min`, the bad image/source decomposition is:

|   image_id | bad_source                         |   bad_cases |   mean_selected_psnr |   mean_oracleK_psnr | selected_runs   |
|-----------:|:-----------------------------------|------------:|---------------------:|--------------------:|:----------------|
|      00154 | final_exact_selector_failure       |           8 |             24.795   |            25.2095  | 1               |
|      00136 | final_exact_selector_failure       |           3 |             11.0757  |            31.297   | 0               |
|      00253 | final_exact_selector_failure       |           2 |             24.1734  |            25.0913  | 3               |
|      00971 | oracle_init_failure_no_good_firstK |          10 |              8.16073 |             8.16497 | 0,2             |
|      00480 | oracle_init_failure_no_good_firstK |          10 |              9.33623 |            12.8973  | 2,4             |
|      00171 | oracle_init_failure_no_good_firstK |          10 |             12.535   |            14.304   | 4               |
|      00046 | oracle_init_failure_no_good_firstK |          10 |             16.2689  |            20.9722  | 0,2             |
|      00746 | oracle_init_failure_no_good_firstK |          10 |             20.0959  |            20.5503  | 2               |
|      00599 | prefix_selector_failure            |          10 |             21.3798  |            28.039   | 0,2             |
|      00272 | prefix_selector_failure            |           7 |             13.932   |            29.4526  | 3,4             |
|      00116 | prefix_selector_failure            |           6 |             23.4914  |            31.0796  | 3,4             |

The image-level story is now clear:

- `00046`, `00171`, `00480`, `00746`, and `00971` are oracle/init failures for the K=5 detector baseline.
- `00116`, `00272`, and `00599` are prefix-selector failures.
- `00136`, `00154`, and `00253` are final-exact-selector failures.

## Oracle-hard images under F6

Using `F6_full_exact` as the strongest full-candidate oracle in this panel, the truly oracle-hard images are:

|   image_id |   cases |   oracle6_mean |   oracle6_min |   oracle6_max |   oracle6_bad25_count |   oracle6_bad25_rate |
|-----------:|--------:|---------------:|--------------:|--------------:|----------------------:|---------------------:|
|      00480 |      10 |        12.8973 |       12.8973 |       12.8973 |                    10 |                    1 |
|      00746 |      10 |        20.5503 |       20.5503 |       20.5503 |                    10 |                    1 |
|      00046 |      10 |        20.9722 |       20.9722 |       20.9722 |                    10 |                    1 |

These images have `oracle6_bad25_rate = 1.0`: even among six completed DAPS trajectories, no final reconstruction reaches PSNR 25 for any of the 10 measurement seeds.  These are the cleanest evidence that simply increasing detector quality is insufficient.

A broader low-oracle list begins as follows:

|   image_id |   oracle6_mean |   oracle6_min |   oracle6_max |   f6_selected_bad25 |   f6_final_failures |
|-----------:|---------------:|--------------:|--------------:|--------------------:|--------------------:|
|      00480 |        12.8973 |       12.8973 |       12.8973 |                  10 |                   0 |
|      00746 |        20.5503 |       20.5503 |       20.5503 |                  10 |                   0 |
|      00046 |        20.9722 |       20.9722 |       20.9722 |                  10 |                   0 |
|      00253 |        25.0913 |       25.0913 |       25.0913 |                   6 |                   6 |
|      00154 |        25.2095 |       25.2095 |       25.2095 |                   8 |                   8 |
|      00523 |        25.6556 |       25.6556 |       25.6556 |                   0 |                   0 |
|      00224 |        26.7092 |       26.7092 |       26.7092 |                   0 |                   0 |
|      00171 |        27.305  |       27.305  |       27.305  |                   0 |                   0 |
|      00599 |        28.039  |       28.039  |       28.039  |                   0 |                   0 |
|      00592 |        28.0479 |       28.0479 |       28.0479 |                   0 |                   0 |
|      00221 |        28.4036 |       28.4036 |       28.4036 |                   0 |                   0 |
|      00368 |        28.6822 |       28.6822 |       28.6822 |                   0 |                   0 |
|      00700 |        28.8455 |       28.8455 |       28.8455 |                   0 |                   0 |
|      00753 |        29.1716 |       29.1716 |       29.1716 |                   0 |                   0 |
|      00520 |        29.1746 |       29.1746 |       29.1746 |                   0 |                   0 |

## Conclusions

### 1. The detector-level baseline is now final enough as a baseline.

`P5_c50-75_keep2_loss_only_min` is simple, early, and cheap.  It uses only the clean-free measurement-loss rank at checkpoints 50 and 75, keeps two candidates, and costs only 3.125 full-run equivalents.  It should be retained as the current detector-level compute-allocation baseline.

However, on FFHQ100 it is not more reliable than `F6_full_exact`.  It has more bad25 cases than F6, but at roughly half the compute.  Therefore it is best described as a compute-saving baseline, not as a robustness solution.

### 2. FFHQ100 reveals the limitation of detector-only methods.

On FFHQ100, many bad reconstructions are oracle/init failures: the first `K` trajectories simply do not contain a good reconstruction.  This was not sufficiently visible on the old FFHQ-25 panel.

This means the detector layer can answer only:

```text
Given a small candidate pool that already contains a good basin,
can we keep the good candidate cheaply?
```

It cannot answer:

```text
How do we ensure the good basin is sampled in the first place?
```

### 3. Track 2 is urgent.

Track 2 should now become the main research direction.  The immediate evidence is:

- 50 / 86 bad cases for the current detector baseline are oracle/init failures.
- F6 still has 30 oracle/init failures and 22 final-exact failures.
- Several images are oracle-hard even under six full DAPS trajectories.
- Some failures are final-exact-selector failures, showing that exact measurement loss alone can still prefer a bad ambiguity/mode.

Thus, merely changing prefix metrics or taking more candidates is not a principled solution.

## Track 2 directions

The most promising solver-level routes are:

### A. Initialization / basin-sampling treatment

Goal: increase the probability that at least one candidate enters the correct basin.

Possible tests:

- Use a spectral or classical phase-retrieval initialization as an anchor or warm start.
- Refine initial states before reverse diffusion.
- Generate candidates by structured perturbations around a measurement-consistent anchor rather than independent random prefixes.
- Measure whether oracleK improves on the oracle-hard image set.

Primary target images:

```text
00046, 00480, 00746
```

Secondary K=5-hard / borderline images:

```text
00171, 00971
```

### B. Per-step measurement guidance / repair

Goal: prevent trajectories from drifting into self-consistent wrong basins.

Possible tests:

- Add a small measurement-gradient correction at selected reverse steps.
- Use a trust-region version of measurement correction to avoid overcorrection.
- Trigger guidance adaptively when measurement loss or correction residual becomes suspicious.
- Compare oracleK and selected PSNR before/after guidance.

### C. Candidate diversity control

Goal: make the K candidates cover different basins rather than repeatedly sampling the same bad mode.

Possible tests:

- Penalize near-duplicate prefixes using pairwise image/low-frequency distances.
- Force retained or continued candidates to be diverse.
- Use clustering before continuation: keep candidates from different basins, not merely the lowest early loss.
- Evaluate whether oracleK improves on hard images.

### D. Rejection / restart

Goal: detect when the whole candidate pool is unreliable and start a new batch.

Possible tests:

- Reject when all early measurement losses are high or when the kept candidates are mutually inconsistent.
- Reject when the predicted oracle risk is high.
- Use this as an outer-loop safety mechanism rather than pretending a bad pool can be fixed by selection.

### E. Better final selector

Goal: fix cases where a good candidate is available but exact operator loss chooses a bad one.

Target images from baseline final-exact failures:

```text
00136, 00154, 00253
```

Possible tests:

- Add prior-consistency tie-breaks to exact loss.
- Use symmetry-aware evaluation.
- Use stability under small measurement refinements as a final-selection diagnostic.
- Compare with exact-loss-only selection on final-exact failure cases.

## Recommended next experiment

The next stage should not be another broad detector metric sweep.  It should be a small Track 2 pilot:

```text
B20.1 oracle-hard initialization/refinement pilot
```

Initial target set:

```text
00046, 00480, 00746, 00171, 00971
```

Protocol:

1. Run the current DAPS/F6 baseline and record oracleK.
2. Add one solver-level intervention.
3. Measure whether oracleK improves, not merely selected PSNR.
4. Only after oracleK improves should detector selection be re-evaluated.

The core success metric for Track 2 is:

```text
Does the intervention turn oracle/init failures into cases where at least one good reconstruction exists?
```

That is the right next question after B19.20.
