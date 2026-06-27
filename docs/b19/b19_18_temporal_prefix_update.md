# B19.18 Temporal Prefix Detector Numerical Update

Date: 2026-06-27  
Branch: `b19_solver_integration`  
Local PAC repo: `/egr/research-pac/huang248/pr_diffusion_b19_solver`  
Output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver`

This note records the B19.17/B19.18/B19.18B temporal-prefix detector results before moving on to additional detector metrics. The main purpose is to distinguish:

1. policies that were discovered by replay search,
2. policies that were frozen and validated on fresh measurements,
3. decision-time/checkpoint regimes that appear unsafe,
4. failure modes of the remaining bad selected prefixes.

The key conclusion is that temporal stability is useful, but a single frozen keep-one detector is not yet empirically sufficient in general. The strongest validated directions are:

- temporal stability plus keep-two, especially `P5_c50-75_keep2_lhc_mean`, and
- triple temporal keep-one, especially `P5_c50-75-100_keep1_lhc_mean`, which passed the fresh validation but with a lower PSNR floor than the keep-two policy.

## 1. Background and notation

The B19 prefix-policy setting is clean-free: selection is allowed to use only measurement/operator quantities and trajectory-derived quantities, not ground-truth PSNR. For a policy

```text
P{K}_c{checkpoints}_keep{keep_k}_{score}_{aggregation}
```

- `K` is the number of DAPS prefixes started.
- `checkpoints` is the set of intermediate annealing steps used for scoring.
- `keep_k` is the number of prefixes kept after the decision checkpoint.
- The final selector among kept completed candidates is exact DAPS operator loss, unless `keep_k=1`, where the chosen prefix is the final output.
- Cost is measured in full-DAPS equivalents:

```text
cost = K * decision_checkpoint / 200 + keep_k * (200 - decision_checkpoint) / 200.
```

Scores used here:

```text
lc  = rank(x0y measurement loss) + rank(correction RMS)
lhc = rank(x0y measurement loss) + rank(x0hat measurement loss) + rank(correction RMS)
lhc_x0yjump = lhc + rank(x0y jump RMS)
```

For temporal scores, the score is computed independently at each checkpoint and then aggregated, usually by mean.

## 2. B19.17 keep-one failure diagnostic

B19.17 showed that the original single-checkpoint keep-one detector was not enough.

### Failing policy

```text
P6_c100_keep1_lhc_last
```

On the B19.16B fresh-trajectory replay, this policy failed on image `00010`:

| item | value |
|---|---:|
| image | `00010` |
| keep1 selected run | `0` |
| keep1 selected PSNR | `23.6359` |
| keep2 policy | `P6_c100_keep2_lhc_last` |
| keep2 kept runs | `0,3` |
| keep2 selected run | `3` |
| keep2 selected PSNR | `29.3625` |
| oracle best run among first 6 | `5` |
| oracle best PSNR | `30.3691` |
| selected bad symmetry class | `true_bad_or_collapsed` |
| selected identity PSNR | `23.6281` |
| selected rot180 PSNR | `9.1996` |

Checkpoint trace for the key bad/good comparison:

| checkpoint | run 0 score `lhc` | run 0 status | run 3 score `lhc` | run 3 status |
|---:|---:|---|---:|---|
| 75 | 15 | final bad | 6 | final good |
| 100 | 3 | final bad, selected by keep1 | 8 | final good, rescued by keep2 |
| 125 | 15 | final bad | 3 | final good |

Interpretation: run `0` was a transient false positive at checkpoint 100. It looked locally promising at the decision step, but was unstable across neighboring checkpoints and later became a bad reconstruction. This motivated temporal-stability scores.

## 3. B19.18 replay-only temporal metric search

B19.18 was a replay-only metric search over three existing datasets:

| replay dataset | cases |
|---|---:|
| B19.16A seed3000 same/mixed trajectories | 25 |
| B19.16B meas3000 runseed4100 fresh trajectories | 25 |
| B19.16D meas4000 runseed4100 fresh measurement | 25 |
| total | 75 |

No new DAPS trajectories were run in B19.18. The script recomputed prefix selections from existing exact-loss and window-feature CSVs.

### 3.1 Reference policies in replay

| policy | cost | mean PSNR | min PSNR | bad25 | bad20 | prefix failures | final failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P5_c125_keep1_lc_last` | 3.500 | 30.8340 | 29.2008 | 0 | 0 | 0 | 0 |
| `P5_c125_keep2_lc_last` | 3.875 | 30.8630 | 29.2491 | 0 | 0 | 0 | 0 |
| `P6_c100_keep2_lhc_last` | 4.000 | 30.8340 | 29.2085 | 0 | 0 | 0 | 0 |
| `P6_c100_keep1_lhc_last` | 3.500 | 30.6581 | 23.6359 | 2 | 0 | 2 | 0 |
| `F4_full_exact` | 4.000 | 30.2827 | 9.1622 | 2 | 2 | 0 | 1 |
| `F6_full_exact` | 6.000 | 30.2827 | 9.1622 | 2 | 2 | 0 | 2 |

Main replay takeaway: single-checkpoint `P6_c100_keep1_lhc_last` failed, while temporal variants could eliminate the replay failures.

### 3.2 Top zero-bad replay policies

The replay search found several zero-bad temporal keep-one policies. The most notable ones were:

| policy | decision checkpoint | cost | mean PSNR | min PSNR | bad25 | selected >=29 | selected >=30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P5_c75-100_keep1_lhc_mean` | 100 | 3.000 | 30.8057 | 29.0098 | 0 | 75 | 62 |
| `P5_c75-100_keep1_lhc_median` | 100 | 3.000 | 30.8057 | 29.0098 | 0 | 75 | 62 |
| `P5_c50-75-100_keep1_lc_median` | 100 | 3.000 | 30.8258 | 28.7744 | 0 | 74 | 63 |
| `P5_c50-75-100_keep1_lhc_xtjump_median` | 100 | 3.000 | 30.8229 | 28.7744 | 0 | 74 | 64 |
| `P5_c50-75_keep2_loss_only_min` | 75 | 3.125 | 30.8161 | 28.9840 | 0 | 74 | 63 |

The replay initially suggested `P5_c75-100_keep1_lhc_mean` as the most attractive low-cost keep-one detector.

## 4. B19.18B frozen fresh validation

B19.18B froze selected policies and evaluated them on new measurements:

```text
measurement seeds: 4001, 4002, 4003
run seed: 4200
images: FFHQ-25
cases: 3 * 25 = 75
```

Important caveat: because the run seed was fixed at `4200`, repeated failures across measurement seeds are not fully independent. The same image/run can repeat across measurement seeds.

### 4.1 Frozen validation summary

| policy | family | decision checkpoint | cost | mean PSNR | min PSNR | bad25 | bad20 | selected >=29 | selected >=30 | prefix failures | final failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Abl_P5_c50-75-100_keep1_lhc_mean` | early ablation / triple temporal keep1 | 100 | 3.000 | 30.7656 | 27.4188 | 0 | 0 | 72 | 63 | 0 | 0 |
| `Abl_P5_c50-75_keep2_lhc_mean` | early ablation / double temporal keep2 | 75 | 3.125 | 30.8925 | 29.2596 | 0 | 0 | 75 | 65 | 0 | 0 |
| `P5_c125_keep2_lc_last` | old robust reference | 125 | 3.875 | 30.8528 | 28.8614 | 0 | 0 | 74 | 62 | 0 | 0 |
| `F4_full_exact` | reference | 200 | 4.000 | 30.8881 | 28.9284 | 0 | 0 | 74 | 64 | 0 | 0 |
| `P6_c100_keep2_lhc_last` | old robust reference | 100 | 4.000 | 30.8597 | 28.9284 | 0 | 0 | 72 | 65 | 0 | 0 |
| `F6_full_exact` | reference | 200 | 6.000 | 30.9025 | 29.2781 | 0 | 0 | 75 | 64 | 0 | 0 |
| `Abl_P5_c75_keep1_lhc_last` | early ablation | 75 | 2.500 | 30.6508 | 24.6124 | 3 | 0 | 72 | 63 | 3 | 0 |
| `Abl_P5_c50-75_keep1_lhc_mean` | early ablation | 75 | 2.500 | 29.9805 | 10.9373 | 3 | 3 | 69 | 60 | 3 | 0 |
| `P5_c75-100_keep1_lhc_mean` | frozen temporal primary | 100 | 3.000 | 30.6360 | 24.6124 | 3 | 0 | 70 | 63 | 3 | 0 |
| `P6_c50-75-100_keep1_lhc_x0yjump_mean` | frozen temporal secondary | 100 | 3.500 | 29.8882 | 11.8007 | 3 | 3 | 69 | 54 | 3 | 0 |
| `Abl_P6_c75_keep1_lhc_x0yjump_last` | early ablation | 75 | 2.875 | 29.2851 | 11.8007 | 6 | 6 | 68 | 57 | 6 | 0 |
| `Abl_P5_c50_keep1_lhc_last` | early ablation | 50 | 2.000 | 28.5795 | 8.8869 | 9 | 9 | 66 | 57 | 9 | 0 |
| `Abl_P6_c50_keep1_lhc_x0yjump_last` | early ablation | 50 | 2.250 | 28.4749 | 10.2662 | 9 | 9 | 63 | 50 | 9 | 0 |
| `Abl_P6_c50-75_keep1_lhc_x0yjump_mean` | early ablation | 75 | 2.875 | 28.2478 | 10.2662 | 10 | 10 | 63 | 47 | 10 | 0 |

### 4.2 Fresh-validation interpretation

The replay-selected primary policy did not generalize:

```text
P5_c75-100_keep1_lhc_mean:
  bad25 = 3 / 75
  all failures are image 00039, run 2, selected PSNR ~24.61
```

The secondary jump-aware keep-one policy also did not generalize:

```text
P6_c50-75-100_keep1_lhc_x0yjump_mean:
  bad25 = 3 / 75
  all failures are image 00028, run 5, selected PSNR ~11.80
```

However, the triple temporal `lhc` keep-one ablation passed:

```text
Abl_P5_c50-75-100_keep1_lhc_mean:
  decision checkpoint = 100
  cost = 3.0
  bad25 = 0 / 75
  min PSNR = 27.4188
```

And the early double temporal keep-two policy was stronger in PSNR floor:

```text
Abl_P5_c50-75_keep2_lhc_mean:
  decision checkpoint = 75
  cost = 3.125
  bad25 = 0 / 75
  min PSNR = 29.2596
```

Thus the current best validated directions are:

1. **aggressive keep-one:** `P5_c50-75-100_keep1_lhc_mean`, cost 3.0, decision checkpoint 100;
2. **more robust early keep-two:** `P5_c50-75_keep2_lhc_mean`, cost 3.125, decision checkpoint 75.

## 5. What checkpoint/decision time appears safe?

Current evidence from B19.18B:

| decision regime | current status | evidence |
|---|---|---|
| checkpoint 50 keep-one | unsafe | 9 bad25 for `P5_c50_keep1_lhc_last`; 9 bad25 for `P6_c50_keep1_lhc_x0yjump_last` |
| checkpoint 75 keep-one | unsafe | 3 bad25 for `P5_c75_keep1_lhc_last`; 6 bad25 for `P6_c75_keep1_lhc_x0yjump_last` |
| checkpoints 50,75 keep-one | unsafe | 3 bad25 for `P5_c50-75_keep1_lhc_mean`; 10 bad25 for `P6_c50-75_keep1_lhc_x0yjump_mean` |
| checkpoints 50,75 keep-two | promising / fresh-validated | `P5_c50-75_keep2_lhc_mean`: 0 bad25, min PSNR 29.2596 |
| checkpoints 75,100 keep-one | not sufficient | `P5_c75-100_keep1_lhc_mean`: 3 bad25, min PSNR 24.6124 |
| checkpoints 50,75,100 keep-one | promising / fresh-validated | `P5_c50-75-100_keep1_lhc_mean`: 0 bad25, min PSNR 27.4188 |
| checkpoint 100 keep-two | robust reference | `P6_c100_keep2_lhc_last`: 0 bad25, min PSNR 28.9284 |
| checkpoint 125 keep-two | robust reference | `P5_c125_keep2_lc_last`: 0 bad25, min PSNR 28.8614 |
| full exact F4/F6 | can be good or fail depending on seed panel | B19.18B had 0 bad25, but earlier B19.16A/B had failures due to initialization/final selection |

Important nuance: B19.18B alone was an easier panel for full exact F4/F6 than B19.16A/B. Earlier validations showed F4/F6 can still fail on other measurement/run panels.

## 6. Failure-mode audit: B19.18C

B19.18C classified bad selected candidates from the B19.18B bad-case table by identity and rot180 PSNR.

Classification thresholds used in the audit script:

```text
unaligned_good: identity_psnr >= 25
rot180_rescuable: identity_psnr < 25 and rot180_psnr >= 25
near_miss_true_bad: identity_psnr >= 22 and identity_psnr >= rot180_psnr
true_bad_or_collapsed: otherwise
```

### 6.1 Unique bad selected candidates

| image | run | identity PSNR | rot180 PSNR | best aligned PSNR | best alignment | audit class | interpretation |
|---|---:|---:|---:|---:|---|---|---|
| `00039` | 2 | 24.6074 | 11.0780 | 24.6074 | identity | `near_miss_true_bad` | identity-aligned near-miss, not flipped |
| `00000` | 5 | 15.3500 | 12.1646 | 15.3500 | identity | `true_bad_or_collapsed` | low-quality/collapsed |
| `00007` | 5 | 10.2704 | 18.5194 | 18.5194 | rot180 | `true_bad_or_collapsed` | rot180 improves but not enough |
| `00013` | 1 | 8.8874 | 15.2332 | 15.2332 | rot180 | `true_bad_or_collapsed` | rot180 improves but still bad |
| `00014` | 4 | 10.9380 | 13.7856 | 13.7856 | rot180 | `true_bad_or_collapsed` | low-quality/collapsed |
| `00015` | 5 | 12.6643 | 14.5837 | 14.5837 | rot180 | `true_bad_or_collapsed` | low-quality/collapsed |
| `00028` | 4 | 13.6332 | 13.0843 | 13.6332 | identity | `true_bad_or_collapsed` | low-quality/collapsed |
| `00028` | 5 | 11.8005 | 24.8121 | 24.8121 | rot180 | `true_bad_or_collapsed` by hard threshold; better described as `rot180_near_miss` | orientation-adjacent near-miss |

Counts under the original hard-threshold classification:

| class | count |
|---|---:|
| `near_miss_true_bad` | 3 |
| `true_bad_or_collapsed` | 21 |

The hard-threshold label hides an important nuance: `00028` run 5 has identity PSNR about 11.80 but rot180 PSNR about 24.81. It is not rot180-rescuable above the 25 threshold, but it is best described as a **rot180 near-miss** rather than pure collapse.

### 6.2 Refined failure taxonomy

A better taxonomy for future audits is:

```text
identity_good:
  identity_psnr >= 25

rot180_good:
  identity_psnr < 25 and rot180_psnr >= 25

identity_near_miss:
  22 <= identity_psnr < 25 and identity_psnr >= rot180_psnr

rot180_near_miss:
  22 <= rot180_psnr < 25 and rot180_psnr > identity_psnr

collapsed_or_bad:
  best_aligned_psnr < 22
```

Under this taxonomy:

- `00039` run 2 is `identity_near_miss`.
- `00028` run 5 is `rot180_near_miss`.
- Many 8--15 PSNR cases remain `collapsed_or_bad` even after rotation.

## 7. Current scientific conclusions

### 7.1 Current detector is not empirically solved

The B19.18/B19.18B results show that temporal stability is necessary but not enough to uniquely determine a final keep-one detector. The replay-selected keep-one metric `P5_c75-100_keep1_lhc_mean` failed under frozen fresh validation.

Thus, the current detector is insufficient to guarantee empirical success in the strong sense unless we include either:

- keep-two robustness, or
- a better detector metric than the current rank-based temporal scores.

### 7.2 Keep-two is not a cosmetic addition

The best early keep-two temporal policy:

```text
P5_c50-75_keep2_lhc_mean
```

passed the fresh validation with a high PSNR floor:

```text
bad25 = 0 / 75
min PSNR = 29.2596
cost = 3.125
decision checkpoint = 75
```

This suggests keep-two is a principled uncertainty buffer: when the clean-free detector is not confident enough to commit to one basin, keeping two branches avoids transient false positives.

### 7.3 Failure types are heterogeneous

Bad selected prefixes include:

1. identity-aligned near-misses, e.g. `00039` run 2;
2. orientation-adjacent rot180 near-misses, e.g. `00028` run 5;
3. truly poor/collapsed reconstructions, e.g. several 8--15 PSNR cases.

Therefore, a single scalar measurement-loss metric is unlikely to be sufficient. Detector metrics should probe both:

- measurement consistency, and
- prior/basin consistency under the diffusion dynamics.

## 8. Recommended next detector directions

The next stage should not declare a final detector prematurely. Instead, test detector metrics motivated by the diffusion inverse-problem dynamics.

### 8.1 Temporal stability and confidence margins

Already supported by B19.17/B19.18.

Candidate ideas:

```text
mean score over checkpoints
max score over checkpoints
range penalty: last + alpha * (max - min)
path-difference penalty: last + alpha * sum |score_t - score_{t-1}|
margin between rank-1 and rank-2
adaptive keep count: keep1 only if margin/stability is large; otherwise keep2
```

### 8.2 Denoiser/prior consistency after measurement correction

Each DAPS step has the form:

```text
xt -> x0hat -> x0y -> next xt
```

Existing `correction_rms = ||x0y - x0hat||` is only a crude proxy. A stronger detector should test whether `x0y` remains in a stable diffusion-prior basin after measurement correction.

Possible metrics:

```text
||x0y - x0hat|| relative to ||x0hat||
change in denoiser output after correction
one-step denoiser consistency from x0y re-noised at the same sigma
local score norm or predicted noise norm if available
Tweedie/ODE disagreement if both are available
```

This is especially relevant to per-iteration initialization failures: a prefix may match the measurement briefly, but if the measurement correction pushes it out of distribution, the next denoising step may enter a bad basin.

### 8.3 Branch consensus / isolation

Candidate basins that are supported by multiple nearby prefixes may be more reliable than isolated low-loss branches.

Possible metrics:

```text
nearest-neighbor distance among x0y candidates
mean/median distance to other candidates
distance to candidate median/Fréchet center
cluster size among low-score candidates
consensus rank combined with temporal rank
```

### 8.4 Orientation/symmetry priors

Magnitude-only phase retrieval cannot resolve some orientation/symmetry information from measurement loss alone. B19.18C shows that some failures are orientation-adjacent even when not fully rot180-rescuable.

Possible clean-free metrics:

```text
prior plausibility of x versus rot180(x)
denoiser residual of x versus rot180(x)
score/energy comparison of x and rot180(x)
face-uprightness prior, if using an external face prior is allowed
```

### 8.5 Multi-threshold failure taxonomy

Future bad-case audits should report:

```text
identity_good
rot180_good
identity_near_miss
rot180_near_miss
collapsed_or_bad
```

rather than only `bad25`, because 11.8 identity PSNR with 24.8 rot180 PSNR is very different from 11.8 identity PSNR with 12.0 rot180 PSNR.

## 9. Current recommended policy labels for future experiments

Until further validation, use the following labels:

```text
Old robust reference:
  P6_c100_keep2_lhc_last
  P5_c125_keep2_lc_last

Fresh-validated aggressive keep-one candidate:
  P5_c50-75-100_keep1_lhc_mean

Fresh-validated early robust candidate:
  P5_c50-75_keep2_lhc_mean

Known insufficient/failing under B19.18B:
  P5_c75-100_keep1_lhc_mean
  P6_c50-75-100_keep1_lhc_x0yjump_mean
  P5_c50_keep1_lhc_last
  P5_c75_keep1_lhc_last
  P5_c50-75_keep1_lhc_mean
  P6_c50_keep1_lhc_x0yjump_last
  P6_c75_keep1_lhc_x0yjump_last
  P6_c50-75_keep1_lhc_x0yjump_mean
```

The most defensible current statement is:

> Temporal prefix stability improves clean-free basin selection, but a single keep-one detector is not yet empirically guaranteed. The strongest current validated policy is temporal stability plus a keep-two uncertainty buffer, while triple temporal keep-one remains a promising aggressive low-cost candidate.
