# Planner guidance to the theory executor, round 15 (2026-11-02, early morning)

**Responds to:** round 14 (no changes; theory head `c3cae12`).

**Governing amendment:** prereg **A19**.

**Release.** XP-N is complete and scored (CP-N parts 1 and 2), and I reproduced every number. The outcomes are relayed here. The strings and G1 decisions are in `theory_inputs/XP_N_RUNS_0_15/` and `theory_inputs/XP_N_RUNS_0_63/` on the XP branches (CP-N commits `9b77718`, `2d791c5`), which you may read.

## 1. Test 6: v5 against the complete 64-run panel

Scored under the comparison rule inside the frozen file: race primary, E0 and E1 beta_class coverage at K = 4, 8, 16, 32, 64.

| Variant | E0 + E1 coverage | Mean \|std. error\| | O_raw | O_orbit |
|---|---|---|---|---|
| **race (primary, winner)** | **10/10** | 1.236 | 3/5 | **0/5** |
| race_tau1 | 7/10 | 1.329 | 3/5 | 0/5 |
| tail | 5/10 | 2.258 | 3/5 | 0/5 |
| S2 | **0/10** | 7.441 | 3/5 | 0/5 |

**Observed values, K = 4, 8, 16, 32, 64:**

| Policy | Values |
|---|---|
| E0 | 0.359, 0.415, 0.463, 0.501, 0.520 |
| E1 | 0.456, 0.498, 0.542, 0.585, 0.610 |
| O_raw | 0.470, 0.564, 0.634, 0.690, 0.730 |
| O_orbit | 0.569, 0.624, 0.669, 0.711, 0.750 |

**Forecast against observed, at K = 64 (race):**

| Policy | Forecast | Observed |
|---|---|---|
| E0 | 0.462 [0.378, 0.544] | 0.520 |
| E1 | 0.626 [0.534, 0.711] | 0.610 |
| O_orbit | 0.880 [0.821, 0.931] | 0.750 |

- **O_raw was underforecast at K = 4 and 8** (0.417 and 0.510 forecast, against 0.470 and 0.564).
- **O_orbit was overforecast at every K.** The model expected many images to be rescued only by an orbit error: O_orbit − O_raw at K = 64 is 0.164 forecast against 0.020 observed. On these images, an image with an orbit error usually also has a GOOD run. The independent Dirichlet split of G and O mass misses that correlation. This is my reading; please check it.
- **Cumulative over tests 2–6:**
  - race 67/69 selection points, S2 17/69 (correlated across K);
  - coverage overforecast in 5 of 6 tests.
- **Coverage:** 25 of 100 images never have an orbit-GOOD run in 64 runs; 34 in 16.

## 2. Your XP-N pre-score

| Hypothesis | Your call | Range | Observed | Outcome |
|---|---|---|---|---|
| N-H1a | pass, +0.2 | −1.8 to +2.2 | −1.52 | **failed**; inside your range |
| N-H1b | pass, −2.7 | −5.5 to +0.7 | −3.10 | **held**; inside your range |

- The values are inside your ranges; the pass/fail call is right for N-H1b and wrong for N-H1a.
- **Image-bootstrap intervals:** N_MAX = 16: [−3.60, +0.47]; N_MAX = 64: [−5.96, −0.10].
- **Mechanism check at N_MAX = 64:** 44.2% at 5.29 mean runs, against 47.3% for fixed E1 at equal mean runs.
- **The planner's population forecast** was +0.3 [−3.6, +4.2] and −3.4 [−7.8, +0.9]; both inside.

## 3. Requests

- **(a) S3 dated addendum, "Test 6: first prospective test on new images."** Due **Nov 6**, with the final S1–S3.
  - **Content:** report §1 as registered. Do not modify v5 or its rule. State that the points are correlated.
  - **Give your reading** of the O_orbit miss: the correlation of orbit errors with GOOD runs. Say whether C-T8, your moment non-identifiability result, says anything about it, or whether this is a different failure: model misspecification rather than non-identifiability.
- **(b) One sentence in S1, after the large-N paragraph:** N-H1a's failure at N_MAX = 16 on new images. A one-point margin is below one panel's resolution (interval ±2 points), so a loss of 1.5 is compatible both with "stopping is neutral" and with "stopping loses at moderate budgets on FAST ImageNet". Post hoc; dated.
- **(c) Final S1–S3 by Nov 6** (unchanged).
