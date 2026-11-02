# A19: planner reproduction and scoring of XP-N (2026-11-02)

**Inputs.**

| Return | Commit | Files |
|---|---|---|
| CP-N part 1 | XP `9b77718` | `XP_N_CPN16_STRINGS.csv` `b0ef31bc…`, `XP_N_CPN16_G1_DECISIONS.csv` `3420633a…` |
| CP-N part 2 | XP `2d791c5` | `XP_N_CPN64_STRINGS.csv` `4b256e53…`, `XP_N_CPN64_G1_DECISIONS.csv` `f63c3d05…` |

- **Checksums:** every SHA256SUMS line verifies.
- **Consistency:** runs 0–15 of the 64-run files equal the 16-run files, labels, ρ and G1 rows alike. Runs 0–3 equal the v5 inputs (`cadf55a`).

**Code.**
- Frozen protocol `gtfree_protocol.py` at `c598176`.
- `code/score_v5.py` (planner).
- The A16 scripts `boot_panel.py` and `decomp.py`.

## Registered outcomes (A16.5, A18)

**Protocol outputs.** Part 2's two protocol JSONs (primary with the 12-point frontier, and secondary) are reproduced field for field: 392 and 287 fields, maximum difference 0. Part 1's summary values match its return.

**Registered hypotheses:**

| Hypothesis | Observed | Criterion | Outcome |
|---|---|---|---|
| N-H1a, N_MAX = 16 (runs 0–15) | gain −1.516 points; 41.49% at 3.57 runs vs 43.00% fixed E1 at equal runs | ≥ −1.0 | **FAILS** |
| N-H1b, N_MAX = 64 | gain −3.102; 44.20% at 5.29 runs vs 47.30% | < −1.0 (H1 fails again) | **Holds as predicted** |
| N-H4, N_MAX = 16 | unflagged 0.5380 vs answered 0.4217 (+11.6 points) | ≥ +5 points | **Holds** |
| N-H4, N_MAX = 64 | +10.7 points | ≥ +5 points | **Holds** |

**Secondary (label-free τ from D1 runs 0–3):** −0.17 points at N_MAX = 16, +0.10 at 64.

**Gain intervals** (image bootstrap, 5,000 replicates; exploratory):
- N_MAX = 16: −1.52 [−3.60, +0.47];
- N_MAX = 64: −3.10 [−5.96, −0.10].

**Planner forecasts (A16.5)** — all five inside or satisfied:

| Forecast | Predicted | Observed |
|---|---|---|
| P-N1, gain at 16 | +0.3 [−3.6, +4.2] | −1.5, inside |
| P-N2, gain at 64 | −3.4 [−7.8, +0.9] | −3.1, inside |
| P-N3, label-free gain at 16 | +0.3 [−2.0, +2.6] | −0.2, inside |
| P-N4, no orbit-GOOD run, K = 16 | 25% [13, 37] | 34%, inside |
| P-N4, no orbit-GOOD run, K = 64 | 22% [11, 33] | 25%, inside |
| P-N5, partial-orbit share of O_orbit − E1 at K = 64 | below 1 point | 0.0, satisfied |

The descriptive target P-N6 (E1 − E0 at K = 16, 0.170 on D1) is **not met**: it is 0.079 here, because the generic orientation test helps half as much on these images.

**Theory pre-score:** N-H1a was predicted to pass at +0.2 (range −1.8 to +2.2); it failed at −1.5, inside the range. N-H1b was predicted to pass at −2.7 (range −5.5 to +0.7); it passed at −3.1.

## Test 6: v5 against the complete 64-run panel

| Variant | E0 + E1 (beta_class) coverage | Mean \|std. error\| | O_raw | O_orbit |
|---|---|---|---|---|
| **race (primary)** | **10/10** | 1.236 | 3/5 | **0/5** |
| race_tau1 | 7/10 | 1.329 | 3/5 | 0/5 |
| tail | 5/10 | 2.258 | 3/5 | 0/5 |
| S2 | **0/10** | 7.441 | 3/5 | 0/5 |

- **Winner under the frozen rule:** race. The executor's comparison file agrees on every count and error.
- **Observed values, K = 4, 8, 16, 32, 64:**
  - E0: 0.359, 0.415, 0.463, 0.501, 0.520;
  - E1: 0.456, 0.498, 0.542, 0.585, 0.610;
  - O_raw: 0.470, 0.564, 0.634, 0.690, 0.730;
  - O_orbit: 0.569, 0.624, 0.669, 0.711, 0.750.
- **O_orbit was overforecast at every K** (0.880 forecast against 0.750 at K = 64). O_raw was *under*forecast at K = 4 and 8.
  - The model expected many images to be rescued only by an orbit error.
  - On these images, an image with an orbit error usually also has a GOOD run.
  - At K = 64, O_orbit − O_raw is 0.164 forecast against 0.020 observed.

## Decomposition (exploratory)

O_orbit − E1:

| K | Gap | Selection (consistent-wrong picks) | Orientation (G broken / rotations unfixed / partial) |
|---|---|---|---|
| 16 | 12.8 | 3.7 | 9.1 (4.5 / 4.3 / 0.3) |
| 64 | 14.0 | 7.0 | 7.0 (3.0 / 4.0 / 0.0) |

On the D1 images at K = 64 the split was 8.0 + 5.0: the orientation share grows on new images.
