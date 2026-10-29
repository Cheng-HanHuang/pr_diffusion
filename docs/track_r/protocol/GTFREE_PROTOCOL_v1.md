# GT-free best-of-K protocol, v1 (FROZEN)

**Frozen by the planner on 2026-10-29 (ET). This commit is the freeze; its hash and the SHA-256 of this file and of `gtfree_protocol.py` are recorded in `PREDICTIONS_REGISTRY.md` (section "Frozen protocols") and in prereg A11.**

**Why.** On Oct 29, Cheng chose to reframe the paper around a deliverable: a ground-truth-free way to run and report best-of-K, plus the finding that GT-free selection is predictable while unsolved images are not (prereg A11). This file fixes that protocol and its held-out test **before** any held-out per-image data exists in the planner's hands.

**What the planner had seen at freeze:**
- **Development data:** the FAST FFHQ 64-run strings (A9), the historical A8b prefixes, and the D1 phase-1 strings with CLIP decisions.
- **D1 phase 2: aggregates only.** Policy curves and class rates (XP `70b290f`); no per-image strings, residuals or CLIP margins.
- **D2/D3 (ImageNet DAPS-4K / SITCOM, 4 runs): aggregates only.**
- **No B16 (SITCOM, 16 runs) or F8 (DAPS, 8 runs) outcome of any kind.** E′ (DAPS-4K FFHQ) is not scored yet.

## 1. The deployable rule

Inputs:
- the measurement y and the noise level σ;
- a solver S;
- a per-solver threshold τ_S;
- a zero-shot orientation scorer u.

The rule never uses the ground truth.

```
for j = 1, 2, ..., N_MAX:
    x_j  <- S(y; fresh seed j);   rho_j <- ||A(x_j) - y|| / sqrt(m)
    if #{i <= j : rho_i <= tau_S} >= M:  stop
x_hat  <- argmin_j rho_j                                  (residual pick, E0)
FLAG_C <- no drawn run has rho_j <= tau_S                 ("no consistent reconstruction")
delta  <- u(R_111 x_hat) - u(x_hat)                       (zero-shot orientation margin)
output <- R_111 x_hat if delta > 0 else x_hat             (E1)
FLAG_O <- |delta| < c_Q                                   ("orientation uncertain")
return output, FLAG_C, FLAG_O
```

**Frozen parameters:**

| Parameter | Value |
|---|---|
| M (consistent runs collected before stopping) | **2** |
| N_MAX | the panel size, i.e. all available runs, for held-out evaluation (16, 16, 8); deployment reports must state it |
| τ_S, FAST-DIPS | **0.0500829**: the FFHQ R0 calibration (A8b `S5_r0_fast`), used **unchanged on ImageNet** (cross-dataset transfer; ImageNet's own GT τ, 0.050183, is never used by the protocol) |
| τ_S, SITCOM | **0.0448503** (B22), **0.0449706** (DEV80): the registered A8b per-cohort values |
| τ_S, DAPS (400 × 5) | **0.0474649** (B22), **0.0475986** (DEV80): the registered A8b per-cohort values |
| τ_S, DAPS-4K and any solver without a calibration set | **label-free `self_mad3`**: median + 3 · 1.4826 · MAD of the per-image minimum residual over runs 0–3 of the panel's images |
| Orientation scorer, FFHQ | the frozen N2 face prompt set (CLIP ViT-L/14, pinned weights), as used for A7/A8 |
| Orientation scorer, ImageNet | G1: "a photo." vs "an upside-down photo." (ViT-L/14); G2 secondary |
| c_Q | the **Q_FLAG = 0.30** quantile of \|δ\| over the panel's outputs that are not FLAG_C (label-free, transductive) |

`self_mad3` is reported as a **secondary** variant on every panel.

**τ is a solver constant, not a certificate.** DAPS and SITCOM fit the data below the noise level (paper Sec. 6, C-A6). So τ_S is calibrated per solver, once, and FLAG_C means "no run reached this solver's own consistency level", not "the output is wrong".

## 2. Offline evaluation (reference implementation `gtfree_protocol.py`)

**Sequential sampling is simulated.** Runs of an image are exchangeable draws (A9.2(a); D1 phase 1 vs phase 2 class rates show no shift), so random orders of an image's n available runs simulate sequential sampling with N_MAX ≤ n. Settings: R = 4000 orders per image, seed 20261029.

**Labels and oracles are used only for scoring:**
- "success" = the output is GOOD (`G`; with the orientation step, the chosen orientation is GOOD);
- "unsolved image" = no GOOD run in the panel (E0) or no orbit-GOOD run (with the orientation step).

**Baselines** are exact all-subset U-statistics on the same runs:
- fixed best-of-K E0 and E1, for K = 1..n;
- O_raw and O_orbit.

The fixed curve is linearly interpolated at non-integer K.

**Outcomes:**

| Outcome | Definition |
|---|---|
| `success_all` | Success over all images (flagged outputs included) |
| `runs_mean` | Mean runs per image |
| `gain_vs_fixed_equal_runs` | success_all − fixed best-of-K at K = runs_mean |
| `gap_vs_fixed_nmax` | success_all − fixed best-of-N_MAX |
| FLAG_C expected count, precision and recall | For unsolved images |
| `coverage_unflagged`, `success_unflagged` | Neither flag raised |
| `success_flag_o_only` | Outputs with FLAG_O but not FLAG_C |

**Input schema.**
- Strings CSV: `image_id`, `cohort`, then `labels_by_run` (`;`-joined G/O_g/O_p/C/I, run order) or `run_labels` (one character per run), then `rho_signed_by_run` (`;`-joined).
- CLIP CSV: `image_id, run, <chosen_good col>, <margin col>`, one row per run, run index matching the strings.

## 3. Development results (in-sample; all seen before the freeze)

Primary configuration (M = 2, N_MAX = panel size, frozen τ). Outputs are in `dev_v1/`, reproducible with `run_protocol_v1_dev.sh`.

| Panel | Runs/img (N_MAX) | τ | Success | Mean runs | Fixed best-of-K at equal runs | Gain | Fixed best-of-N_MAX | O_orbit at N_MAX | FLAG_C (expected images) | of which unsolved | Unsolved images |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FAST, FFHQ, 180 img | 64 (64) | 0.05008 | 0.913 | 8.39 | 0.891 | +0.022 | 0.939 | 0.961 | 8.0 | 6.0 | 7 |
| FAST, FFHQ | 64 (16) | 0.05008 | 0.893 | 4.85 | 0.856 | +0.037 | 0.912 | 0.930 | 12.8 | 6.0 | 7 |
| FAST, ImageNet D1-P1 (E1, G1) | 4 (4) | 0.05008 | 0.497 | 2.42 | 0.479 | +0.018 | 0.550 | 0.660 | 6.0 | 3.0 | 34 |
| DAPS, FFHQ DEV80 | 4 (4) | 0.04760 | 0.810 | 2.95 | 0.761 | +0.049 | 0.825 | 0.863 | 7.0 | 6.0 | 12 |
| SITCOM, FFHQ B22 | 4 (4) | 0.04485 | 0.862 | 2.35 | 0.845 | +0.017 | 0.890 | 0.960 | 4.0 | 0.0 | 5 |
| SITCOM, FFHQ DEV80 | 4 (4) | 0.04497 | 0.585 | 2.53 | 0.587 | −0.001 | 0.688 | 0.725 | 5.0 | 4.0 | 25 |
| NP, FFHQ DEV80 | 4 (4) | 0.05050 | 0.683 | 2.84 | 0.657 | +0.026 | 0.700 | 0.713 | 22.0 | 18.0 | 23 |
| FAST, FFHQ, label-free τ | 64 (64) | 0.05024 | 0.884 | 4.73 | 0.854 | +0.030 | 0.939 | 0.961 | 1.0 | 1.0 | 7 |
| FAST, ImageNet, label-free τ | 4 (4) | 0.05043 | 0.477 | 2.18 | 0.466 | +0.011 | 0.550 | 0.660 | 3.0 | 3.0 | 34 |

**ImageNet D1-P1, orientation flag.** The unflagged outputs (66.6% of images) are GOOD 65.3% of the time. That compares with 50.7% for all outputs without FLAG_C, and 15.2% for those with FLAG_O only.

**Frontier (FAST FFHQ).** Over M ∈ {1, 2, 3, 4} × τ multipliers {0.997, 1, 1.003}, the protocol beats fixed best-of-K at equal mean runs at **12 of 12 points**: by +0.8 to +4.7 points at N_MAX = 64, and +1.4 to +7.6 at N_MAX = 16.
- At M = 2 and τ × 0.997, it reaches 0.935 at 16.5 runs per image; fixed best-of-64 gives 0.939 at 64 runs.

**Reading.** Stopping gains where consistency predicts correctness, i.e. where failed runs are mostly INCONSISTENT:
- FAST and DAPS on FFHQ: about 8–10% of failed runs are CONSISTENT (C-A5).
- SITCOM (57% CONSISTENT) and ImageNet FAST (C is 47% of all runs) gain little: an unsolved image produces consistent-but-wrong runs and stops early.

## 4. Held-out test and frozen hypotheses

**Primary held-out panels**, each with M = 2, N_MAX = panel size, the frozen τ and the orientation step:

| Panel | Content |
|---|---|
| **D1-P2** | ImageNet FAST, 100 images × 16 runs, G1. Secondary: runs 4–15 only, via `--run-range 4:16` |
| **B16** | SITCOM FFHQ, 180 images × 16 runs, face scorer, per-cohort τ, pooled over cohorts and per cohort |
| **F8** | DAPS 400 × 5 FFHQ, 180 images × 8 runs, face scorer, per-cohort τ, pooled and per cohort |

**Secondary panels (descriptive, no hypothesis):**
- E′ (DAPS-4K FFHQ, 4 runs, `self_mad3`);
- D2/D3 (ImageNet DAPS-4K / SITCOM, 4 runs; τ = `self_mad3`, and for SITCOM also the FFHQ τ transferred);
- FAST FFHQ 64 with face-CLIP E1;
- every panel without the orientation step (E0) and with `self_mad3`;
- the frontier (M × τ multipliers).

**Pooled panels with per-cohort τ:** run each cohort with its own τ, then average the cohort results weighted by images. The FLAG_O cut c_Q is computed per run of the script, i.e. per cohort.

**Hypotheses**, each tested once on the primary configuration:

| ID | Statement | Rationale |
|---|---|---|
| **H1** (no loss) | `gain_vs_fixed_equal_runs` ≥ −0.01 on each of D1-P2, B16 and F8 | Stopping should never cost accuracy at equal compute |
| **H2** (gain tracks failure type) | `gain` on F8 ≥ +0.02, **and** gain(F8) > gain(B16) **and** gain(F8) > gain(D1-P2) | DAPS fails mostly by inconsistency; SITCOM and ImageNet FAST mostly by consistent-but-wrong runs |
| **H3** (abstention finds unsolved images) | FLAG_C precision for unsolved images ≥ 0.5 on F8 and on D1-P2, each evaluated only if the expected FLAG_C count is ≥ 2 | B16: reported, no prediction (SITCOM B22's dev precision was 0) |
| **H4** (orientation confidence) | On D1-P2: `success_unflagged` ≥ `success_given_not_flag_c` + 0.05 | Dev: +0.146 |

**Rules:**
1. The executor runs `gtfree_protocol.py` from this commit, unchanged, and the planner reproduces the run. Any discrepancy above 0.002 (Monte Carlo) is a stop.
2. Every hypothesis is reported as held or failed, in the paper's main text or supplement. There is no re-tuning: a later "protocol v2" is labeled exploratory and is evaluated only on panels it has not seen.
3. If H1 fails on a panel, the paper says so in the main text, and the protocol recommendation is restricted accordingly.
4. If H2 fails, the mechanism claim is dropped from the paper.

## 5. Files

| File | Content |
|---|---|
| `gtfree_protocol.py` | Reference implementation (stdlib + NumPy) |
| `make_dev_inputs.py` | Historical-prefix dev panels from A8b |
| `run_protocol_v1_dev.sh` | Reproduces `dev_v1/` from the four SHA-checked inputs listed in the script |
| `dev_v1/*.json` | Development outputs |
