# A10 v3 specification — FAST64 difficulty prior for held FFHQ B16 / F8

**Registered v1 remains untouched.** Prepared 2026-10-28 **after FAST64 unblinding**, but **before any SITCOM B16 or DAPS F8 later-run outcome** is opened. This is a separately registrable, explicitly FAST-informed *second prospective experiment*, not a retroactive fix to v1. Code: scripts/track_r/theory/fit_th1_fast_difficulty_v3.py; contract: check_th1_fast_difficulty_v3.py.

## 1. Target and strictly permitted data

- DAPS **F8**: 180 fixed FFHQ images, B22 original 2-run prefix, DEV80 original 4-run prefix, nmax8, K=4/6/8. Exactly the v1 convention excludes DEV80/pooled K4 from reported predictions.
- SITCOM **B16** primary: nmax16, K=8/12/16. Secondary nmax32 with K=8/12/16/24/32, as in v1.
- Same metrics (Good25), policies (E0, E1_CLIP, O_raw, O_orbit), original prefix conditioning, fixed B22:DEV80=100:80, model-conditional realized-panel intervals, rank/selection variants and coverage rule as registered v1.
- Allowed A8 raw SHA 197d0c8e..., A8b SHA e099138f..., and the now-released FAST64 per-image strings SHA d616dd50.... This code reads **no SITCOM/DAPS extra run**, no xp/returns, and never modifies registered v1.
- FAST's released 64-run GOOD count is a **cross-solver outcome covariate**. Its use means v3 is *not* independent of the released FAST dataset. All predictions for the held target solver families remain prospective.

## 2. Generation model fixed before B16/F8 results

For each fixed image and cohort, use the original A8 target-solver class counts \(C_i\) and the cohort Dirichlet fit \(\alpha_c\). Let \(g_i^{F}\) be FAST's newly released number of GOOD candidates among exactly 64 runs; let \(\bar g_c^{F}\) be its cohort mean. Define
\[
 p_i=(g_i^{F}+.5)/65,\quad
 \bar p_c=(\bar g_c^{F}+.5)/65,\quad
 t_i=\exp\!\left\{\mathrm{clip}_{[-4,4]}\left[\tfrac12
   (\mathrm{logit}\,p_i-\mathrm{logit}\,\bar p_c)\right]\right\}.
\]
The active-image Dirichlet prior is \(a_i=(t_i\alpha_G,t_i\alpha_O,t_i\alpha_P,\alpha_C,\alpha_I)\). This makes FAST performance a *soft* orientation-success prior, without equating DAPS/SITCOM to FAST.

There is a point mass \(H_i=1\) for an image that **never** generates G/O/P. Its prior is 0.60 for FAST \(g=0\), 0.25 for 1–4, 0.08 for 5–15, and 0.015 for ≥16. These four probabilities and tilt exponent 1/2 are **heuristic locked sensitivity parameters, not identified from four target runs**. None equals one; the seven FAST-never-GOOD images are not automatically declared unsolvable for other methods.

After observing only the target's historical prefix, \(\Pr(H_i=1\mid C_i)=0\) if any G/O/P was observed. Otherwise,
\[
 \Pr(H_i=1\mid C_i,g_i^F)=
 \frac{\pi(g_i^F)}{\pi(g_i^F)+(1-\pi(g_i^F))(a_C+a_I)_{n_0}/(\sum_j a_j)_{n_0}}.
\]
If \(H_i=1\), future runs only draw C/I from the corresponding posterior two-class Dirichlet. Otherwise they draw all five classes from \(\mathrm{Dirichlet}(C_i+a_i)\). This is a *generative structural-zero hypothesis*; seeing no GOOD in 64 FAST runs does not prove true probability zero.

An exact beta–hypergeometric conditional expectation of \(O_{\rm raw},O_{\rm orbit}\) is obtained by mixing the **active** posterior non-success moment with the **hard** moment equal to one, while holding the observed historical prefix fixed. The oracle means are shared exactly by all selection variants.

## 3. Selection, uncertainty and comparison lock

No rank weights were learned from B16/F8 outcomes. The default race is the original family-specific A8 composite BT/Plackett–Luce fit with Gaussian logweight regularization scale 2.5. S2 is the exchangeable-ranks reference, tail is the original pooled 5/205 C-over-G stress arm (not a DAPS-specific fitted tail), race_tau1 is the strict-prior sensitivity. The four original run rank permutations are conditioned upon using exponential score spacings, not reordered. \(E_1\) includes exactly G and global O, and excludes partially oriented O.

Simulation retains the old prefix and draws only future labels and class-ranking scores, then evaluates exact all-K-subset U-statistics. The point estimate is the **analytical orbit oracle mean minus the jointly simulated oracle-to-selector loss**, keeping each selector under its own oracle. Image-stratified cluster bootstraps refit only the A8 class priors and rank weights, keeping the FAST-difficulty rule fixed and always forecasting the original image IDs. Intervals predict the **realized future panel** under this model, not distribution-free identification bands or uncertainty about the locked hyperparameters.

**Comparison rule:** on each family PRIMARY panel, count observed E0 and E1_CLIP points within preregistered [lo95,hi95] and break ties by mean absolute standardized residual using max((hi95-lo95)/3.92,0.005). Report S2, tail and race_tau1, and the secondary SITCOM panel separately. Do not select between v1 and v3 after seeing B16/F8; present both even if v3 loses.

## 4. Execution, QA and access gate

Source and contract were smoke-tested on the real permitted A8/A8b/FAST64 files: **496 policy forecasts, 132 anatomy records, 36 crossovers, 360 per-image difficulty provenance rows**, with oracle point agreement across all four selection variants, ordering and decomposition checks. This smoke used reduced simulation/bootstraps; it is NOT a sealed primary numerical forecast.

For production, export PY, A8, A8B and FAST64 and run scripts/track_r/theory/run_th1_fast_difficulty_v3.sh on a clean theory branch. It performs full 250/60/3 simulation, invokes check_th1_fast_difficulty_v3.py, then the checksum-bound sealer and Git commit. **Planner must push and register the resulting commit, SHA-256 and UTC time before releasing any B16/F8 outcome.** The code is available well before the October 31 06:00 ET target; the hard registration deadline is October 31 12:00 ET. Any failure of production QA requires correction **before** a v3 freeze, not retroactively afterward.
