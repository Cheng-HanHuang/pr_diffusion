# A12 optional v4 model lock BEFORE permitted ImageNet 0–15 input arrives

**29 October 2026. PRE-INPUT DECLARATION.** At writing, the authorized docs/track_r/executors/xp/theory_inputs/ directory on codex/prdiff-track-r-xp-20261026 exposes only the historical 0–3 D1 CSV and receipt, **not** the 0–15 calibration strings or G1 flags. No XP-D4 run 16–63 outcome, XP-G outcome or xp/returns has been read. This note locks modeling options before inspecting the new calibration file. The original v2 and v2 E1 forecasts remain immutable.

## Primary model and identifiability choice

**No structural-zero point mass in v4.** This explicitly selects the original v2 family, rather than introducing a new speculative spike probability after seeing pilot counts. Fit one 5-class Dirichlet image-level prior alpha to the 100 ImageNet FAST Good20 16-run count vectors, with existing +.25 component smoothing and unchanged optimizer from fit_th1_dirichlet.py. Conditional on EACH fixed image's actually observed runs 0–15, use its Dirichlet posterior C_i+alpha to simulate only unknown future runs. A 16-run pilot improves per-image identification but does not prove q_i>0 or exclude a near-zero-success subpopulation. High-K interval failure is therefore admissible and scientifically informative.

Calibrate within-image class PL score rates from **only the first 16 residual rankings**, using exactly the original v2 pairwise composite BT fit with logweight prior N(0,2.5^2), w_G=1, w_P=w_O, and S1 surrogate w_I=0 with finite-rate treatment of any **already observed** I-first exceptions. The four frozen variants are race (primary), S2 (exchangeable passing), tail (FFHQ 5/205 transfer, explicitly stress/sensitivity), and race_tau1 (N(0,1^2) regularization sensitivity). Do not reweight based on XP-D4 results.

Fit phase-1 G1 orientation-success probabilities ONLY from the 0–15 allowed CLIP decisions: independent Beta(1,1) class-conditional rates for future G/O/P; C/I successes structurally zero if consistent with class definitions. **Preserve the observed G1 outcomes for all 16 original runs exactly**, including GOOD misorientation. Label a conflicting 0–15 C/I G1-success record as a data/label audit STOP, not automatic model adjustment. The sensitivity is bootstrap-resampling the 100 IMAGE clusters to estimate class-specific G1 success rates, always evaluating the original identified cohort.

## Panels and exact estimands

Targets: same 100 fixed ImageNet Good20 images; **primary** nmax=32, K=(16,24,32); **secondary** nmax=64, K=(32,48,64). Each completed panel contains the SAME observed first 16 runs in their original run-index slots, plus only new iid future candidates. A uniform K-subset is scored by the minimum residual rank using the registered exact combinatorial U-statistic weights \(\binom{n-r-1}{K-1}/\binom nK\).

Policies: E0 = chosen original-orientation GOOD, E1_CLIP = chosen candidate after its G1 orientation action is GOOD, O_raw = any GOOD within K, O_orbit = any G/O/P within K. **E0 may exceed E1 on some images**, because CLIP can flip GOOD incorrectly; but E0<=O_raw and E1<=O_orbit must always hold.

For each panel and variant, integrate the Dirichlet posterior against the finite-panel hypergeometric subset to obtain **exact** O_raw/O_orbit posterior means, common across race/S2/tail/tau1. Report E0 as \(O_{\rm raw}^{\rm exact}-\mathbb E_{\rm MC}[O_{\rm raw}^{\rm panel}-E0]\) and E1 as \(O_{\rm orbit}^{\rm exact}-\mathbb E_{\rm MC}[O_{\rm orbit}^{\rm panel}-E1]\), with the same **paired control-variate shifts applied to draws**. Require simulated oracle-vs-exact deviations below 0.01 before freezing; require modelwise E0<=O_raw, E1<=O_orbit and closed-form posterior oracle equality across variants. Predictive 95% intervals represent **realized finite completed panels conditional on the stated parametric model**, not distribution-free identification bands.

Primary comparison: across 3 K values x E0+E1 on n32, count observations in frozen [lo95,hi95]; standardized residual tie-break \(|obs-point|/\max((hi-lo)/3.92,0.005)\). Report secondary panel and all sensitivity arms, but do not promote them to primary after seeing the held outcomes.

Expected JSON: 4 variants x 2 panels x 3 K x 4 policies = **96 primary policy rows**, plus **24 separate E1 cluster-sensitivity records** if using the existing v2-E1 reporting convention; 24 four-variant panel/K anatomy records. The E1 pooled-Beta primary and cluster sensitivity must be distinguishable in schema, not silently counted twice. Lock the exact output contract before production.

## Access and registration gate

To permit a v4 freeze, the executor must publish in docs/track_r/executors/xp/theory_inputs/ a reproducible 100-image CSV with **16** candidate labels and residual rank permutations, an EXACT 1,600-run phase1 G1 decision file, and each SHA-256/UTC/receipt. Verify all 100 first-four candidate labels, ranks and run indices against the previously registered D1 P1 K4 CSV (SHA c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1). Only then fit. Do not use source D4 output or unknown held run indices 16–63.

The planner runs final production, source QA, 0.01 oracle anchor contract, then commits/pushes/registers v4 with exact SHA256, code head and UTC before inspecting any held result. Deadline Nov 2 12:00 ET; if the inputs do not allow fully validated production before that deadline, **decline** and keep v2 unchanged. The pre-input declaration remains valuable as a historical record of why no spike was fit.
