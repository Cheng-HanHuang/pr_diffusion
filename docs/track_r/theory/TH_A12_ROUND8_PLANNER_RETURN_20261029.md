# Track R theory executor — A12 / Round 8 return (2026-10-29)

**PRE-OUTCOME ACCESS STATEMENT.** Opened only the 2026-10-29 PM user ZIP after verifying all THREE entries in SHA256SUMS.txt. Read 08_PLANNER_GUIDANCE_20261029_PM.md and frozen 4_protocol.tex. Checked the allowed theory_inputs directory on XP branch: at this round it held the previously authorized D1 P1 **0–3** CSV and receipt only; NO D1 **0–15** pilot strings/CLIP file had been published. No XP-D4 16–63, XP-G result, xp/returns or held B16/F8 outcomes accessed. Prior protocol code and registered forecasts unchanged.

## 1. Accepted A11 direction and current main-text boundary

Planner accepts Proposition 1 and all three exact protocol identities, and reports frozen 4,000-permutation evaluator within 3e-4 success, .003 mean-run and .03 FLAG_C-image tolerance of the exact finite-panel calculation on five dev panels. The equal-budget heterogeneity counterexample is accepted. **No new edit to main paper Section 4 or frozen protocol**; theory-owned S1/S2/S3 drafts (as committed in round 7) remain available for integration by Nov 6. The paper's core is the GT-free best-of-K protocol, with selection accuracy versus generation nonidentification explaining its outcomes. Fourier orbit symmetry is classical supporting material.

## 2. XP-G pre-score note committed

docs/track_r/theory/TH_XPG_TAU_SIGMA_PRE_SCORE_20261029.md was committed **before accessing any XP-G outcomes**. H-G1's linear reference thresholds are tau(0.02)=.02003316 and tau(.10)=.1001658, each with a 1% relative band.

The normalized signed-residual identity
\[
 \rho_\sigma/\sigma=\|Z-(A(\hat x_\sigma)-a)/\sigma\|/\sqrt m
\]
implies exact quantile scaling ONLY under sigma-invariance of the normalized reconstruction displacement, and the appropriate class/selection conditioning. FAST's hard-projection epsilon=.05 is *fixed*, so epsilon/sigma changes from 2.5 at sigma=.02 to .5 at sigma=.10, and the returned denoiser estimate need not be the projected iterate. Thus C-T9 gives a plausible **noise-dominated scaling null**, not a theorem that the especially tight 1% H-G1 must pass; the pre-score classification is **unresolved, substantive risk of failure**, with no principled directional deviation. Do not turn this conditional statement into a post-result prediction.

## 3. Optional v4 — MODEL PREDECLARED and ENGINEERING STAGED, NOT FROZEN

Committed the pre-input declaration TH_A12_V4_PREINPUT_MODEL_LOCK_20261029.md **before the new 16-run calibration became available**. The primary model has **NO structural-zero spike**, preserving the originally registered v2 Dirichlet class-rate family and class-score race. It will refit alpha, G/O/C residual race rates and a per-class G1 Beta(1,1) orientation posterior using ONLY the 100 fixed ImageNet images' authorized runs 0–15. The original 0–15 class strings, residual rank and CLIP decisions are held **fixed** in the simulated 32/64 completed panels; only 16–31 or 16–63 candidates are simulated, never read.

Targets primary nmax32 K16/24/32, secondary nmax64 K32/48/64, four variants (race, S2, FFHQ-tail-transfer, race_tau1), four policies (E0, E1_CLIP, O_raw, O_orbit) plus separately reported E1 image-cluster sensitivity. Point estimate uses the **exact posterior finite-panel oracle** minus the jointly simulated oracle-to-policy loss; check all A9.5 oracle anchors within .01 and every E0<=O_raw, E1<=O_orbit. Comparison primary E0+E1 predictive interval coverage, standardized-residual tie-break; secondary/tail/prior sensitivity disclosed.

New code, on theory branch:
- scripts/track_r/theory/fit_th1_imagenet_v4.py
- scripts/track_r/theory/check_th1_imagenet_v4_contract.py
- scripts/track_r/theory/seal_th1_imagenet_v4.py
- scripts/track_r/theory/run_th1_imagenet_v4.sh

**Status:** Python syntax and Bash syntax passed locally; a synthetic input constructed from the *authorized original four-run IDs and ranks plus invented 4–15 labels/decisions* passed the complete 120-policy/24-anatomy/8-anchor structural contract, with only **simplified local stubs** for sibling fitting/simulation modules. This is **NOT a full native-repository production QA** and is not numerical v4 calibration. The actual permitted 0–15 input and exact SHA-256 are still required, and the planner must run all native modules end-to-end. The code enforces exact 100 image IDs/first-four ranks, 16 labels per image, all 1600 phase-1 G1 decisions, verified input SHAs, no 16+ input indices, and no overwriting prior v2. The clean-branch launcher (PY, PREFIX4, PILOT, PILOT_SHA256, CLIP, CLIP_SHA256) fits 250/60/3, validates 120/24/8, seals, and commits. Planner must push/register v4 **before** D4 unblinding by Nov 2 12:00 ET.

**Important unresolved production risk:** 16-run rank conditioning (especially the inherited exhaustive C-tail latent-subtype enumeration) may be computationally expensive for highly CONSISTENT images; the actual source schema may differ from the conservative expected per-image CSV + per-run G1 decision format. Do NOT bypass SHA/contract or reduce final run budgets secretly. If these cannot be resolved on authorized data before deadline, explicitly DECLINE v4 instead of publishing unvalidated forecasts. No calibration output or v4 file hash exists yet.

## 4. Next permitted exchange

Cheng/planner should deliver ONLY the committed 0–15 ImageNet per-image class/rank strings and 0–15 G1 CLIP decisions, plus SHA256/receipts, in docs/track_r/executors/xp/theory_inputs/. Verify user-released bytes and inspect the actual schema before production. No XP-D4 output file, XP-G scored result or xp/returns needed. S1–S3 theory-owned drafts remain available; final supplement edit deadline Nov 6.
