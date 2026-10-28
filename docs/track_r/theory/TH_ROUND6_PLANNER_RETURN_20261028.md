# Track R theory executor — Round 6 return (October 28, 2026)

**Governance.** Input archive theory_upload_20261028_PM.zip checked FIRST: all 11 SHA256SUMS.txt entries match exactly, including the now-registered v2 E1, released FAST64 comparison/strings, and planner conditional-selection analysis. Read 06_PLANNER_GUIDANCE_20261028_PM.md only after verification. No docs/track_r/executors/xp/returns access and NO SITCOM B16 or DAPS F8 post-historical run observed. v1/v2/v2 E1 and the planner's registration remain unchanged.

## A. Locked-plan FAST interpretation DONE

Read docs/track_r/theory/TH1_FAST64_LOCKED_INTERPRETATION_20261028.md (committed at theory 777d92b). It declares **NEITHER model wins**, per the locked 7a09153 rule: primary E0 coverage race 6/21 vs S2 15/21; O_raw 6/21 and O_orbit 4/21. Explicit failure is **generation**: seven never-GOOD images, six all-INCONSISTENT across 64. The companion predicted 22.8 of 25 zero-prefix images would eventually become GOOD; only 18 did. DEV80 0/4 predicted 89% but realized 70%. The empirical tail of near-hopeless images was not identified from four runs.

Conditioning on the *observed* class composition (explicitly exploratory after outcome, NOT prospective recalibration), primary race predicts pooled K64 E0 0.9526 vs 0.9389 observed (~+1.37 pp), tau1 predicts 0.9379, and S2 predicts 0.8773. S2's unconditional coverage came from cancellation of a too-pessimistic selection law and a too-optimistic generator. Four K64 E0 selection losses have 1-4 GOOD runs, with two ORBIT winners and two CONSISTENT winners. Do not overclaim E1: its FFHQ outcome is still pending.

## B. Optional v3 code SENT EARLY; REGISTER SEPARATELY BEFORE B16/F8

Committed under theory branch (implementation @6a962113, full protocol @1d51d9e2):
- scripts/track_r/theory/fit_th1_fast_difficulty_v3.py
- scripts/track_r/theory/check_th1_fast_difficulty_v3.py
- scripts/track_r/theory/seal_th1_fast_difficulty_v3.py
- scripts/track_r/theory/run_th1_fast_difficulty_v3.sh
- docs/track_r/theory/TH1_V3_FAST_DIFFICULTY_PROTOCOL_20261028.md

**Full model statement:** The held DAPS/SITCOM image prior is tilted using the released 64-run FAST per-image GOOD count via log-odds exponent 1/2, with a prespecified soft structural-zero mass for no-orbit-success images (FAST g=0 => 0.60, g1-4 => .25, g5-15 => .08, g>=16 => .015). Update that mass only using the target's historical A8 prefix. Exact per-image conditional oracle moments are mixed between the active Dirichlet and hard C/I-only component. **Same four rank laws and policies as v1, and same comparison rule.** The heuristic difficulty prior is *not identified*, and the v3 result is informed by the previously released FAST outcomes. It must be reported ALONGSIDE v1, never as a refit.

**Authorized input SHAs:** A8 197d0c8e..., A8b e099138f..., released FAST64 d616dd50.... A8 source prefix is checked against FAST64's first four labels (O_g/O_p normalized). No target B16/F8 later run is needed.

**QA:** Original A8/A8b/FAST64 bytes passed all guards. A local execution of the committed v3 algorithm using faithful reconstructed sibling-module equivalents passed BOTH:
- reduced smoke 4/2/1, and
- full budget **250 main simulation draws, 60 image cluster bootstrap iterations, 3 bootstrap simulations**.
The standalone v3 contract reports **TH1_FAST_DIFFICULTY_V3_CONTRACT_PASS 496 132 36**. There are 360 per-image difficulty audit records. The local module implementations of rank fitting and Dirichlet fitting were reconstructed to reproduce the repository functions; the planner must make a final run with ACTUAL repository siblings before freezing and register that exact output SHA/commit/UTC.

**Planner production run:** On clean theory branch, export PY, A8, A8B, FAST64 absolute paths, optionally WORK scratch, then run bash scripts/track_r/theory/run_th1_fast_difficulty_v3.sh. It performs source QA, simulation (250/60/3), contract, sealed v3 JSON and a new theory Git commit. Push and register before any B16/F8 unblinding. Code is ready ahead of Oct31 06:00 ET handoff; hard registry deadline Oct31 12:00 ET. If any assertion fails, fix BEFORE freeze with dated rationale; do not open held results or silently retro-edit v1.

## C. Optional C-T8 numerical identification supplement: DONE as exploratory population analysis

Committed scripts/track_r/theory/check_th1_identification_lp.py and docs/track_r/theory/TH1_CT8_DATA_SPECIFIC_MOMENT_BOUNDS_20261028.md. Only original FAST A8 four-run GOOD-count frequencies enter the optimization. Under conditional iid Bernoulli runs and iid images within the cohort, five categorywise 99% exact Clopper-Pearson intervals produce a joint >=95% region by Bonferroni; grid-LP constraints are expanded by Bernstein polynomial derivative bounds, and a target-Lipschitz correction gives a **continuum mixing-distribution outer enclosure**.

Result for the POPULATION oracle \(O_{\rm raw}(64)\), not the fixed-image conditional v1 estimand: **B22 [0.8696,1.0000], DEV80 [0.6113,1.0000]**. Both are broad, explaining why tiny calibration sample depth cannot identify a high-K structural-zero mass. DEV80 enrichment potentially breaks iid image sampling: its nominal 95% coverage is therefore conditional and should not be quoted as an unconditional theorem. This analysis was conceived after seeing the FAST64 miss, so it is supplement evidence, not a new preregistered prediction.

## D. Upcoming paper lane

The registration of ImageNet v2 E1 is acknowledged (theory 9cb57f5, SHA-256 13ff550ca42c0941c430eff75e1f9ecc107dbe3801ee19d355371518980ec406; planner 7d1882c). No independent theory interpretation of D1 phase 2 is possible until the planner relays the result. Prepare Supplement S1 source-accurate solver qualifications, S2 statistical estimators/uncertainty and half-page Sec5, explicitly reporting the failed v1 forecast and posthoc conditional selection test rather than recasting the latter as validation of the entire forecasting model.
