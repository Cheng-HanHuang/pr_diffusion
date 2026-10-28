# Track R theory executor: Round 4 return — October 28, 2026

**Pre-unblinding and access statement.** All work below was performed without opening docs/track_r/executors/xp/returns, FFHQ K>4, D1 phase-2, D2/D3, or A5.5 outcomes. The user-uploaded ZIP was integrity-checked before reading its guidance. The theory branch was confirmed at registered head 340dc5dcbcde96df4edc938f1b807faedc18936b BEFORE this round's subsequent theory commits.

## 1. Registered forecasts are immutable and reviewed

SHA256SUMS.txt in theory_upload_20261028.zip verified **all nine member files** by recomputing each digest. The v1, companion and v2 bytes match exactly their registry SHA-256s:
- v1 @8ef5f94: c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753 (976 predictions / 252 anatomy / 60 crossovers);
- companion @5782149: 74190e6722a11d1e7e16118984f89bbc5281f665629902db244b5740cd783462 (14,800 image records);
- v2 @340dc5d: 8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55 (36 forecasts / 12 anatomy).

Examined cfefd00 threshold-display fix: consistent with storing A8b per-image tau to 6 significant digits, model unchanged. Examined dd5c5e6 v2 sealer and launcher: source-input SHA guard, draft contract, run provenance, abstention, output immutability and UTC freeze correctly recorded; no numerical change detected. **NO ERRATUM NEEDED based on this audit.** This source/packaging review is not a second independent execution of the original production freeze.

**Terminology correction only:** ImageNet variant tail is the **transferred pooled FFHQ 5/205 C-over-G tail**, and is *optimistic* in ImageNet: E0 tail/race are 0.4132/0.3882 at K8, 0.4173/0.3993 at K12, 0.4152/0.4049 at K16. In text and figures use "FFHQ-tail-transfer sensitivity", NOT a pessimistic stress envelope. The frozen JSON field remains tail. Full review: TH_A8_REGISTERED_FREEZE_PACKAGING_REVIEW_20261028.md.

## 2. One-page interpretation locked prior to results

Committed TH1_PRE_RESULTS_INTERPRETATION_PLAN_20261028.md independently after v1/v2 registration but before this theory executor saw any outcome. It states what a rank-race win, an S2 win, or neither means for DAPS (nmax8), SITCOM (primary nmax16), and FAST (primary nmax64); interprets Kstar=null for DAPS as **no crossover on the registered 4/6/8 grid only**; and specifies the frozen coverage and standardized-residual rule. Intervals are explicitly model-conditional and not distribution-free high-K identification guarantees. Never refit v1/v2 after outcomes.

## 3. Optional ImageNet v2 E1 addendum prepared, NOT FROZEN

The planner's A8.3 definition is a single-candidate orientation test after E0 selects a candidate; the chosen candidate's frozen class label remains the input to its conditional probability of Good20 after G1 correction. Existing four-run phase-1 G1 decision values are preserved **exactly** in the simulated nmax16 panel. For future unknown candidates, class-specific success indicators are drawn independently conditional on a shared Beta(1,1) posterior rate per G/O/P. C/I rates are structurally zero unless the input has conflicting observed labels, in which case the loader stops for a label/GT audit. G can be miscorrected, so E1 may legitimately be BELOW E0.

New theory-owned scripts:
- scripts/track_r/theory/fit_th1_imagenet_v2_e1.py — phase-1-only checksum-guarded orientation input adapter; copied frozen v2 alpha/score weights without re-fitting or rewriting the registered v2 artifact; four variants; Beta pooled primary and image-cluster-resampled decision sensitivity; **paired E1 <= O_orbit for every simulated draw**, with frozen O_orbit means as control-variate anchors.
- scripts/track_r/theory/check_th1_imagenet_v2_e1_contract.py — verifies immutable original v2 SHA, 400-input provenance, 24 E1 forecast records (4 structural variants x 2 decision uncertainty arms x 3 K), per-class Beta counts, E1 point <= registered O_orbit, interval bounds and embargo declarations.
- scripts/track_r/theory/seal_th1_imagenet_v2_e1.py — separate SHA-guarded, nonoverwriting addendum seal. Does not mutate the registered v2 JSON.
- scripts/track_r/theory/run_th1_imagenet_v2_e1.sh — clean-theory-branch controlled production runner, then contract, seal and Git commit. Inputs: PY, IN (original phase1 CSV), CLIP (400-run decision CSV), CLIP_SHA256. Pass optional explicit column overrides to the launcher as command-line arguments.
- scripts/track_r/theory/check_th1_imagenet_v2_e1_synthetic.py — source-free numerical invariant test designed to exercise all variants and demonstrate paired E1/orbit ordering.

**Execution status:** the CLIP decision CSV was NOT yet in the authorized theory_inputs directory at this round's last check. Hence the new model, schema adapter and tests have NOT been run on the actual decision file; the addendum is **not registered or frozen**. When the executor commits and Cheng uploads the permitted 400-run file + SHA, validate its exact column schema, run synthetic/production tests, seal and register a new key v2 E1 by Oct 29 12:01 ET. If it has not arrived by approximately Oct 29 06:00 ET, issue an explicit NO-ADDENDUM statement; the registered v2 E1 abstention remains valid. Do not access D1 phase-2 outcomes to patch any estimate.

## 4. Paper lane

Continue to prepare Supplement S1/S2 and ~half-page Sec.5 under the planner's Nov 3–6 schedule; no re-analysis of embargoed XP outcomes. Required citations/claim boundaries remain classical Fourier symmetry, competing-rank theory, Stein noise-fitting, Gaussian posterior trace laws, and Gauss-Radau moment nonidentification versus the new, prospective cross-native-solver phase-retrieval characterization.

**Specific request to Cheng:** supply the authorized XP_D1_P1_K4_CLIP_DECISIONS.csv (phase-1 indices 0–3 only) with its independently recorded SHA-256 once published, to enable actual v2 E1 production validation. No forbidden data are needed.
