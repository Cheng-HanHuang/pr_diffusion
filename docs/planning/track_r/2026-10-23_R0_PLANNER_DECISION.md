# Track R — R0 planner decision

**Date:** 2026-10-23 (America/New_York) · **Planner:** Claude
**Input:** executor return `docs/track_r/executors/r0/R0_FROZEN_COMPLETE_SCIENTIFIC_RETURN_TO_PLANNER_20261023.md` at `c8eb608` on `codex/prdiff-track-r-r0-20261021`. The R0.8 ZIP (SHA-256 `13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab`) is on PAC and with Cheng, but not yet in the repo.

**Decision:** `R0_ACCEPTED / TRACK_R_METHOD_PAPER_STOP / NO_R1 / R0N_LETTER_EVIDENCE_AUTHORIZED (CPU + ≤0.25 GPU-h) / PROJECT_WIND_DOWN_AFTER_LETTER`

---

## 0. Planner's view for Cheng

### How I view the results

**R0 did its job.** It killed the method paper cleanly for 3.4 GPU-hours, before anyone spent weeks on R1. Both pillars of the R1 plan failed:
- **Too rare.** Exact-orbit errors are only 7.1% of single-run failures (85 of 1,200): FAST 3.6%, DAPS 13.1%.
- **Wrong tool for detection.** The prior is a worse general failure detector than both a learned realness detector (UnivFD) and the free signed measurement residual.

**One result is genuinely strong.** The diffusion prior's own density resolves the exact 180° ambiguity almost perfectly:

| Selector | Orbit errors fixed | Good outputs harmed |
|---|---|---|
| Prior S (primary) | 84/85 | 0/2,040 |
| S_mid (half the cost) | 85/85 | 0/2,040 |
| UnivFD (FA-FPR's detector) | 62/85 | 220/2,040 (10.8%) |

**The most interesting finding was not a pre-registered hypothesis: failure modes are solver-specific.**
- DAPS and FAST mostly fail by *not fitting* the measurement:
  - DAPS: 78% of failures are measurement-inconsistent;
  - FAST: 89%;
  - the free signed residual flags them with AUROC ≈ 0.985.
- SITCOM's failures are much harder to tell apart by residual (AUROC 0.68): its wrong reconstructions fit the measurement about as well as its right ones.
- So your founding intuition, "fitting the measurement does not mean a good reconstruction," holds strongly for the optimization-heavy solver and much less for the sampling-style ones.
- The exact per-family split for SITCOM and NP is requested in R0-N.

**After deployable selection, little is left to fix.** Signed-residual selection plus prior repair comes within 0–2 images of the GT orbit oracle in every stratum. Almost all remaining failures are *generation* failures: no run in K is good, even after orbit changes.

| Stratum | Failures after residual selection | Fixed by repair | Generation failures |
|---|---|---|---|
| B22 DAPS, K=2 | 8 | 3 | 5 |
| B22 SITCOM, K=4 | 11 | 5 | 4 |
| DEV80 DAPS, K=4 | 14 | 3 | 11 |
| FAST, K=4 (180 images) | 26 | 1 | 24 |

- The B22 DAPS row matches the *counts* in the B22 visual taxonomy for Fresh2: 3 rotation failures; 5 chromatic, twin/ghost or collapse failures. E0 is not Fresh2's native selector, so the image identities are to be confirmed in R0-N N1b.
- Selection is not the bottleneck; sampling is.

**Bottom line:** there is no method paper here. There is one small, honest evaluation letter.

### My plan

1. **Stop the Track R method line.**
   - No R1.
   - The executor's two "narrow R1" ideas are rejected as method studies: conditional detection of consistent failures (n = 23 and 18), and cheap selective repair. Both survive only as exploratory analyses inside the letter.
2. **One bounded evidence package, R0-N** ([handoff](../../track_r/executors/r0n/00_EXECUTOR_HANDOFF.md)). CPU only, except an optional orientation baseline capped at ≤0.25 GPU-h. It produces exactly what the letter needs:
   - anatomy for the random B22 cohort versus the hard-enriched DEV80 cohort;
   - the post-selection decomposition;
   - a cheaper repair configuration chosen on DEV80 and tested on B22;
   - a CLIP zero-shot orientation baseline;
   - residual distributions;
   - committed tables.
3. **I draft the letter** once the numbers are in ([letter plan](LETTER_PLAN.md)). You and your coauthors edit.
4. **Then put pr_diffusion to rest.**
   - No new GPU campaigns unless a genuinely new idea passes a fresh literature gate.
   - Large-box inpainting: not recommended now.
   - The T1 selection-theory note stays parked; it's optional desk work if you ever want it.

### Publication image (current)

- **Working title:** "What best-of-K hides in diffusion phase retrieval: failure anatomy and exact-symmetry repair."
- **Format and venue:** IEEE Signal Processing Letters, four pages plus one page of references. Fallback: EUSIPCO 2027 (Darmstadt; listed submission deadline 2027-02-06, verify on the official site) or ICIP 2027 (deadline not yet published).
- **Expected content:**
  1. Failure anatomy of DAPS, SITCOM, FAST-DIPS and NP under deployable (GT-free) selection: orbit, measurement-consistent and measurement-inconsistent failures, random versus hard-enriched cohort.
  2. A one-line proposition: on the measurement-invariant orbit, posterior ratio = prior ratio for any magnitude-noise model. Plus the paired-noise denoising-loss estimator and its cost.
  3. Repair results against UnivFD and a CLIP zero-shot orientation baseline.
  4. Deployable selection + repair versus the field's GT best-of-K oracles. Message: report the deployable number; the remaining failures are generation failures.
  5. Practical recommendations: use the signed residual (not clipped) for selection, and do not use realness detectors for orientation.
- **What it is not:** an NP paper, a new solver, or a claim about FA-FPR's mid-trajectory restart mechanism. We compare detectors on final outputs only.
- **Honest odds:**
  - SPL acceptance: about 30–40%.
  - If SPL rejects, EUSIPCO/ICIP: about 50%.
  - Some publication within roughly nine months: about 65–70%.
- **Time box:** submission-ready by 2026-12-11. If not, retarget EUSIPCO, or stop.

---

## 1. What the planner verified

1. **Frozen files untouched.** The executor branch has no diff against the planner branch for `configs/track_r/r0_frozen_v1.json`, `docs/planning/`, `AGENTS.md` or the planner-authored Track R docs. The executor branch is based on planner HEAD `8f6302f`.
2. **Code review against the spec.**
   - `prdiffusion/track_r/prior_score.py`:
     - paired coupling is exact: member s gets input R_s(√ᾱ x + √(1−ᾱ) ε), with target R_s ε;
     - seed rule is SHA-256(`R0PRIOR|uid|t|j`) on a CPU generator;
     - S is the equal-weight mean over 8 timesteps × 4 draws; S_mid is the mean over t ∈ {300, 400, 500, 600};
     - ties within 1e-7 go to `000`;
     - no GT access.
   - `prdiffusion/track_r/r0_7_stats.py`:
     - labels, the τ rule (0.95 quantile of GOOD ρ_signed per source × family), tie-averaged Mann–Whitney AUROC (higher = failure), and the E0–E3 and O_raw/O_orbit policies match spec §3–4;
     - first-K uses recorded run order.
3. **Counts reconcile.**
   - Candidates: 3,240 = 1,400 (S1) + 960 (S2) + 128 (S3) + 32 (S4) + 720 (S5).
   - Labels: 2,040 GOOD + 85 orbit + 324 consistent + 791 inconsistent = 3,240.
   - Failures by family: 198 + 308 + 314 + 359 + 11 + 10 = 1,200.
   - Orbit errors: 26 + 11 + 25 + 22 + 0 + 1 = 85.
   - Repairs: 26 + 11 + 25 + 21 + 1 = 84.
   - DAPS: 26 + 18 + 154 = 198. FAST: 11 + 23 + 274 = 308.
4. **Canonical case G1** (A1.2 image 68539, DAPS rep2): 6.36 → 31.87 dB, paired margin 3.9e-4, z = 6.75. The absolute margin is tiny but the sign is consistent across draws, which is why the paired coupling matters.
5. **Budget:** 3.43 of 16 GPU-hours.
6. **Not yet verified by the planner:** the ZIP's full tables (O_raw values, per-family consistent counts for SITCOM/NP, all 37 policy tables). R0-N task N0 commits them to the repo.

## 2. Gate rulings (frozen v1, unchanged)

| Gate | Observed | Ruling |
|---|---|---|
| D-material (orbit share ≥ 20% of single-run failures, FAST or DAPS) | FAST 3.57% [1.34, 6.39]; DAPS 13.13% [7.02, 20.10] | **FAIL.** DAPS's upper CI touches 20%; the point estimate decides, as frozen. |
| D-repair (success ≥ 90%, harm ≤ 1%) | 98.82%; 0/2,040 | **PASS** |
| D-detect (best prior AUROC − UnivFD ≥ +0.03 for both; non-inferior ≥ −0.02) | FAST −0.096; DAPS −0.055 | **FAIL**, including non-inferiority |

**Branch taken:** the frozen table has no automatic path for this combination. The roadmap pre-committed that a failed R0 leads to "at most a short evaluation note and freeing your time, rather than another rescue cycle." That commitment governs this decision.

## 3. Answers to the executor's questions

1. **Park Track R as a method-paper route now?** Yes.
2. **A small prospective R1 for conditional rescue or the half-grid estimator?** No, not as method studies.
   - The cheap-repair question becomes an exploratory cost–accuracy analysis inside R0-N. Configuration selection happens on DEV80 images only and is evaluated on B22 images only.
   - Conditional detection appears as an exploratory table for all four families.
3. **Preserve as a bounded note and release resources?** Yes: one letter, then wind down.
4. **Frozen R1 protocol?** Not applicable.

## 4. Descriptive statistics added by the planner (not gates)

**Post-selection orbit share** = (E1 − E0 Good25 images) / (E0 failures). Net equals gross here because repair harmed zero GOOD outputs.

| Stratum | Post-selection orbit share |
|---|---|
| B22 DAPS, K=2 | 3/8 = 38% |
| B22 SITCOM, K=4 | 5/11 = 45% |
| B22 NP, K=8 | 0/6 |
| DEV80 DAPS, K=4 | 3/14 = 21% |
| DEV80 SITCOM, K=4 | 3/25 = 12% |
| DEV80 NP, K=4 | 0/24 |
| FAST, K=4 | 1/26 = 4% |

**Reading:**
- Orbit errors are rare per run but over-represented after residual selection, because the residual removes inconsistent failures and is blind to orbit errors by construction.
- This matters for DAPS and SITCOM, not for FAST, which was the R1 engine. So the D-material verdict for R1 stands.

**Field-convention gap.** O_raw — the field's GT best-of-K without alignment — versus E1 still needs the ZIP tables. It is requested in R0-N N1b.

## 5. Decisions

**D1 — Track R method paper STOPPED.** No R1 in any variant. R0 artifacts are immutable and archived.

**D2 — R0-N authorized:** letter evidence package, per the [handoff](../../track_r/executors/r0n/00_EXECUTOR_HANDOFF.md) and `configs/track_r/r0n_letter_analysis_v1.json`.
- CPU only, except optional N2 (CLIP zero-shot orientation baseline) at ≤0.25 GPU-hours.
- No new reconstructions. No new measurements. No confirmation305 access.

**D3 — ImageNet: not authorized.** R0-N records only whether ImageNet assets exist on PAC (N3). After the letter draft, the planner may propose a FAST-only ImageNet generality probe of about 2 GPU-hours, as a separate decision.

**D4 — Wind-down.** After the letter is submitted, pr_diffusion becomes dormant. Reopening requires a new written hypothesis, a fresh literature gate (post-October 2026), and Cheng's explicit go.
- confirmation305 stays locked and is not needed for anything planned.

**D5 — Letter kill criteria** (pre-registered before the R0-N numbers exist):
- **K1 — repair vanishes on the random cohort.** If, on the B22 random cohort, repair fixes 0 post-selection failures in every family, the repair contribution is gone. The letter becomes anatomy only, and the planner will likely recommend not writing it.
- **K2 — CLIP is at least as good.** If CLIP zero-shot global orientation matches or beats the prior's global-only decision on GLOBAL orbit errors (success ≥, harm ≤) at lower cost, prior repair is not a contribution in its own right. The letter is reframed as "any orientation test suffices; the point is what the oracle hides." Odds drop; the planner decides whether to write.
- **K3 — replay fails.** If N0/N1 cannot reproduce the R0.7/R0.8 headline numbers exactly: stop and diagnose before any writing.
- **K4 — time box.** If not submission-ready by 2026-12-11: retarget EUSIPCO 2027, or stop. Cheng decides.

## 6. Ledger

- **R0:** COMPLETE, accepted.
- **R0-N:** AUTHORIZED.
- **R1 / R2:** CANCELLED.
- **T1:** parked.
- **Letter:** planned (see [LETTER_PLAN.md](LETTER_PLAN.md)).
