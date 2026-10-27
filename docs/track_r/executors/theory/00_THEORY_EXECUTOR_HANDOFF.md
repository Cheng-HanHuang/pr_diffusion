# Track R · theory executor handoff (TH1 now, TH2 memo), revision B.1

> **2026-10-27: read [`01_PLANNER_GUIDANCE_20261027.md`](01_PLANNER_GUIDANCE_20261027.md) first.** It covers where A8 is, the S2 rejection, the v1 contents and target (Oct 29, 18:00 ET), the TH2 refocus on the noise floor, and the reading permissions.
> **2026-10-27, 12:30: then read [`02_PLANNER_GUIDANCE_20261027_PM.md`](02_PLANNER_GUIDANCE_20261027_PM.md).** It covers the mech v1 registration, the planner's run of your code on the real A8, what the v1 freeze still needs (panels, interval type, sensitivity), v2, and `theory_inputs/`.
> **2026-10-27, 17:15: then read [`03_PLANNER_GUIDANCE_20261027_EVE.md`](03_PLANNER_GUIDANCE_20261027_EVE.md).** It covers the A8b and D1-P1 inputs and their SHAs, the mech v1.1 registration, the planner's production run of the A6 code (the validator fails without the control-variate fix), and the v1 and v2 deliverables.

**From:** planner (Claude), 2026-10-26.
- Revision B.1 adds the TGD read and the phase-retrieval-theory research pass (program v1.2; digest §§4–5).
- Revision A (2026-10-25) was never started.

**To:** a separate theory executor (a new chat or agent). **No PAC access is needed or allowed.** The kickoff prompt is `docs/track_r/executors/theory/KICKOFF_PROMPT.md`.

**Cheng's decision (2026-10-26): start now.**

**Binding:**
- `docs/planning/track_r/THEORY_PROGRAM_v1.md`, now **v1.2** (problem statements; read it first);
- `docs/planning/track_r/2026-10-26_THEORY_LIT_DIGEST.md`:
  - main-text digests of Stroebl et al., Dandapanthula & Boffi, TDS and TGD (§§1–4), with planner derivations D1–D15 to prove or refute;
  - the condensed research pass on phase-retrieval theory (§5), whose items marked [R] are not yet verified;
- `docs/planning/track_r/XP_PREREG_v1_1.md` §5 (freeze order and embargo);
- `docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md` (why and when).

## 0. Planner's view

**Results so far** (empirical; FFHQ; 4 solvers; 3,240 runs):
- GT-free selection (signed residual, then a generic uprightness test) recovers 89–97% of the orbit oracle's gain.
- After selection, 93 of 99 remaining failures are generation failures: no run is good in any orbit member.
- SITCOM's failures fit y in 57% of cases. DAPS and FAST-DIPS failures mostly do not fit y.
- CLIP zero-shot fixes 84/84 global 180° errors with no harm observed in 2,040 good outputs. Prior-based repair is therefore **not** a method contribution.

**What the new reading changes:**
- **Stroebl et al. already have our core model.** Their App. C imperfect-verifier model equals our E0 formula once argmin is shown to equal accept-first in law (digest D1). **Credit them.** Our theory is new in four ways:
  - the exact symmetry: Prop. 1 and the equivariance Corollary 4;
  - the orbit classes and the symmetry-aware selector;
  - exact estimators;
  - predictions **frozen before the data and then tested**.
- **The equivariance corollary is the strongest single result available.** Measurement guidance is provably orientation-blind, so orientation preference comes only from the network. XP-C2 measures the network's equivariance defect directly.
- **Cor. 4(b) makes it quantitative with weak assumptions.** It is a total-variation bound on orientation preference in terms of the equivariance defect, via the KL chain rule and Pinsker. It needs no exact score, no log-concavity and no uniqueness: exactly the kind of result Cheng asked for.
- **Cor. 1′ extends G-blindness to particle methods.** Likelihood-weighted SMC and residual pruning (TDS, TGD, A-TGD) cannot fix orientation.
- **The monotone-gap theorem** (digest D3) is a falsifiable shape prediction. It costs nothing to state.
- **Do not overreach on uniqueness.** Our 384 canvas lies outside the classical (Hayes) uniqueness hypothesis. Assume orbit identifiability explicitly.

**Publication image.**
- CVPR 2027 attempt: paper Nov 16, planner go/no-go Nov 8. Fallbacks are SPL and EUSIPCO 2027.
- TH1 becomes a theory section of about one page, with full proofs in the supplement.
- TH2 could become a separate theory paper if your memo says it is tractable and new.

## 1. Setup

- **Branch.** Create `theory/prdiff-track-r-th-<YYYYMMDD>` from `origin/claude/prdiff-track-r-planner-20261021`. Commit only under `docs/track_r/theory/` and `scripts/track_r/theory/`. **Do not edit** `PREDICTIONS_REGISTRY.md`: the planner maintains it.
- **Tools.** Python 3, numpy/scipy/matplotlib, and LaTeX for the note. CPU only. Seed everything.
- **Data you may use** (path-free, committed):
  - `docs/track_r/executors/r0/returns/R0_20261023/R0_8_P4_MATRIX.csv` (policy × K × stratum) and `R0_7_ANALYSIS.json`;
  - R0-N N1 CSVs under `docs/track_r/executors/r0n/returns/R0N_20261024T061111Z/` (on branch `codex/prdiff-track-r-r0n-20261023`);
  - the N2 comparison JSON (same branch, `R0N_N2_RECOVERY_20261024T073405Z/`);
  - **A8 per-image strings**, committed by the PAC executor in R0-N.2 (amendment 2 revision D.1), expected Oct 28–29.
    - **Path:** `docs/track_r/executors/r0n/returns/R0N2_<stamp>/R0N2_PER_IMAGE_COUNTS.csv` on `codex/prdiff-track-r-r0n-20261023` or `codex/prdiff-track-r-xp-*`. Fetch daily.
    - **If A8 is not committed by Oct 29 noon ET,** fit on P4 at K ≤ 4 and say so.
    - **Families:** DAPS, SITCOM, FAST-DIPS only. NP is out of the paper (XP prereg A4).
    - Columns:
    - counts per class;
    - `run_labels`;
    - `e0_rank_order`;
    - `orbit_best_member`;
    - the stored decisions `dec_clip_face`, `dec_s_all8`, `dec_2f`.
- **Embargo.** Read **no XP return** until your v1 predictions are registered. After that, read only what the planner forwards (XP-D1 phase-1 counts for v2).
- **Forbidden:**
  - PAC;
  - ground-truth images;
  - any confirmation305 material;
  - proposing new solvers or methods.

## 2. Tasks and dates

| # | Task | Due |
|---|---|---|
| 1 | Read the digest and the program. Then, as available to you:<br>• Chen et al. 2021 §2.1 (pass@k)<br>• Burns & Fridovich-Keil (arXiv 2605.30330)<br>• Biroli et al. 2024 (*Nat. Commun.*)<br>• PC-DAPS (arXiv 2610.31592) §3–4<br>• Hoogeboom et al. 2022 (equivariant diffusion)<br>• Barnett, Epstein, Greengard, Magland 2020 (arXiv 1808.10747)<br>• DAPS (arXiv 2407.01521) App. F.2 / Fig. 13 (v1): the "eight images" claim, now verified by the planner (digest §5)<br>The PDFs of Stroebl, Dandapanthula & Boffi, TDS and TGD may not be available to you; the digest is enough | Oct 27 |
| 2 | TH1.1: lemma (with the identifiability caveat), Proposition 1, Corollaries 1, 1′, 2–5 and **4(b)**, written and proved. Check Corollary 4's conditions family by family | Oct 29 |
| 3 | TH1.2: closed forms; the Stroebl correspondence; ceilings; the **monotone-gap theorem**; the bottleneck shift; relaxations; the population model | Oct 30 |
| 4 | **TH1.5 v1:** fit on K ≤ 4 only (A8 strings; P4 at K ≤ 4 if A8 is late). **Commit `docs/track_r/theory/TH1_PREDICTIONS_v1.json` and tell the planner** (via Cheng) with the commit SHA | **Oct 31 (hard)** |
| 5 | TH1.3 estimators (with the order-statistic identity and unbiasedness proofs) and TH1.4 per-metric inflation | Nov 3 |
| 6 | **TH1.5 v2** (optional): ImageNet FAST from phase-1 counts, within 48 h of receiving them | ≈ Nov 1–3 |
| 7 | TH1 note (LaTeX + PDF) + scripts + paper-ready half-page and one-page text | **Nov 6** |
| 8 | TH2 feasibility memo:<br>• first 1D-Fourier-toy numerics, with exact zero-flip decoys<br>• candidate theorems (priority: TH2(e), S2 as an expectation statement)<br>• overlap check<br>• go/no-go | Nov 13 |

**`TH1_PREDICTIONS_v1.json` schema:**
```
{"version": "v1", "model": "<short description, e.g. 3-type mixture of (p, o_g, o_p, c, u)>",
 "fit_data": ["A8 strings K<=4" | "P4 K<=4"], "fit_commit": "<sha of the data commit>",
 "code_commit": "<sha of your fitting script>",
 "uncertainty": "<how model and sampling uncertainty are combined>",
 "predictions": [
   {"family": "FAST|SITCOM|DAPS", "dataset": "FFHQ",
    "cohort": "S5_180|B22|DEV80|POOLED",
    "policy": "E0|E1_CLIP|O_raw|O_orbit", "metric": "good25",
    "K": <int>, "point": <x>, "lo95": <a>, "hi95": <b>}, ...],
 "shape_predictions": [
   {"family": "...", "cohort": "...", "statement": "O_orbit - E1_CLIP non-decreasing in K"},
   {"family": "...", "cohort": "...", "statement": "O_raw - E0 non-decreasing in K"}],
 "crossover_K": [{"family": "...", "cohort": "...", "K_star": <x>, "lo95": <a>, "hi95": <b>}]}
```
**K grids:**
- FAST: {8, 12, 16, 24, 32, 48, 64};
- SITCOM: {8, 12, 16, 24, 32};
- DAPS: {4 (B22 only), 6, 8}.

`crossover_K` is the K at which selection misses overtake generation failures (TH1.2 item 4). Give `null` when it never happens within K ≤ 64.

## 3. Standards

- **Rigor.** Every statement is either proven, with its assumptions listed, or labeled CONJECTURE / NUMERICAL.
- **No over-reach.** Never claim a result about real solvers beyond what the numbers show. Give the planner predictions, not conclusions.
- **Citations.** Cite the primary source for every borrowed result:
  - pass@k;
  - the Stroebl model and ceiling (state the correspondence plainly);
  - the D&B mode-selection result;
  - TDS and TGD;
  - the DAPS exactness argument;
  - invariance propagation;
  - speciation;
  - trivial associates and twin stagnation;
  - the CDI precedents for residual selection (GHIO; Chapman et al.).

  Mark which statements you verified in the main text, which came from the digest's [V] items, and which from its [R] items.
- **Reproducibility.** Every script reproduces from a committed commit.

## 4. Return (`docs/track_r/theory/TH_RETURN_<date>.md`)

1. **Status:** at most 8 lines.
2. **Identity:** branch and SHAs.
3. **Results:** the list of propositions (statement, assumptions, proof status).
4. **Predictions:** the predictions-file SHA-256, commit SHA and commit time.
5. **Model fit:** quality, and checks of S1/S2 where the data allow.
6. **Open problems:** including a TH2 go/no-go.
7. **Opinion:** optional, and clearly separated.
