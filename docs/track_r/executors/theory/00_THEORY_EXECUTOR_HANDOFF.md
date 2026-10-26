# Track R · theory executor handoff (TH1 now, TH2 memo)

**From:** planner (Claude), 2026-10-25
**To:** a separate theory executor (a new chat or agent). **No PAC access is needed or allowed.**

**Binding:**
- `docs/planning/track_r/THEORY_PROGRAM_v1.md` (problem statements; read it first);
- `docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md` (why and when).

## 0. Planner's view

**Results so far** (empirical; FFHQ; 4 solvers; 3,240 runs):
- GT-free selection (signed residual, then a generic uprightness test) recovers 89–97% of the orbit oracle's gain.
- After selection, 93 of 99 remaining failures are generation failures: no run is good in any orbit member.
- SITCOM's failures fit y in 57% of cases. DAPS and FAST-DIPS failures mostly do not fit y.
- CLIP zero-shot fixes 84/84 global 180° errors with no harm observed in 2,040 good outputs. Prior-based repair is therefore **not** a method contribution.

**Your job** is to supply the theory that **explains and predicts** these numbers:
- TH1: the orbit proposition, the selection model, estimators, per-metric inflation, and frozen predictions.
- Then scope TH2: inner functionals in exactly symmetric toys.

**Publication image.**
- CVPR 2027 attempt: paper Nov 16, planner go/no-go Nov 8. Fallbacks are SPL and EUSIPCO 2027.
- TH1 becomes a theory section of about one page, with full proofs in the supplement.
- TH2 could become a separate theory paper if your memo says it is tractable and new.

## 1. Setup

- **Branch.** Create `theory/prdiff-track-r-th-<YYYYMMDD>` from `origin/claude/prdiff-track-r-planner-20261021`. Commit only under `docs/track_r/theory/` and `scripts/track_r/theory/`.
- **Tools.** Python 3, numpy/scipy/matplotlib, and LaTeX for the note. CPU only. Seed everything.
- **Data you may use** (path-free, already committed):
  - `docs/track_r/executors/r0/returns/R0_20261023/R0_8_P4_MATRIX.csv` (policy × K × stratum);
  - `R0_7_ANALYSIS.json`;
  - R0-N N1 CSVs under `docs/track_r/executors/r0n/returns/R0N_20261024T061111Z/` (on branch `codex/prdiff-track-r-r0n-20261023`);
  - the N2 comparison JSON (same branch, `R0N_N2_RECOVERY_20261024T073405Z/`);
  - **A8 per-image label counts**, committed by the PAC executor in R0-N.2. The planner will tell you the path; expected Oct 29.
- **Forbidden:**
  - PAC;
  - ground-truth images;
  - any confirmation305 material;
  - proposing new solvers or methods.

## 2. Tasks and dates

| # | Task | Due |
|---|---|---|
| 1 | Read the program, plus at least: Burns & Fridovich-Keil (arXiv 2605.30330); Stroebl et al. (arXiv 2411.17501); pass@k (Chen et al. 2021, §2.1); Dandapanthula & Boffi (arXiv 2606.02884); Biroli et al. 2024 (*Nat. Commun.*); PC-DAPS (arXiv 2610.31592) §3–4 | Oct 27 |
| 2 | TH1.1: lemma + Proposition 1 + corollaries, written and proved | Oct 29 |
| 3 | TH1.2: closed forms, ceilings, relaxations, population model | Oct 30 |
| 4 | **TH1.5:** fit on K ≤ 4 data only (P4 aggregates now; A8 counts if they arrive in time). **Commit `docs/track_r/theory/TH1_PREDICTIONS_v1.json` and notify the planner.** It must exist **before** the XP-A/XP-B returns | **Oct 31 (hard)** |
| 5 | TH1.3 estimators (with unbiasedness proofs) and TH1.4 per-metric inflation | Nov 3 |
| 6 | TH1 note (LaTeX + PDF) + scripts + paper-ready half-page and one-page text | **Nov 6** |
| 7 | TH2 feasibility memo: first 1D-Fourier-toy numerics, candidate theorems, overlap check, go/no-go | Nov 13 |

**`TH1_PREDICTIONS_v1.json` schema:**
```
{"model": "...", "fit_data": ["P4 K<=4" | "A8 counts K<=4"],
 "fit_commit": "<sha>", "predictions": [
   {"family": "FAST", "dataset": "FFHQ", "ids": "S5_180",
    "policy": "E1_CLIP|O_raw|O_orbit", "metric": "good25",
    "K": 8|12|16, "point": x, "lo95": a, "hi95": b}, ...]}
```

## 3. Standards

- **Rigor.** Every statement is either proven, with its assumptions listed, or labeled CONJECTURE / NUMERICAL.
- **No over-reach.** Never claim a result about real solvers beyond what the numbers show. Give the planner predictions, not conclusions.
- **Citations.** Cite the primary source for every borrowed result: pass@k, the Stroebl ceiling, the DAPS exactness argument, speciation. Mark which statements you verified in the main text.
- **Reproducibility.** Every script reproduces from a committed commit.

## 4. Return (`docs/track_r/theory/TH_RETURN_<date>.md`)

1. **Status:** at most 8 lines.
2. **Identity:** branch and SHAs.
3. **Results:** the list of propositions (statement, assumptions, proof status).
4. **Predictions:** the predictions-file SHA and commit time.
5. **Model fit:** quality, and checks of S1/S2 where data allow.
6. **Open problems:** including a TH2 go/no-go.
7. **Opinion:** optional, and clearly separated.
