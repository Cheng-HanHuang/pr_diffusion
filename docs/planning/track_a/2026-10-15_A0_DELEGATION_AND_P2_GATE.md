# Track A lead return — A0 delegated, prospective benchmark gated
**Date:** 2026-10-15  
**Owner:** autonomous Track A scientific lead  
**Source-read branch HEAD at planning:** `bd83d5093e67f69473c8b7815ad2730b2a5d161a`; no PAC source/artifact access asserted.

## Decision and delegation
**CONTINUE** the NP reliability paper viability investigation, **NOT** a blanket GO for NP superiority or a new GPU campaign. A0 data/source forensics is the immediate highest-value step. The autonomy amendment (same planning branch) now supersedes older no-delegation and docs-only restrictions, while preserving held-out/compute/source firewalls.

A dedicated executor branch was created:

- `codex/prdiff-track-a-a0-native-selector-audit` (initial handoff commit `c61c5029d27220f34ad6b4b93f1b2c7f5a2767ba`).
- [Full 2026-10-15 A0 executor handoff](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-a0-native-selector-audit/docs/track_a/executors/a0-native-selector-audit/2026-10-15_EXECUTOR_HANDOFF.md).
- Registered as a distinct [executor in coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41#issuecomment-6205939527).
- Scope is metadata/source inventory first, then retrospective **zero-GPU** analysis of exposed B22/B24 candidates, **no** new native FFHQ trajectories or FFT recomputation in A0. PAC access always via user-run commands; proposed worktree/output paths are unverified reservations. No premature claim that work executed.

## What existing results already show
B22 compares 2 DAPS Fresh2 (312.6 GPU-s/image), four sequential-RNG SITCOM-4S (196.1 GPU-s/image), and eight heterogeneous LF+S2 NP-8-RS (411.8 GPU-s/image). Good25 are 92/100, 93/100 and 95/100 respectively. NP8 q05=25.146 dB, Fresh2 q05=15.860 dB; DAPS wins paired raw PSNR on 91/100 images; NP–Fresh2 Good25 discordance 7 rescue / 4 harm (McNemar p≈0.549), **not a statistically decisive NP advantage**.

B24 DEV80 has independent DAPS4, SITCOM4 and homogeneous LF NP4 candidate sets, but baseline best-of-four is the GT-PSNR oracle; no established executable native DAPS4 selector exists and a SITCOM native correction-norm history may not have been recorded at all. B24 is deliberately equal-stratum DEV80, not a natural-population unbiased rate estimate; its GT PSNR on saved/8-bit canonical outputs differs from B22 float raw metric representation. Historical B24 selected NP4 Good25 55/80, Fresh2 57/80; observed exceptional NP4 successes in DAPS/SITCOM oracle-shared-failure subset are **diagnostic complementarity** only.

B26 is machine-complete PASS and scientifically negative for the simple signed-score repair on the already exposed DEV80: historical H selected 55/80 Good25 vs signed R 53/80; mean R−H −0.2215 dB with wide bootstrap CI. The held-out confirmation remains locked. Source: [B26 accepted final return](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/planner-b26-resume-handoff-20261015/docs/handoffs/B26_PLANNER_RETURN_2026-10-15.md). This result is retained as negative reliability/mechanism evidence, not a new NP publication arm. The B26 experiment is closed.

## Fixed A0 estimands and no-retrospective-tuning contract

### Question 1: candidate generation, independent of deployment selection
For each **real stored cohort**, calculate `max_{j≤K} PSNR(x_j,GT)` and Good25/Bad20; ground truth is *only* for the offline oracle ceiling. This diagnoses existence of a good candidate. Do not compare B24 stratum-balanced population percentages to B22 natural final100 as independent confirmation.

### Question 2: native executable selection regret
Given a *historically defined and present* native score, calculate selected and oracle PSNR for the **same candidate set**. DAPS Fresh2 K2 `L(rep1)<L(rep0)-0.7`; SITCOM B22 K4 step160 correction-norm min; B22 NP8 postprojection LF winner-trajectory mean min over LF4+S2-4; B24 NP4 homogeneous LF terminal min. Explicitly report selected/PSNR-oracle Good25 difference **and** continuous PSNR regret. Historical SITCOM B24 K4 and DAPS K4 clean-free policies should be **N/A**, not replaced with a favorable new selector.

### Question 3: runtime/operations and fair axes
Compute two **separate** comparisons: **(a)** matched candidate counts K where candidates exist, and **(b)** observed hardware/compute-budget frontiers with all candidates charged. B22 NP8 (411.8s), DAPS2 (312.6s), SITCOM4 (196.1s) are neither matched K nor equal cost. Different models and forward/backward calls prevent equating a UNet call across families. B24 partial dispatch FLOPs exclude custom operator/FFT work; do not pretend exact totals. Report cost intervals and empty cells when no rung is present.

### Question 4: executable same-observation scoring
Source identity verification first. Both official DAPS/SITCOM native main losses use **signed** `y_raw`; historical NP scores and projects with `y_plus`. A shared signed terminal residual is potentially meaningful as a **separately named common external non-GT diagnostic**, not a native selection or prior-validity certificate. Use only already stored fields in A0; any proposal to recompute terminal residual with operator FFT must be a **new separately specified CPU stage**, never hidden inside the metadata-only audit.

### Required artifacts
An inventory of candidate source identity, seed, operator semantics and field presence; reproduced frozen headline rows; candidate-by-candidate dataset with method/seed/cfg/K; native-vs-oracle tables where score exists; missing-cell table; per-image paired CIs and lower-tail summaries; charged GPU seconds/UNet/candidate accounting; clear stop flags; committed A0_EXECUTOR_RETURN and ledger registration. No source or scientific results from B19–B26 may be changed.

## Prospective P2 decision: preliminary NOT YET
**P2 is not ready to launch today.** A new genuinely prospective benchmark is scientifically justifiable only if A0 answers the following without retrospective selector re-optimization:
1. **Source comparability PASS:** effective DAPS/SITCOM dirty PAC source and checkpoint states, operator/noise, observation preprocessing and metric transformations explicitly reconciled; every reported K policy backed by actual candidates.
2. **Scientific signal:** NP offers an identifiable reliability/complementarity mechanism after separating GT oracle candidate-generation ceiling from native selection regret, not merely a comparison to an under-provisioned or GT-oracle baseline. If ordinary matched-K DAPS4 has stronger oracle capability at a larger cost, this is a cost tradeoff requiring *budget matching*, not evidence that either deployable policy wins.
3. **Nontrivial fair control:** protocol can be frozen prospectively with real GT-free selector for **each family** (or transparently N/A for one), reported offline GT oracle separately, and adequate actual cost/GPU memory accounting. Evaluate NP K1/K2/K4/K8 and feasible DAPS/SITCOM K1/K2/K4/K8 only with explicit source/seed costs; no unsupported K8 interpolation as observed data.
4. **Eligible data:** fresh, non-overlapping actual-image cohort or, if only new measurements on exposed images are available, explicitly weaker generalization claim. Track B has preference when contested, issue #41 data reservation mandatory; B24 confirmation305 is never an option.
5. **Falsifier and effect:** predeclare prospective primary Good25 + catastrophic Bad20 + q05 (with distribution-aware uncertainty) against best available native **matched budget**. A substantive effect filter is ≥5-point Good25 increase with paired 95% CI lower >0 (or robust strict better Pareto at equal/lower cost), plus no material mean/median degradation; thresholds must be fixed before seeing new data. Otherwise stop the positive methods paper, optionally report a negative methods/reproducibility study.

**Research lead can independently authorize a well-bounded P2 development pilot if A0 passes scientific/source gates under the autonomy amendment, but large prospective final tests or shared cohorts require metaplanner coordination.** No stage has been initiated by this document.

## Contingent next handoff after A0
If the A0 executor finds missing native K4 SITCOM data or no comparable DAPS4 external clean-free score, the next limited work should be a **protocol design** for prospective native GT-free selector collection—not a large GPU run to manufacture favorable results. If instead A0 reveals no meaningful signal after charging work, recommend **STOP** NP method paper before new FFHQ computation. Track B owns basin-aware algorithm innovation.
