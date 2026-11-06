# Triage of two AI reviews (planner, 2026-11-06)

**Source.** Two reviews by the Stanford ML Group's agentic reviewer (paperreview.ai) of the Nov 5 draft, run by Cheng:
- **R1:** ICLR calibration, estimated score 7.1/10, "I recommend acceptance" ([`AI_REVIEW_1_ICLR_SCALE_20261105.txt`](AI_REVIEW_1_ICLR_SCALE_20261105.txt)).
- **R2:** CVPR calibration, no score, positive overall ([`AI_REVIEW_2_CVPR_20261105.txt`](AI_REVIEW_2_CVPR_20261105.txt)).

**Cheng's instruction (Nov 5, 23:37).** Resolve the weaknesses and questions in writing. Sort the method and experiment items into:
- **(1)** out of this paper's control;
- **(2)** addressable with more experiments under a reasonable budget.

Record both. Focus on (2) for the supplement if there is time; the executor is still on last round's jobs. Use natural or intuitive reasoning wherever it can answer an item without new theory or experiments.

**Categories used below.**
- **W:** writing only.
- **1:** out of this paper's control or scope.
- **2:** addressable with more experiments.

"Done" means the paper text now answers the item. "Done + data" means a new planner analysis on committed data backs the answer (`planner_analysis/A26_reviewer_checks/`, ledger C-V1 to C-V3; CPU only, no new runs).

## 1. Items and responses

| # | Item (source) | Cat. | Response (reasoning) | Where | Status |
|---|---|---|---|---|---|
| 1 | GOOD uses PSNR thresholds 25/20 dB; arbitrary? Sensitivity curves? (R1 W, R2 Q1) | 2 | Mean PSNR/SSIM/LPIPS results do not use θ. θ sits below typical correct runs (SITCOM GOOD median 27.3 dB) and above most failures. Rescoring at θ − 3 to θ + 2/3 dB keeps every comparison (C-V1). | Sec. 3 (one sentence); supp §Sensitivity, Tabs. 13–14 | **Done + data** for SITCOM (16 runs) and PC-DAPS (FFHQ, ImageNet). **Open (2):** the other panels store labels only; per-run PSNR export queued (X1). |
| 2 | Independence/exchangeability of runs; correlations, near-duplicate seeds (R1 W, Q5; R2 W, Q6) | W + 2 | Fresh seeds, shared measurement only, no warm starts or adaptation, so runs are exchangeable. By de Finetti, exchangeable runs are independent given the image's rates, which is exactly the model. Near-duplicates of easy images reflect a concentrated posterior, not seed dependence. Tested: no drift (C-F4); no adjacent-run dependence, p 0.17–0.69 on five panels (C-V3). PC-DAPS particles interact, so stopping is never applied within a sample. | Sec. 3 data paragraph; Sec. 5 (mixture is the only dependence); supp §Dependence | **Done + data** |
| 3 | τ_S needs per-solver, per-noise calibration; untested on other noise models/operators (R1 W, R2 W) | 1 + W | Label-free rule needs no labels; within 2% at σ = 0.05 and within 3.5% at σ = 0.02/0.10 (C-G3, C-E4). Other noise models need their own likelihood (supp §Other verifiers). | Sec. 4 calibration; supp | **Done.** Other noise models: (1), recorded. |
| 4 | Label-free τ robust to heavy tails, heterogeneity, overfitting solvers? Pilot size 2 vs 4? Online τ? (R1 Q3, R2 Q5) | 2 | Median/MAD resist heavy tails and a minority of hard images. Overfitting solvers (DAPS-2K, SITCOM) are in the calibration set. Pilots of 2/3/4 runs land within 2.4% at σ = 0.05; the worst case is +7.6% (cohort B, σ = 0.02, 2 runs) (C-V2). Medians can be updated online. | Sec. 4 (one clause); supp Tab. 15 | **Done + data** |
| 5 | Orientation via CLIP prompts/heuristics is imperfect on ImageNet; learned or paired classifier? (R1 W, Q2; R2 W, Q3) | 1 + W | The Bayes choice orders orbit members by the prior. CLIP, the denoising-loss score and a learned classifier are proxies; any one plugs into step (3). A learned classifier needs target-domain training and breaks zero-shot. The gap decomposition bounds the gain: orientation part 4.0 (DAPS-4K, K = 8), 5.0 (FAST-DIPS, K = 64) on ImageNet, ≤ 0.6 on FFHQ. | Sec. 6 (one sentence); supp §Orientation scorers | **Done.** Training a classifier: (1)/(2) optional, not planned. |
| 6 | Only one operator, σ = 0.05 mostly; coded diffraction, oversampling, real data untested (R1 W; R2 Q7) | 1 + W | The residual comparison and Prop. 1 need only Gaussian noise on a fixed y and exchangeable runs. The operator sets the failure mix. CDP: no twin, step (3) unnecessary. Lower oversampling: more consistent failures, less stopping benefit. Inpainting/SR: null space, so failures are consistent and stopping cannot help. Real data: protocol needs no GT, scoring does. | Sec. 7 "Limitations and scope"; supp §Other operators | **Done** (stated as expectations). New operators: (1), recorded. |
| 7 | Cohort B partly from the denoiser's training split (R1 W) | W | Cohorts reported separately; conclusions hold on cohort A alone (Tab. 3). All ImageNet images are validation images, unseen by the ImageNet denoiser. | Sec. 3 | **Done** |
| 8 | Dense; consolidate the per-metric oracle, orbit oracle, E0/E1 (R1 clarity) | W | The policies paragraph lists them with test-time availability. A policy table was considered but does not fit 8 pages after the other changes. | — | **Partial.** Table optional if space is freed. |
| 9 | PDF-extraction artifacts in tables (R1) | — | Artifact of the reviewer's text extraction; the PDF tables render correctly. | — | No action |
| 10 | Flow-based PR (FlowSGS, FlowFPR, DDRM-PR), DPnP not discussed (R1, R2) | W + 1 | Verified main texts. FlowSGS (arXiv 2609.20769) reports the best of 8 posterior samples for FPR. DDRM-PR (Appl. Opt. 2025) and I2I-PR (DSP 2026) keep HIO starts with the smallest residual and resolve the twin ambiguity against the ground truth. DPnP (NeurIPS 2024) uses a coded mask. **"FlowFPR" could not be found** and is not cited. Running these solvers is (1) for this submission; the protocol is solver-agnostic (supp §Other solvers). | Sec. 2 (three sentences) | **Done.** Experiments: (1), recorded. |
| 11 | Sequential allocation, R&S, interacting particle guidance (R1) | W | Adaptive test-time compute for LMs (Snell et al., Damani et al., ICLR 2025) added. Our rule allocates by observed consistency with no learned difficulty predictor. | Sec. 2 | **Done** |
| 12 | Relate race vs exchangeable to tail competition and verifier limits; implications for verifiers (R2) | W | The race is the tail competition of best-of-n: a consistent run wins only if it beats every good run, not merely if it passes. | Sec. 5 (one sentence) | **Done** |
| 13 | Alternative verifiers: robust, per-frequency, multiscale residuals (R2 W, Q2); features to detect consistent-but-wrong runs (R1 Q6) | W + 2 | Under Gaussian noise, the likelihood depends on y only through ρ. Other residuals are other noise models. Consistent runs fit within noise, so separating them needs the prior. | Sec. 6 (one sentence); supp §Other verifiers | **Done (reasoning).** Optional (2): band-wise residuals from stored reconstructions (X3), low priority. |
| 14 | Wall-clock / NFE-normalized comparisons; overhead of orientation scoring (R1 Q4, R2 W, Q4) | W + 2 | Within a solver, runs have equal denoiser calls, so equal mean runs is equal compute. Overhead: one FFT per run, 8.5 ms CLIP per output, against 30.6–66.0 s per run. Cross-solver equal compute only for PC-DAPS vs DAPS-4K (calls and seconds given). | Sec. 4 (one sentence); supp §Compute | **Done.** Optional (2): a full per-solver compute table from receipts (X2). |
| 15 | Sensitivity to M; confidence-conditioned stopping (R1 Q1) | W | Fig. 1b shows M = 1–4: each M is a point on a frontier above fixed best-of-K. The confidence-conditioned variant was tested and does not help (S4, `tab:s4-frontier`). | Sec. 4 (existing); supp §The target M | **Done.** Optional (2): M = 1–4 on held-out panels (P2). |
| 16 | Minimum per-image budget to characterize unsolved coverage; uncertainty with small K (R1 Q7) | W | Report the share with no good run at a large K, as an upper bound: FAST-DIPS FFHQ 25 images at K = 4 against 7 at K = 64. Sec. 5 shows why small K cannot identify the mass near zero (C-T8, C-B1 enclosure). | Sec. 7 recommendation (vi) | **Done.** Optional (2): unsolved-share-vs-K curves with bootstrap (P3). |
| 17 | Sketch proofs or intuitions in the main text (R2) | W | Prop. 1 states its proof idea (first two passing runs are iid draws); the estimators are given in closed form; the race now has its tail-competition intuition. | Secs. 4–5 | **Done** |
| 18 | A compact ablation figure of where each component helps (R2) | W + 2 | Tab. 3 and the gap decomposition give the ladder (first → E0 → E1 → O_orbit). No page space for a figure. | — | **Optional (2):** a supplement ladder figure (P4). |

## 2. Queue for the supplement (category 2), in priority order

All of this is CPU work on stored data, with no new runs. It is exploratory and post hoc (prereg A25).

| ID | Task | Owner | Needs |
|---|---|---|---|
| X1 | Export per-run PSNR (raw, all eight orbit members), SSIM and VGG-LPIPS for every panel without it: FAST-DIPS FFHQ 64, DAPS-2K FFHQ 8, DAPS-4K FFHQ/ImageNet 8, FAST-DIPS ImageNet 64, ImageNet′ 64, σ panels | Executor (PAC CPU) | After CP4, or in parallel if CPU is idle |
| P1 | θ sweep on every panel, including the protocol (exact subset form), from X1 | Planner | X1 |
| P2 | M = 1–4 on the held-out panels with the frozen protocol code (`c598176`) | Planner | Committed strings |
| P3 | Share of images with no good run vs K, with image-bootstrap bands | Planner | Committed strings |
| X2 | Per-solver compute table (denoiser calls, seconds per run, from receipts) | Executor | Receipts |
| X3 | Band-wise and ℓ1 residuals from stored reconstructions; does any reorder GOOD vs CONSISTENT (baseline 200/205)? | Executor (PAC CPU) | Low priority |
| P4 | Component-ladder figure from existing aggregates | Planner | — |

## Planner's view

- **Current results.** Both reviews are positive: R1 estimates 7.1 on the ICLR scale and recommends acceptance; R2 is positive without a score. No item challenges a main claim.
  - The two shared concerns were the PSNR threshold and run independence. Both are now answered with data on committed runs.
  - The orientation scorer and other operators are scope items. They are answered by reasoning and bounded by our own gap decomposition.
  - The main text still ends on page 8, with about 2 lines of slack.
- **Plan.**
  - The executor exports per-run metrics (X1) after CP4. The planner then completes the θ sweep on every panel (P1) and, if time allows, the M sweep (P2) and the unsolved-vs-K curves (P3).
  - The supplement freezes Nov 22.
- **Image of the publication.** The reviews read the paper as intended: a rigorous, preregistered audit with a deployable, ground-truth-free protocol. The likely reviewer asks were scope (other operators, flow solvers, learned orientation) and robustness (threshold, independence). The paper now answers each one briefly, with checks in the supplement.
