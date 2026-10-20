# Track A scientific planner — A1.3 ACCEPTED; next publication/novelty gate (A1.4)
Date: 2026-10-20 (America/Detroit)
Scientific lead branch: `codex/prdiff-track-a-np-paper-planner-20261015`
Decision: **ACCEPT A1.3 / CLOSE CPU FORENSICS; CONTINUE ONLY A1.4 DESK SCIENCE UNTIL NOVELTY / FAIR BENCHMARK GATES PASS; NO GPU, NO PAC TASK, NO NEW COHORT YET**.

## 1. Provenance of completed stage
- **Executor:** `codex/prdiff-track-a-a13-candidate-audit-20261020`. Original PAC A1.3 **execution source** `b55c5a47f010fad019270b77c4c9366a9c8b99cc`. Later **final documentation/report commit** `f002643fbaff0b0541886e625258f428cac3ea90`; preserve this distinction.
- [Authoritative A1.3 return](https://github.com/Cheng-HanHuang/pr_diffusion/blob/f002643fbaff0b0541886e625258f428cac3ea90/docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md). User PAC stage0 source/inputs PASS, 128 saved A1.2 candidates verified; stage1 7/7 synthetic tests; 128 candidate rows, 112 exact oracle replays, 224 selector rows, 96 extra float-vs-PNG diagnostics, CPU wall 6.3815s. Native DAPS/SITCOM reference Fourier forward operators both numerical relative error 0.0; no GPU, new measurement/candidate or protected B24 confirmation305 access.
- A1.3 compact PAC metadata ZIP SHA256 `6b889c5fdca0db20b58d1ffa4c44b339021af1ae4a387da44cdd5d8ccd7bb7bb`, recorded under `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/A13_CPU_METADATA_RETURN_V1.zip`; report remains the reviewed scientific source. No claim of current direct PAC file access.
- The A1.2 **GPU candidate-generation source** is different: `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`, 8 *already-exposed B22* images, 128 native trajectories. Both A1.2 and A1.3 outputs frozen/read-only; A1.2 GPU3 lease released.

## 2. What the A1.3 result actually proves
Predeclared **common external** terminal FFT residuals evaluated on the same saved RGB8 outputs, signed raw noisy measurement versus internally clipped `y_plus`. Mean raw PSNR for K4 score-selected candidates (GT-oracle in last column):

| Portfolio | Signed-min | Clipped-min | GT best-of-K | Charged candidate subprocess time / image |
|---|---:|---:|---:|---:|
| DAPS K4 | 28.258 | 31.493 | 31.518 | 260.11 s |
| NP-LF K4 | 30.367 | 25.880 | 30.388 | 147.40 s |
| NP-S2 K4 | 30.203 | 25.914 | 30.222 | 147.17 s |
| SITCOM K4 | 27.423 | 27.404 | 27.439 | 122.48 s |
| NP mixed8 | 30.253 | 23.451 | 30.392 | 294.58 s |

Do **not** turn DAPS-clipped/NP-signed into a validated method-specific rule. Such a *choice between two examined scores* made using these eight GT outcomes is retrospectively optimized, regardless of both individual scorers' GT-free runtime implementation. Missing native cross-run K4 scores are N/A; these are NOT the historical DAPS/SITCOM/NP policies.

Critical mechanistic findings:
1. DAPS image68539 signed-min picked approximately 6.36-dB raw output whose **180-degree rotated** result is ~31.87 dB. Signed residual of the rotated vs raw-aligned candidate differs by ~9e-6 (normalized); exact conjugate-reflection ambiguity makes orientation unidentifiable from magnitude alone. Raw PSNR remains PRIMARY practical metric, rotation-aware only explanatory.
2. NP clipped-min missed available 29–31-dB outputs on images66631 and67174; these poor selections largely **cannot** be rescued by 180-degree rotation. Negative entries account for ~36.6–39.4% of noisy measured magnitude entries; external signed/clipped scores picked different IDs in 33/112 fixed portfolios. Clipped noisy target changes comparison geometry.
3. Retrospective common residual among **all 16 candidates** always chose SITCOM 8/8, but pooled GT PSNR oracle always found a DAPS candidate; cross-family residual is NOT calibrated quality confidence. This pooled reranking was diagnostic, not preregistered.
4. Terminal taxonomy `86 raw Good25 / 3 rotation-recovered / 25 low-raw measurement-consistent / 14 unresolved / 0 unambiguous poor-fit` is **heuristic** with 3× noise RMSE cutoff, not proof of distinct posterior basins or convergence.
5. Strong apparent selector performance is from **8 previously exposed images** with GT observation in reporting; no statistical/population or prospectively validated selector claim. Descriptively interesting but **not** independent evidence.

## 3. Publication focus and immediate hard truths
Retain the **paper-convention ground-truth best-of-K candidate-set PSNR versus ALL K-run charged compute** as PRIMARY claim family; raw orientation PSNR; full per-method candidate work charged; executable GT-free selection is separate SECONDARY. The strong A1.2 8-image independent-control result itself is not confirmation: DAPS K4 oracle 31.518 dB versus NP-LF K4 30.388 (DAPS wins all 8, costs 260.11 vs 147.40 s/image), SITCOM K4 27.439 at122.48s/image. All 8 Good25 for each K4 family; original NP-S2 vs SITCOM Good25 primary +5 percentage-point gate was not met on these *development* images. NP mixed8 adds 0.004 dB vs LF4 for approximately twice candidate cost.

This is a plausible **quality/efficiency positioning**, but without a distinct method-level contribution and new independent evidence it is weak as a separate publication. Generic prior-mismatch robustness or generic GT-free failure detection are NOT firsts.

**External literature threats checked as of 2026-10-20**:
- FAST-DIPS (Kim, Shin, Lim, **ICLR 2026**, `https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html`, `https://github.com/ququlza/FAST-DIPS`) compares compute-vs-quality for nonlinear inverse problems **including phase retrieval**, claiming competitive quality at substantially reduced runtime. Need exact operator, checkpoint, noise, and paper's best-of-K/one-run timing audit before deciding if it is an *applicable* additional baseline. Never assume compatibility without verifying implementation.
- Hu, Song, Fessler, Shen, `Patch-Based Diffusion Models Beat Whole-Image Models for Mismatched Distribution Inverse Problems`, `https://arxiv.org/abs/2410.11730`, already investigates mismatched diffusion priors and inverse reconstructions; OOD alone is not NP novelty.
- Xue et al. `Fourier Phase Retrieval With Diffusion Priors Guided by Failure Detection`, ISCAS 2025, DOI `10.1109/ISCAS56072.2025.11042933`, is directly relevant to generic failure detection and PR. Full source/details NOT YET inspected; existence and title alone cannot settle method overlap.
- DiffFPR (Li, Wang, ICML 2024, `https://proceedings.mlr.press/v235/li24bj.html`) foregrounds equivalent-solution/orientation issues in color Fourier PR; symmetry failure alone is not novel.

**Hard conclusion:** more FFHQ GPU runs would not resolve the principal *novelty* concern. Do the desk research first. Do not prematurely claim NP outperforms contemporary FAST-DIPS or any newer baselines; also do not demand an impossible GT-free orientation selector from an invariant forward measurement.

## 4. Next actionable stage: A1.4 desk-only "publication kill test"
Autonomous Track A can delegate a separate **A1.4 literature, source and experimental-design executor**, zero GPU/PAC, no external dataset access, no new FFHQ or confirmation305. Its goal is a **publishable-or-kill** decision and a genuinely independent **small** prospective protocol *only if warranted*. Proposed scope:
- Detailed research/prior-art matrix of Noise Picking itself / DiffFPR / DAPS / SITCOM / FAST-DIPS / failure-detection PR / OOD prior mismatch and other current competitive methods; inspect original papers, supplement and code where available. Identify exact algorithmic novelty, original contribution and benchmark reporting conventions, particularly independent-vs-sequential restarts, GT best-of-K, true total K cost, raw PSNR vs symmetry metric, signed/clipped noisy measurements, model identity and operator.
- Recalculate from **published verified A0/A1/A1.3 tables only** the feasible oracle quality/compute Pareto points. No looking at protected data, recalculation of model candidates or GT-screening/reweighting eight exposed images as a future test.
- Compare distinct **claims**: (I) an improved candidate generator for oversampled Fourier PR at fixed K/cost with better tail; (II) a calibrated non-GT failure/selector method with independent evidence; (III) robustness to *quantified pretrained prior mismatch* at cost-matched K, with proper controls. Assign each a critical competing paper, falsifier, potential venue and minimum novel deliverable. It's acceptable to **recommend STOP** for all three.
- If any survives, propose a **gated smallest prospective pilot** with scientifically justified in-domain, moderate and severe shifts, common pretrained checkpoint/operator/noise/measurement semantics, an independent data provenance/custodian, source licensing, fixed seeds and freeze before GT outcomes, ablation of prior vs shift vs image difficulty and matched GPU-s budgets. Favor zero GPU now, pilot only after a separate explicit user PAC launch and coordination #41 [DATA]/[RUN] reservations. No original B24 confirmation305 query or membership check ever.
- A1.4 report must candidly address FAST-DIPS compatibility and 2025 ISCAS overlap; no fabricated novel contribution, no hidden claims of source equivalence.

**Go/no-go before launching an expensive new campaign:** Need a credible differentiator that survives strong 2026 literature baselines; an in-principle observable, falsifiable primary metric (raw PSNR and lower-tail, GT best-of-K+K-run time as primary); clear independent non-contaminated data provenance and exact physical measurement normalization, robust budget accounting and a bounded GPU pilot with early stop. If not, freeze Track A NP method-paper project or recast as openly disclosed benchmark/reproducibility report; do not expand exploratory K4 sweep.

## 5. Guardrails and coordination
A1.3 completed/closed, no more PAC commands. A1.4 is **desk only** until independently justified and separately consented; don't allocate GPUs or launch OOD. Track B retains new basin-transition algorithms and scarce-GPU priority. Write only `docs/track_a/executors/a14-publication-gate/`, `docs/planning/track_a/`, optional `scripts/track_a/a14_*` if necessary but no experiments. Worktree `/egr/research-pac/huang248/pr_diffusion_track_a_a14_publication_gate` is merely reserved and **not** to be created or used in this stage. No new outputs tree needed.

When stage returns, planner decides STOP vs smallest bounded prospective study; keep A0/A1/A1.3 negative/positive evidence together and label all exposed panels honestly. Register decisions in canonical coordination [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41), no automatic merge to main.
