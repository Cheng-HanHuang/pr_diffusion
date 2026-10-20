# Track A A1.4 — scientific publication-viability and prospective-benchmark gate (DESK RESEARCH ONLY)
Date: 2026-10-20
Branch ONLY: `codex/prdiff-track-a-a14-publication-gate-20261020`
Base scientific planner decision `37b872fef94090171b1afcb27359355eb76b513b`.
Owner: autonomous Track A scientific planner; you are a dedicated **research/literature/methodology executive**. This assignment is **not** a new PR reconstruction solver or a GPU experiment.

## 0. Mission and first-response contract
**Decide whether Noise Picking (NP) can credibly support a smaller-venue scientific paper** given our existing A0/B22, A1.2 8-image independent 128-run development, A1.3 saved-candidate forensics, and serious 2025–26 competing work. A negative finding is a valid outcome. Do not merely repeat existing optimistic narrative or automatically propose larger computational sweeps. Your job is to produce **adversarial, evidence-based novelty assessment**, a falsifiable hypothesis with fair prospective controls **only if** it survives, and a publication STOP/GO recommendation to the Track A planner.

Start by reading complete:
- root `AGENTS.md`, autonomy amendment `docs/planning/metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md`, shared charter and live [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).
- Planner `docs/planning/track_a/2026-10-20_A13_ACCEPTANCE_AND_A14_RESEARCH_GATE.md`, previous `2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md`, initial `2026-10-15_TRACK_A_PLANNER_RETURN.md`.
- **Verified A1.3 return** on branch `codex/prdiff-track-a-a13-candidate-audit-20261020`, *final doc commit* `f002643fbaff0b0541886e625258f428cac3ea90`, file `docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md`. Distinguish A1.3 *PAC execution source* `b55c5a47f010fad019270b77c4c9366a9c8b99cc` from final documentation.
- **Verified A1.2 full report** on branch `codex/prdiff-track-a-a1-oracle-budget-20261016`, file `docs/track_a/executors/a1-oracle-budget/A1_PLANNER_HANDOFF_RESULTS_PAC_INDEX_20261020.md`, machine index `A1_PAC_ARTIFACT_INDEX_20261020.json`, source of actual GPU candidates `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`.
- A0 retrospectively audited B22 report on branch `codex/prdiff-track-a-a0-native-selector-audit` at `ecc66b890b63123c0445a13d33277e09f6ab4ab5`, file `docs/track_a/executors/a0-native-selector-audit/A0_B22_RETROSPECTIVE_RESULTS_2026-10-16.md`.

**First response**: summarize where the 8-image evidence is positive/negative, create a specific evidence plan for searching and directly reading 2024–2026 papers/code, and **DO NOT** ask the user for PAC access or issue GPU commands. Commit desk audit drafts if needed, then return a fully sourced A1.4 report. You can complete this stage entirely via external official literature + connected GitHub access; the user need not run anything. Do not imply that merely accessing Github gives you any PAC access.

## 1. Core scientific evidence you must preserve (don't overstate)
**A1.2, 8 exposed B22 images, four independent seeds per method, ALL same saved measurement and same PNG8 raw quality metric**:
- K4 *offline GT oracle*: DAPS mean 31.518dB, Good25 8/8, Good30 8/8, charged candidate process wall 260.11s/img.
- NP LF4 30.388dB, 8/8, Good30 4/8, 147.40s/img; NP S2 30.222dB, 8/8, Good30 4/8, 147.17s/img.
- SITCOM K4 27.439dB, 8/8, Good30 0/8, 122.48s/img.
- DAPS4 > NPLF4 in *all eight* by mean 1.13dB at 1.765x time; NP-LF vs SITCOM4 +2.949dB at 1.203x time.
- NP mixed8 oracle 30.392dB, only +0.004dB beyond LF4 but ~294.58s total K8 candidate time. Good25 saturated at K4, preregistered NP-S2 vs SITCOM Good25 improvement zero in exposed pilot. **Not independent validation**.
- DAPS K2 raw score disproportionately punished by conjugate inversion; raw PSNR remains primary, symmetry-aware strictly auxiliary.

**A1.3, SAME exposed 128 candidates, entirely CPU re-evaluated:** common terminal scored signed/clipped `||A(x)-y||`, source identity/FFT DAPS and SITCOM matched exactly; excellent family-specific fixed scorers retrospectively:
  - DAPS K4 signed 28.258 vs clipped31.493 vs oracle31.518;
  - NPLF signed30.367 vs clipped25.880 vs oracle30.388;
  - NPS2 signed30.203 vs clipped25.914 vs oracle30.222;
  - SITCOM signed27.423 vs clipped27.404 vs oracle27.439.
  - Choosing DAPS-clipped & NP-signed after viewing these GT outcomes is **adaptive retrospective policy selection**, NOT validated non-GT generalization. Method-native cross-root selection stats not established uniformly.
  - DAPS 68539 flipped candidate 6.364 raw but31.871 rotated; signed residual near tie with good oriented candidate. 66631/67174 NP clipped selects low quality despite better generated candidates. External pooled min residual across all16 selects SITCOM 8/8 while GT best family is DAPS. Negative measured entries ~36.6–39.4% means clipping changes physics target, 33/112 score portfolios choose different candidates on signed vs clipped. Taxonomy 86 Good25, 3 rotation, 25 measurement-consistent low raw, 14 unresolved, zero obvious poor-fit uses heuristic non-certifying thresholds.
- Distinguish selection *as runtime process with no GT* from choosing the selection rule on GT-exposed samples. Low measurement residual alone does not certify orientation or image quality. Exact magnitude symmetry cannot be resolved by that likelihood score alone.
- A0 B22 final100: 2 DAPS Fresh2 native92/100, sequential SITCOM4S native93/oracle95, heterogeneous NP LF4+S24 native95/oracle96; these are exposed retrospective, **NOT** equal K/time or same metrics as A1.2. The first paper-style risk is false baseline equality.

## 2. Required up-to-date first-party literature audit
Locate **full papers, appendix/supplements, and official code** whenever available, not just abstracts; distinguish verified detail vs title-only existence. Prioritize direct competitors:

1. **DAPS** Zhang et al. CVPR2025: `https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.html`, `https://arxiv.org/html/2407.01521v3`; official GitHub implementation frozen upstream for our studies. Independent best-of-four, posterior approximation, GT oracle convention, phase retrieval operator and costs.
2. **SITCOM-ODE / SITCOM** ICML2025: `https://arxiv.org/html/2410.04479v2` and official `https://github.com/sjames40/SITCOM_ODE`; distinguish SITCOM paper's best-of-four reconstruction quality from one-run runtime, and **exact ODE native source** versus generic algorithm family.
3. **FAST-DIPS** Kim/Shin/Lim **ICLR2026**: `https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html`, `https://arxiv.org/abs/2603.01591`, project `https://fast-dips.github.io/`, official `https://github.com/ququlza/FAST-DIPS`. **CRITICAL NEW POTENTIALLY STRONGER BASELINE**: already explicitly plots phase-retrieval quality against compute budgets. Verify actual phase retrieval operator (oversampled 2× Fourier magnitude?), image source/size, checkpoint, measurement noise (including signed/clipping), GT best-of-K convention, raw PSNR/orientation, runtime charging and independent seeds. If sources incomparable, say so; never announce NP is beaten based solely on abstracts.
4. **Fourier Phase Retrieval With Diffusion Priors Guided by Failure Detection** Xue/Dai/Zheng/Peng/Zou/Xiong, ISCAS2025 DOI `10.1109/ISCAS56072.2025.11042933`, DBLP `https://dblp.org/rec/conf/iscas/XueDZPZX25`. Existence verified; methodology may be paywalled. Search legitimate unpaywalled author version, paper appendix, abstract, demo/code; compare failure detection definition and GT-free selection to NP. Mark *UNKNOWN* where full source is inaccessible instead of pretending an overlap is established.
5. **DiffFPR** Li/Wang ICML2024 `https://proceedings.mlr.press/v235/li24bj.html`: orientations, multichannel coherence, learnable prior.
6. **OOD mismatch** Hu et al. `Patch-Based Diffusion Models Beat Whole-Image Models for Mismatched Distribution Inverse Problems` `https://arxiv.org/abs/2410.11730`, plus distinct prior adaptation and inverse-problem mismatch literature. A pretrained face prior failing on landscapes is already unsurprising. Ask if **NP vs DAPS/SITCOM candidate-reliability under quantified shift at matched compute** is unstudied and causally interpretable, or just another OOD experiment.
7. Search **other 2025–2026 competitive diffusion PR candidate/reliability/cost algorithms** including DDRM-PR (Kaya/Oktem 2025), I2I-PR (2026), failure-detection approaches and any new best-of-K fairness studies; source pages/official repos and publication dates. Check whether their phase retrieval measurement model, diffusion checkpoint and scoring match. Do NOT silently omit an obviously more efficient recent method.

For each, construct `A14_PRIOR_ART_MATRIX.md` columns publication date/venue, exact PR operator/noise, checkpoint/data regime, native K/GT oracle convention, **charged** sampling-vs-reported runtime, lower-tail/mean reliability reporting, mismatch domain, available source, novelty overlap and fair-baseline feasibility; date and cite links. Make **"not verified"** a legitimate value.

## 3. Three independent publication hypotheses / kill tests
Rank them by scientific novelty, likely effect and affordable evidence—not preference for more experiments:

**H1 Candidate-generation / best-of-K risk–compute frontier:** Is NP LF/S2 formulation intrinsically (not merely intermediate by runtime) competitive against **DAPS independent4 and FAST-DIPS** at equal image/measurement/prior and paired cost? Must establish meaningful mean/q05/Bad20/Good25/Good30 benefit or practically justified frontier improvement, including strong controls and uncertainty. The 8-image exposure and Good25 saturation show no validated tail win. Full K cost always counted; raw PSNR primary, rotation diagnostic only. Avoid 'Pareto point' as substantial novelty absent algorithmic or analytic contribution.

**H2 GT-free reconstruction choice despite magnitude ambiguity:** Can one predeclare a principled per-family rule from physics/prior (e.g. valid posterior score, predictive test, data split of Fourier measurement, symmetry-breaking external information) without GT training on the eight exposed cases? A1.3 shows signed vs clipped score pitfalls, and cross-family residual fails. Novelty threat Xue failure detection and classical phase-retrieval ambiguity. Reject impossible claims (noise-free exact likelihood alone recovers orientation) and GT-fitted rules. The earlier handoff's family-specific combination is explicitly NOT a proven policy.

**H3 Prior mismatch / OOD reliability:** Can NP's way of choosing noise proposals or using low-frequency projection generate *more good candidates* when a **fixed pretrained prior** is misspecified, at fixed runs and GPU seconds, with FFHQ ID control, moderate/severe shifts, model checkpoint/operator/noise identical among methods, plus matched-prior or prior-free structural controls? No NP-OOD evidence exists. Attribute effects to prior mismatch rather than simply harder images; define domain distance proxy pre-outcome without GT-screening to avoid cherry-picking. OOD research novelty must exceed Hu et al. and related 2026 methods.

Deliver `A14_CLAIM_KILL_MATRIX.md`: **GO/PIVOT/STOP**, precise falsifier, minimum meaningful magnitude/uncertainty, missing data, competing paper and cost/dependency for H1,H2,H3. Explicitly justify whether whole Track A paper should STOP now.

## 4. Smallest viable prospective study design — only if one hypothesis survives
Write `A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md`, marked **NOT EXECUTION AUTHORIZATION**:
- Primary score: common raw oriented RGB PSNR with PNG8 vs float standardized **prospectively** across all methods including native DAPS; ambiguity-aware PSNR secondary; preregister Good25/Good30, bad20, quantiles, mean plus paired bootstrap and McNemar as appropriate.
- True new cohort: data/measurement **independent of B21/B22 final100, B24 DEV80 and screening**, plus no B24 protected confirmation305 registry/payload/ID/member access. Consult #41 for potential overlap and Track B cohort negotiations. An independently sourced dataset/custodian is preferable to a new subset of the possibly completely screened FFHQ IDs.
- No GT-driven selection of test images or method parameters, including method-specific signed/clipped score switch.
- Exact fixed operator 256 RGB oversampling2, sigma_y=.05 unless scientific reason explicitly predeclared; signed `y_raw` shared, historical frozen NP clipping disclosed. If OOD source differs in resolution/color/dynamic range, freeze preprocessing and report effect.
- Factorial minimum meaningful arms: independent DAPS/SITCOM/NP LF/NP S2 K1/K2/K4, plus stronger new FAST-DIPS where verified/appropriate; expensive K8 arms only if separately motivated with cost and uniqueness. Match **both run counts K and charged wall/GPU budget** across portfolios; source/hardware/time ledger and concurrency controls, stop checks with early source/operator/provenance gate.
- Distinguish **pilot engineering DEV** from pristine eventual confirmation; small 8–12 image pilot may show code fidelity, but cannot be the paper's critical inferential basis, and the user must explicitly run disclosed PAC command. Include failure/cost gates and no threshold tuning on shown images; power/precision rationale for n≥100 or bigger prospective validation **if and only if** scientific kill test justifies it. No false claim new FFHQ final since B24 confirmation cannot be opened.
- True GPU access is **not** authorized by writing this document. Before any future run must register [DATA RESERVATION] and [RUN RESERVATION] with physical GPU IDs, total GPU hours/memory, source SHA, immutable outputs, Track B priority, and user-supplied execution. Without source/dataset consent, STOP.
- Do not add new basin-transition algorithm; Track B scientific ownership.

## 5. Artifacts / return criteria
Create **docs only**, all under `docs/track_a/executors/a14-publication-gate/`: `A14_PRIOR_ART_MATRIX.md`, `A14_CLAIM_KILL_MATRIX.md`, `A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md` (if justified), `A14_EXECUTOR_RETURN.md` or `A14_STOP_REPORT.md`. Citations to real articles and exact source refs; do not invent accepted papers or make paper PDF comparisons without reading them. This handoff cannot authorize code/GPU. Do not reuse A1.2/A1.3 PAC directories for artifacts; **NO PAC output root is needed**. The reserved but **unused** hypothetical future Track A worktree `/egr/research-pac/huang248/pr_diffusion_track_a_a14_publication_gate` must not be created by your stage.
- commit scientific results to **your dedicated branch** `codex/prdiff-track-a-a14-publication-gate-20261020`, with exact commit and evidence URLs, and comment result to issue #41;
- final return states feasible small-venue positioning, confidence level and technical gap; one clear planner choice among **STOP**, **ONE bounded prospective pilot**, or **REFOCUS to a reproducibility/benchmarking study**. Do not equate 'completed literature research' with paper acceptance;
- Explicit no PAC operations/no GPUs/new cohort or protected confirmation access.

## 6. Unambiguous start instruction
Read full handoff and prior reports immediately. Begin source-backed desk research and literature/code audit, commit results without waiting for additional PAC output. Critically identify what could *falsify* the NP paper's case. The planner makes the final research allocation after receiving the A1.4 committed return. Preserve raw PSNR primary and GT best-of-K as legitimate nondeployable benchmark; that endpoint alone does not guarantee publishability.
