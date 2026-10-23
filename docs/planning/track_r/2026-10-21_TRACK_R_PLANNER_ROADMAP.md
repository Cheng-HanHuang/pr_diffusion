# Track R planner roadmap — reliability per compute for diffusion-prior Fourier phase retrieval

> **Superseded status (2026-10-23).** R0 is complete. D-material and D-detect failed; D-repair passed. The method line, R1 and R2 are cancelled. The current plan is one short evaluation letter, then wind-down: see [`2026-10-23_R0_PLANNER_DECISION.md`](2026-10-23_R0_PLANNER_DECISION.md). This roadmap is kept unchanged below as the pre-registered record.

**Date:** 2026-10-21 (America/New_York)
**Planner:** Claude (planner role assigned by Cheng on 2026-10-21)
**Executor:** a GPT/Codex session chosen by Cheng (no PAC access; writes code, tests, PAC command blocks)
**PAC operator:** Cheng (runs every PAC command; uploads returns)
**Planner branch:** `claude/prdiff-track-r-planner-20261021`, based on `codex/prdiff-track-a-a14-a16-planner-handoff-20261021` @ `71a8339` plus copied Track A/B/B26/A1/A13 records.
**Status:** R0 authorized for executor implementation. GPU sub-stages run only by Cheng, inside the gates and budget of the [R0 spec](R0_EXPERIMENT_SPEC.md).

---

## 0. Planner's view for Cheng (read this first)

### How I view the current results

- **NP as a standalone method is closed.** Three independent reasons, any one of which would be enough:
  1. FAST-DIPS (ICLR 2026) occupies the cost/quality slot NP aimed for. Its own Table 1 (FFHQ phase retrieval, best of 4): DAPS 30.253 dB at 122.1 s/run, FAST 29.253 dB at 10.354 s/run, SITCOM 28.512 dB at 37.425 s/run. Our 8-image reproduction (exposed images, GT best of 4): FAST 31.525, DAPS 31.518, NP-LF 30.388, SITCOM 27.439 dB.
  2. NP's per-step selection rule (sample K noises, denoise, keep the best x̂₀ by a measurement loss) is SCG (Huang et al., ICML 2024).
  3. NP's "prior first, measurement late" schedule is already in DiffFPR's official code (RAAR output mixed in with weight √ᾱ_t, so the prior dominates early).
- **NP's one real empirical advantage** was tail reliability at the highest cost: B22 NP-8-RS Good25 95/100, q05 25.1 dB, 412 GPU-s/image, versus two DAPS runs 92/100, q05 15.9 dB, 313 GPU-s/image.
- **GT-free failure detection is published** (FA-FPR, Xue et al., ISCAS 2025): a CLIP ViT-L/14 fake-image detector triggers restarts of a DAPS-style stage. It reports means only, no tails, no fast engine, and it never repairs a failure.
- **What looks genuinely open** (literature pass of 2026-10-21; see [literature basis](2026-10-21_LITERATURE_BASIS.md)):
  1. Repair exact-symmetry failures in place using the prior ratio. Members of the per-channel 180° orbit have identical Fourier magnitudes, so for any noise model the posterior ratio equals the prior ratio. DiffFPR's code enumerates the same 8-member orbit but scores it against a reference image (GT).
  2. Test whether the diffusion prior itself detects the remaining failures as well as an external realness detector.
  3. Evaluate reliability per unit of compute with tail metrics and GT-free selection, across solver families including FAST.
- **Biggest risk:** our own data suggest exact-orbit errors may be a minority of failures. A1.3 found 3 rotation-rescuable endpoints among 42 failures (global rotation only). B22's two-run DAPS had 3 rotation and 2 chromatic failures out of 8 (the chromatic ones were never tested as channel flips). A1.6 had 10 of 260 NP candidates rescuable by rotation. **R0 measures this before we build anything.**

### My plan

| Stage | What | Rough time | Gate |
|---|---|---|---|
| R0 | Failure anatomy, prior-ratio orbit repair and prior-consistent detection, offline on existing outputs (B22, DEV80, A1.x) plus cheap FAST-DIPS K=4 runs on DEV80 and B22 measurements | 1–2 weeks | Pre-registered decisions in the R0 spec |
| R1 | Pipeline on the FAST engine: final and/or mid-trajectory orbit repair, prior-based failure detection and restart; FA-FPR reimplemented as the direct baseline; matched-NFE comparisons on development data | 2–3 weeks | R0 decision |
| R2 | Prospective frozen evaluation on new FFHQ validation-split images; ImageNet phase retrieval if R1 holds on FFHQ | 1–2 weeks | R1 freeze |
| R3 | Writing | — | R2 results |
| T1 (parallel, low priority) | Theory note: per-step hard best-of-K selection has drift O(√h) per step versus O(h) for likelihood weighting, so no continuous-time limit and a step-count-dependent effective guidance strength | desk/CPU only | none |

### Publication image (as of R0 start)

- **Working title:** "Repair, don't repeat: exact-symmetry repair and prior-consistent failure detection for reliable diffusion phase retrieval."
- **Claim to test:** with a fast diffusion phase-retrieval solver, GT-free orbit repair plus prior-consistent failure detection reaches the tail reliability of multi-run quality-first methods at a fraction of the compute. Tail reliability means Good25, 5th-percentile PSNR and catastrophic-failure rate. It should also beat FA-FPR's detect-and-restart on reliability per NFE.
- **Expected content:**
  1. Failure anatomy of DAPS, SITCOM, NP and FAST under GT-free selection: what GT best-of-4 and GT orientation alignment hide.
  2. A short proposition: on an exact measurement-invariant orbit, posterior ratio = prior ratio for any magnitude-noise model. The estimator is a paired-noise denoising-loss difference, with variance and cost analysis.
  3. Prior-consistent failure detection versus UnivFD (FA-FPR's detector) and measurement residuals.
  4. Reliability-per-compute frontier on a prospective panel.
  5. Ablations: repair timing, K, scoring timesteps, signed versus clipped measurements in selection.
- **Venue:**
  - Realistic quick target: IEEE Signal Processing Letters. SPS-journal papers accepted by 2027-01-06 can be presented at ICASSP 2027, but this is tight.
  - Alternatives: ICIP or EUSIPCO 2027; deadlines still to verify.
  - A top ML venue only if R2 includes ImageNet, a clear frontier win, and the theory parts.
  - TMLR is excluded by collaborators.
- **My subjective odds:**
  - R0 shows orbit errors are a material share of FAST/DAPS single-run failures: about 40–50%.
  - Prior-ratio repair accuracy ≥ 90% on FFHQ orbit errors: about 70%.
  - Prior beats UnivFD at detecting non-orbit failures: about 40%.
  - A paper-worthy outcome overall: about 40–50%.
  - If R0 fails its gates, I will recommend at most a short evaluation note and freeing your time for other projects, rather than another rescue cycle.

---

## 1. Binding decisions for Track R

**D1 — Track A stays parked.**
- The NP standalone method paper remains stopped.
- Track A's open reliability question is superseded by Track R, whose hypothesis meets the reopen checklist in `docs/planning/track_a/TRACK_A_PARKED_README_2026-10-21.md` §4:
  - a new observable: the prior ratio on exact orbits;
  - literature gate done, including FA-FPR full text, DiffFPR code, TReg main text and FAST-DIPS;
  - identifiability audit: exact orbit invariance;
  - new data for R2;
  - strong baselines;
  - preregistration.

**D2 — Track B is paused.**
- TB0's STOP is accepted but read narrowly. Its 1-D homometric toy models an ambiguity class that is non-generic in 2-D Fourier phase retrieval, where signals are generically unique up to trivial ambiguities. Its greedy local acceptance also could not cross energy barriers by construction.
- Track B may be re-scoped after R0. If non-orbit, measurement-consistent failures (twins/ghosts) dominate, the next method question is early commitment, studied cheaply with FAST.

**D3 — NP is a baseline only.** NP enters Track R only through the B22 NP-8-RS protocol. No further NP method development.

**D4 — Engines and baselines.**
- FAST-DIPS (native, pinned) is the primary engine, on the speed track.
- DAPS is the quality-track reference.
- SITCOM-4S and NP-8-RS are baselines.
- FA-FPR is reimplemented as the direct competitor in R1.

**D5 — Fairness.**
- Quality-first and speed-first methods are compared within their track, or on a reliability-per-compute frontier with full K-charged cost (UNet calls and wall-clock time, with the timing scope stated).
- No GT in any deployable step. GT best-of-K is reported only as a labeled ceiling.

**D6 — Metrics.**
- Primary: raw-oriented RGB8 PSNR with Good25, Bad20, Bad15, q05 and min.
- Secondary: SSIM and LPIPS.
- Orientation-corrected (GT-aware) PSNR is diagnostic only.

**D7 — Data governance.**
- R0 uses exposed data only, as diagnosis.
- B24 confirmation305 stays locked through R1: no read, list, glob, hash or membership check.
- The R2 cohort is decided at the R2 freeze. Default: new FFHQ validation-split images that appear in no prior manifest, including the full B24 7,424-record screen.
- At the R2 freeze the planner may release confirmation305 as a secondary, hard-enriched confirmation cohort. That decision will be recorded explicitly.

**D8 — Governance.**
- Planner (Claude), executor (GPT/Codex) and PAC operator (Cheng).
- No metaplanner charter or issue #41 reservation is required for Track R; Cheng may still log there.
- The invariants in §5 remain binding.

---

## 2. Short-term plan: R0 (authorized now)

Full binding details are in the [R0 spec](R0_EXPERIMENT_SPEC.md). The executor starts at [`docs/track_r/executors/r0/00_EXECUTOR_HANDOFF.md`](../../track_r/executors/r0/00_EXECUTOR_HANDOFF.md).

R0 answers four questions with mostly existing outputs:

- **Q1 — ceiling.** What fraction of single-run failures per solver family are exact per-channel 180° orbit errors?
- **Q2 — repair.** Does the prior's paired-noise denoising-loss score pick the correct orbit member without GT, without harming good outputs? Compared with UnivFD-scored orbit choice.
- **Q3 — detection.** After repair, does the prior score detect the remaining failures (AUROC) as well as UnivFD and the signed/clipped residuals?
- **Q4 — net effect.** Do GT-free K-run policies with repair close the gap to the GT best-of-K ceiling?

**R0 sub-stages:**

| Sub-stage | Content | Runs on |
|---|---|---|
| R0.0 | Inventory and environment check/create | CPU |
| R0.1 | Code and CPU tests | executor side |
| R0.2 | Candidate manifest and validity gates G0 | CPU |
| R0.3 | GPU smoke, including canonical case G1 (A1.2 image 68539, DAPS rep2: 6.364 dB raw, 31.871 dB after rotation) | GPU |
| R0.4 | Full scoring of existing candidates | 4 GPUs |
| R0.5 | FAST K=4 generation on B22 (100) and DEV80 (80) locked measurements | GPU |
| R0.6 | Scoring of the FAST outputs | GPU |
| R0.7 | Analysis and return | CPU |

- **Budget ceiling:** 16 aggregate GPU-hours. Expected total is about 3–6.
- **Planner pre-authorization:** if G0, G1 and the smoke gates pass, R0.4–R0.6 may proceed without a new planner round trip.

---

## 3. Medium term: R1 and R2 (sketch; specified after the R0 decision)

### R1 — pipeline development on development data only (B24 DEV80; B22 as a second exposed panel)

**Engine:** FAST-DIPS native pixel sampler, pinned upstream `a9758b38`, unchanged; wrappers only.

**Variants, cost-matched:**
- P0: FAST with K runs and executable signed-residual selection.
- P1: P0 plus final orbit repair.
- P2: mid-trajectory orbit repair at a checkpoint level (R_s applied to x_t), then continue.
- P3: prior-consistent failure detection at a checkpoint, then restart. Compared with a UnivFD trigger, i.e. FA-FPR logic on the FAST engine.

**Baselines:**
- FA-FPR as described in its paper (Alg. 1; T=200, N_ode=5, T_I=100, T_f=150, β_y=0.01; configs 4-0.3 and 4-0.1), on our DAPS source.
- DAPS-K with residual selection, SITCOM-4S, NP-8-RS (B22 records).

**Output:** reliability-per-compute frontier (Good25, q05, Bad20, catastrophic rate versus UNet calls and wall-clock time) and a frozen pipeline for R2.

### R2 — prospective evaluation

**Primary cohort:**
- About 200 new FFHQ images: official validation category if the metadata is present, excluding every ID in B19–B26, Track A, DEV80 and the B24 7,424-record screen.
- σ = 0.05, oversampling 2 (DPS padding 64), frozen seeds.

**Secondary:**
- ImageNet phase retrieval with an ImageNet-256 unconditional model (about 100–200 images). This is where TReg predicts the unconditional prior is weakest, and where DAPS reports more 180° failures.
- Optional oversampling 1.0 stress test.
- Optional release of confirmation305 (decision D7).

**Analysis:** pre-registered primary contrasts, paired image-level bootstrap, McNemar on Good25.

---

## 4. Long term

1. **If Track R works:** ImageNet, lower oversampling, possibly latent models. Package with T1 as a stronger paper.
2. **T1 theory note.** Position it as importing the (1,λ)-evolution-strategy progress-rate result (Beyer) and deriving the diffusion consequence. Contrast it with Dandapanthula & Boffi (arXiv 2606.02884, soft plug-in limit) and Porteous et al. (arXiv 2605.06538, gradient guidance strength ∝ 1/step).
3. **Track B re-scope.** Early-commitment versus symmetry-breaking study with FAST as the many-seed engine, only if R0 shows non-orbit measurement-consistent failures dominate.
4. **Second problem: large-box inpainting.**
   - Only through the reliability lens: prior-consistent detection and tail metrics per compute.
   - Caution: in noiseless inpainting every completion consistent with the known pixels has the same likelihood, so "pick by prior" degenerates to MAP selection, with a known bias toward bland fills.
   - The method space is crowded: ReGuidance (NeurIPS 2026), RPU (arXiv 2606.02331), SMC samplers.

---

## 5. Invariants, roles and namespaces

**Invariants (binding):**
- Raw-oriented RGB8 PSNR is the primary metric.
- No GT in deployable selection or repair.
- Every PAC run comes from a pushed commit with a clean tree.
- No agent launches GPU jobs; Cheng runs every command.
- Historical outputs are read-only; new outputs go only under the Track R namespace.
- No overwrite of outputs; resume or refuse.
- No recursive scans of large data, model or output trees: use manifests, targeted paths and depth limits.
- Shell scripts avoid `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout` and broad `pkill`. Long jobs use `nohup` with log, PID, status and stop commands.
- B24 confirmation305 is never accessed.
- Every compute unit is charged: K runs, scoring passes, restarts.

**PAC namespaces** (see `configs/track_r/pac_paths_track_r_v1.json`; to be verified in R0.0):

| Item | Path |
|---|---|
| Checkout | `/egr/research-pac/huang248/pr_diffusion_track_r` |
| Outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/` |
| Logs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/` |
| Caches | `/egr/research-pac/huang248/outputs/.cache/track_r/` (never `/home`) |
| New env (if needed) | `/egr/research-pac/huang248/conda-envs/prdiff_track_r`, cloned from `prdiff_ffhq` |

**Branches:**
- Planner: `claude/prdiff-track-r-planner-20261021`.
- Executor: `codex/prdiff-track-r-r0-<YYYYMMDD>`, created from the planner HEAD.
- The planner merges nothing into `main` without Cheng's request.

**Return loop:**
1. The executor returns per the [R0 return template](../../track_r/executors/r0/R0_RETURN_TEMPLATE.md).
2. The planner writes a decision document with a "Planner's view for Cheng" section (current results, plan, publication image, expected content).
3. The decision goes on this branch under `docs/planning/track_r/`.

---

## 6. Reading list for anyone joining

1. This roadmap.
2. [Literature basis](2026-10-21_LITERATURE_BASIS.md).
3. [R0 spec](R0_EXPERIMENT_SPEC.md).
4. Background only as needed:
   - `docs/b22/b22_3_scientific_closeout.md` (100-image baselines, failure taxonomy);
   - `docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md` (signed vs clipped residuals, the 68539 orientation case);
   - `docs/planning/track_a/TRACK_A_PARKED_README_2026-10-21.md`;
   - `docs/track_b/executors/tb0-homometric-bridge/TB0_STOP_REPORT.md`.
