# Track A planner handoff — A1.4 → A1.6 complete scientific return
**Date:** 2026-10-21 (America/Detroit)
**Recipient:** Track A scientific planner, branch codex/prdiff-track-a-np-paper-planner-20261015
**Author role:** Track A executor / integrated A1.4–A1.6 experimental-return consolidation
**Status:** A1.4 desk study COMPLETE; A1.5 paired native FAST experiment + CPU mechanism/selection audits COMPLETE; A1.6 guarded four-GPU NP×FAST engineering autopilot COMPLETE; **control returned to planner**. No new GPU or data execution is requested or authorized by this document.

## Executive decision request

The A1.4 publication gate identified FAST-DIPS (ICLR 2026) as an indispensable strong baseline, and concluded that a standalone Noise Picking (NP) paper based on its intermediate DAPS–SITCOM efficiency/quality point was not defensible. The subsequently authorized A1.5 and A1.6 explorations have now answered the highest-priority practical questions on the **same eight already-exposed B22 FFHQ images with frozen signed observations**:

- **A1.5 native FAST-DIPS K4:** 31.525134 dB raw oriented common PNG8 PSNR, essentially tied with DAPS K4 31.517951 dB, superior to historical NP-LF K4 30.388019, NP-S2 K4 30.221876 and SITCOM K4 27.438794. FAST K4 sampling-only 14.851s/image; ~32.440s/image summing four full program walls **after Python imports** (not external subprocess wall). No model-weight/source/magnitude-operator identity issue remained at validated FAST adapter: original FFHQ checkpoint SHA and 362 parameter/buffer schemas with seven exact probes verified, same frozen measurement tensors, same forward-physics formula.
- **A1.5 within-FAST GT-free selection:** minimum signed full-Fourier terminal residual among four saved FAST candidates gives **31.483358 dB**, Good25 **8/8**, compared to unavailable GT oracle **31.525134**; regret **0.041776dB**, exact oracle-root identity 5/8, mean within-image Spearman between negative residual and PSNR ~0.60. This is strong *exposed-panel diagnostic*, **NOT new validated method**.
- **A1.6 NP×FAST four-GPU exploration:** 17 preregistered development arms, Stage0/1/2 all PASS, **260/260 successful independent candidate roots**. Best final NP variant (FAST post-selection correction) K4 oracle **30.4151dB**, original NP-LF on the A1.6 *new paired seed set* **30.2307dB**, gain **+0.1844dB** largely from one image. Sparse two-branch plus reduced early proposals plus FAST correction retained NP-LF K4 oracle within **−0.0099dB** while halving UNet calls and reducing subprocess wall ~42%, but **component causality is not isolated**. No A1.6 arm approaches FAST's native K4 quality–compute frontier.
- **A1.6 signed-vs-clipped final candidate ranking:** among four generated NP-LF+FAST-post roots, original-signed-y residual selector produces **30.387dB / Good25 8/8**, while clipped-y residual selector produces **23.619dB / Good25 5/8**; for two-branch NP, **30.153 vs 22.952dB**, respectively. Same images and four-root costs fully charged. This independently *repeats the exposed-data sign sensitivity* seen in A1.3, not prospective validation or a cross-method calibrated scorer.

**Planner must decide:** (A) stop NP as a standalone method paper and retain it as an engineering/reproducibility negative result; (B) permit ONLY a tightly defined, independent validation of signed-residual within-method GT-free candidate selection/phase-retrieval failure reliability; (C) permit ONLY a cost-matched NP branching/correction factorial with an explicit distinct novelty hypothesis; (D) find a more principled basin-correctness observable in coordination with Track B. Combinations are possible only with explicit source/data novelty and prospective protocols. **No next run is automatically authorized.**

## 1. A1.4 — desk-science publication kill gate (completed 2026-10-20)

Authoritative return: https://github.com/Cheng-HanHuang/pr_diffusion/blob/a2b6408551f4df12b69b8edb45bbc4431874c821/docs/track_a/executors/a14-publication-gate/A14_EXECUTOR_RETURN.md .
Dedicated branch: codex/prdiff-track-a-a14-publication-gate-20261020, latest report HEAD a2b6408551f4df12b69b8edb45bbc4431874c821. A1.4 **PAC_OPERATIONS=0, GPU=0, NEW_MEASUREMENTS=0**.

1. Paper/original-source audit: FAST-DIPS ICLR2026 source implements the same oversampled 256 RGB channelwise Fourier-magnitude (pad64, orthonormal centered FFT) and signed Gaussian noise sigma=.05 as our A1.1–A1.2 problem. FAST paper original PR reports **best of four**, while its cited runtime is *mean ONE candidate's sampling time*. On FAST's DIFFERENT 100-FFHQ cohort: native paper Table1 FAST **29.253dB / 10.354s per run** (41.416s for four sampling-only roots); DAPS **30.253dB / 122.100s per run**; SITCOM **28.512dB / 37.425s per run**. These are **not paired against A1.2** and cannot by themselves establish NP inferiority.
2. Prior-art decisions: H1 quality–compute candidate frontier weak without FAST; H2 GT-free failure detection/selection weak given A1.3 signed/clipped reversal, wrong-basin and 180° physics ambiguity; H3 prior-mismatch robustness unsupported and overlaps established literature. An independent NP method publication was **not recommended**. Conditional physics/reliability benchmarking proposal required a genuine differentiator before GPU escalation.
3. Specific unresolved literature item: Xue et al., **Fourier Phase Retrieval With Diffusion Priors Guided by Failure Detection**, ISCAS2025 DOI 10.1109/ISCAS56072.2025.11042933. A1.4 verified existence/title/venue but **did not obtain or review the complete algorithmic paper**. Its detector's actual overlap remains UNKNOWN; do not assert priority, nonoverlap, or specific detector content without full article.
4. Distinct A1.4 docs: A14_PRIOR_ART_MATRIX.md, A14_CLAIM_KILL_MATRIX.md, A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md (design only). The latter stipulates independent provenance/measurement adapter, K1/2/4 all charged, original FFHQ weights/checkpoints, common PNG8 metric, native pipeline, no B24 confirmation305 access, and explicit Track B priority.

## 2. A1.5 — direct paired native FAST-DIPS (actual user PAC execution)

**Source identities:** our execution source HEAD d51170f77ee672990c22f1703c5997e949dda9d6, branch codex/prdiff-track-a-a15-fastdips-exposed-20261020; official FAST source pinned upstream a9758b38906b7c7950fa04491f37f6634948db26. Native FAST uses T=150 diffusion transitions/root, K_ADMM=2 *inner iterations* (NOT trajectory restarts), S=5 image-gradient steps per ADMM iteration, rho=200, epsilon=.05, single FFHQ prior checkpoint SHA256 81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa. User installed an isolated Blackwell-compatible environment fast_dips_a15_blackwell (torch2.7.1+cu128, sm120 GPU RTX PRO 6000). Sample checkpoint loader validated 362 parameter+buffer identities and 7 exact content probes. Native FFT formula/measurement content and timing gates PASS. No new noise/images created; GT accessed post-sampling only.

**Exactly eight B22-exposed image IDs:** 69073, 68539, 67948, 60140, 67293, 67174, 61101, 66631. Frozen input manifest SHA256 fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490. Primary metric is raw, oriented, uniformly rounded 8-bit RGB PNG PSNR (GT offline). Rotated 180° score secondary. K1,K2,K4 consist of independent trajectory *prefixes*, and every sampled root is charged.

| Solver, mean raw RGB8 GT best-of-K | K1 | K2 | K4 |
|---|---:|---:|---:|
| **FAST-DIPS** | **26.879006** | **30.525317** | **31.525134** |
| DAPS (historical A1.2) | 21.804663 | 28.633940 | 31.517951 |
| NP-LF (historical A1.2) | 21.145301 | 23.962594 | 30.388019 |
| NP-S2 (historical A1.2) | 21.272546 | 26.409081 | 30.221876 |
| SITCOM (historical A1.2) | 22.072058 | 27.424658 | 27.438794 |

FAST K1 Good25 6/8, Good30 6/8, Bad20 2/8. Failed initial roots: image67174 **13.249dB**, image66631 **10.968dB**. FAST K2 Good25 7/8, Bad20 0/8, FAST K4 Good25 **8/8**, Good30 **7/8**, Bad20 0/8. The two initially failed images reach K4 GT-oracle **28.32** and **32.21dB**. FAST K4 mean versus DAPS +0.00718dB, a practical tie on eight development images, not population superiority.

FAST sample-only K1 3.72547s/image, K2 7.46307s, K4 14.85115s, from CUDA-event/synchronized wall timing. K4 summing four full sample-program walls *after imports* ~32.43957s/image. Compare A1.2 historical four-candidate whole subprocess walls per image DAPS ~260.11, NP-LF ~147.40, NP-S2 ~147.17, SITCOM ~122.48; A15 after-import vs A1.2 external timings are not identical scopes. **Do not claim an exact speedup ratio by dividing them.** In rough terms FAST dominates the observed NP quality–wall region with this caveat.

Original pinned source and preflight / launch specification: https://github.com/Cheng-HanHuang/pr_diffusion/blob/d51170f77ee672990c22f1703c5997e949dda9d6/docs/track_a/executors/a15-fastdips-exposed/A15_START_HERE.md .

## 3. A1.5 — CPU mechanism and scorer audits (all completed after FAST K4)

**M0 physics audit:** source branch codex/prdiff-track-a-a15-mechanism-audit-20261021 at execution HEAD a9b5dedb0c5586d55991a7f9cb82f13aea847fff. Actual user PAC PASS from 8 frozen signed observation tensors plus all 32 saved output floats/PNGs (no new GPU, no GT reads). Receipt SHA256 b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74. For FFT magnitude A(x)≥0 componentwise, EVERY image obeys

    ||A(x)-y||_2 ≥ ||min(y,0)||_2.

All eight recorded observations have a negative-part lower bound **18.566–19.569**, mean **19.085433**, versus original native FAST's global residual radius **epsilon=.05**. Thus **all eight nominal hard image-space constraints are provably infeasible**, independently of the candidate. Across images, minimum of the four FAST final-terminal residuals ~33.07–33.21, close to sigma sqrt(3×384×384)≈33.26 in the actual signed-noise convention. This is not an algorithm validity failure or proof the correction is useless: native K2 ADMM projects an **auxiliary measurement variable**, while the finite-iteration image estimate need not satisfy the hard ball.

**Source-derived mechanism, NOT yet causal ablation:** at each step FAST initializes p=D_theta(x_t,sigma), x=p, v=A(p), u=0. The FIRST K2 ADMM image-update gradient mathematically vanishes; the first auxiliary projection+dual update sets b2=2v1−A(p), and the SECOND image update has initial gradient 2rho(1−alpha) J_A(p)^T(A(p)−y), alpha=min(1,epsilon/||A(p)−y||). This is the local gradient only, not equivalence of entire nonlinear S=5 update to simple gradient descent. Final FAST timestep source returns x0hat instead of final measurement-corrected x0y; previous corrections still affect the trajectory. Eliminating the initial mathematically redundant update, and final corrected output, **have not been GPU ablated**. Avoid converting this observation into a new novelty claim without source-equivalence experiments.

Detailed A15 M0 findings: https://github.com/Cheng-HanHuang/pr_diffusion/blob/a9b5dedb0c5586d55991a7f9cb82f13aea847fff/docs/track_a/executors/a15-mechanism-audit/A15_FAST_NATIVE_MECHANISM_FINDINGS.md .

**Saved FAST candidate selector:** source branch codex/prdiff-track-a-a15-residual-selector-m1-20261021, execution HEAD 0bf8cb750068ef56e8fbec919a193bf6ab4ea58c; CPU-only PASS, receipt SHA256 dac02309c91c985a32b6784bdeb047b900ec582ea0457c85c20ad546e4d2aa91. Among K independent roots, original signed-y minimum full Fourier residual yields:
- K1 chosen = GT oracle 26.879006 (only one root).
- K2 signed-residual selection **30.504153dB** vs GT oracle **30.525317**, regret **.021164dB**, 6/8 exact chosen oracle root, Good25=7/8.
- K4 signed-residual selection **31.483358dB** vs GT oracle **31.525134**, regret **.041776dB**, 5/8 exact chosen oracle root, Good25 **8/8**.
- Separate saved-32-roots per-image rank audit mean Spearman corr between negative residual and PSNR **0.60**. These eight are exposed; no new GT-free prospective selector was verified. Physically equivalent Fourier conjugate-180° orientations remain indistinguishable from magnitude-only data in exact symmetry.

## 4. A1.6 — NP×FAST four-GPU research autopilot (actual PAC success)

Execution source branch codex/prdiff-track-a-a16-np-fast-autopilot-20261021 at **pinned HEAD 66fad4a6fafcf7f224b2e8cb6e388137b18d7569**. User PAC CPU preflight 6/6 test PASS; A16_CPU_PLAN_PASS receipt SHA256 **b7a87c185a88cb1fee1d86c18aae456c58dbd97461be3c01fa1f6a6e720e5db6**, 17 arms, eight images. Native NP replay GPU0 reference reconstructed exactly: max absolute difference **0.0**, 2200 denoiser calls and no corrections, source/model/input gates. Experiment then executed on all four RTX PRO 6000 Blackwell GPUs in three stages: Stage0 17 arms×4 selected diagnostic images×rep0; Stage1 two promoted arms/lane on all 8×reps0,1; Stage2 best promoted arms/lane with controls on all 8×reps0–3. Predeclared GT-PSNR development-only stage utility was used to promote arms. New seeds fixed by (image,rep) across arms but diverging proposals need not keep identical RNG streams. **260/260** roots PASS, 260 worker logs, 260 subprocess wall receipts, all GT-after-sampling, no new measurements, no protected cohort. Full run PASS and metadata ZIP package PASS.

User-uploaded metadata ZIP **A16_COMPLETE_METADATA_RETURN.zip**, SHA256 **12768e9eedd49b73ba6608c81a8e7521414e666e650b81f6760c0fd307d897df**. Archive manifest 1063/1063 entries validated SHA/size and ZIP CRC; ZIP has 1064 entries with manifest. The completed independent scientific audit is at https://github.com/Cheng-HanHuang/pr_diffusion/blob/584052f75d0a564e56d6d1ab8ac8e759ad678f01/docs/track_a/executors/a16-np-fast-autopilot/A16_POSTRUN_SCIENTIFIC_AUDIT_20261021.md .

**17 engineering arms across four lanes:**
- GPU0: native LF, FAST post-selected correction, matched *conceptual* ordinary gradient correction, replace original GS late projection with FAST correction.
- GPU1: native S2, FAST before scoring, FAST both before-score+propagated, corrected-candidate residual scoring.
- GPU2: sparse five proposals, adaptive 1/2/3 proposals, short-lived two-parent beam plus FAST, and plain early K2 plus FAST. Note these are NOT truly cost-matched factorial controls in the final stage.
- GPU3: short VP schedules 500/250/150 steps with FAST, separately labeled signed-y correction variant, and a simple residual-moment scoring variant (NOT reproduction of a published full distribution-consistency algorithm).

**Final-stage identical eight-image cohort on a NEW A1.6 seed namespace** (do not numerically merge A1.2 independent realization means as if paired). Report mean per-image **GT oracle best of 4 independent roots** in dB, root-level Bad20 count out of 32, sum of FOUR subprocess wall times per image, and FOUR-root UNet forward count:

| Final arm | K4 oracle raw dB | Bad20 / 32 single roots | K4 full external wall/image | UNet forwards / K4 image | K4 Good25 images |
|---|---:|---:|---:|---:|---:|
| NP LF + FAST post (lf_fast_post) | **30.4151** | 9 | **123.30s** | 8800 | 8/8 |
| NP LF original on A16 seeds (lf_native) | 30.2307 | 9 | **123.84s adjusted** | 8800 | 8/8 |
| NP LF + FAST before-score (lf_fast_prescore) | 30.1910 | 12 | 123.54s | 8800 | 8/8 |
| Sparse two-branch + FAST (lf_beam2) | 30.2209 | 9 | **71.51s** | **4392** | 8/8 |
| NP S2 original on A16 seeds (s2_native) | 25.7190 | 17 | 124.24s | 8800 | 6/8 |
| Shortened NP 150-step + FAST (lf_150_fast) | 27.0045 | 16 | 28.31s | 696 | 7/8 |

**Critical timing correction:** the logged native LF K4 raw subprocess wall is **127.29s/image**, but first lf_native/69073/rep0 was also the verification smoke containing **27.64496 seconds of extra native replay**. Subtracting this exact one-time verifier cost divided by eight images gives **~123.84s/image comparable method-only**. Do NOT claim FAST post improves NP runtime based on the unadjusted 127.29s.

**Corrections vs original NP:**
- Adding 16 sparse post-selection FAST effective image updates/root yields K4 +**0.1844dB** over native LF, 6/8 paired image wins, but image 67174 alone contributes **+1.206dB**; other seven improve by only **+0.038dB mean**. Across all 32 roots, mean gain only +**0.032dB**, **Bad20 unchanged 9/32**. It is an image-specific conditional improvement, not a robust/catastrophic-tail success.
- Before-score correction (not propagated after winning) gave K4 −**0.040dB**, root mean −**1.694dB**, and catastrophic raw roots **12/32 vs 9/32**; correction-before-ranking can destabilize basin selection.
- Two-branch+FAST sparse proposal arm gives K4 −**0.0099dB** vs native LF while reducing 4-root UNet evaluations **8800→4392 (~50.1% retained, 49.9% reduction)** and adjusted wall **123.84→71.51s (~42.3% reduction)**; single-root mean quality worsened −**0.697dB** and Bad20 stays 9/32. Since proposal schedule, branching and correction all changed, **DO NOT attribute matched K4 to branches without sparse-only matched controls**. K4 GT oracle hides per-root failures. Cross-method FAST itself is still substantially more efficient.
- Shortened 150-step NP+FAST is a **different VP/NP schedule** from native 150-step EDM FAST; it reached only **27.004dB** K4 (−3.226 vs A16 native LF, −4.521 vs A15 FAST), **16/32** raw Bad20, and image 68539 had best-of-four only ~**9.85dB**. This is not a reliable 150-step substitute or proof a FAST denoiser call has different quality per call than NP.
- Native S2 had poor K4 on this A16 seed set 25.719dB, including two K4 image failures; A1.2 historical NP-S2 30.222dB belongs to a **different seed set**. Do not infer universal S2 inferiority.

**A1.6 post-hoc deployable candidate selection retrospective on the exact same saved four roots:** rescore each completed candidate with the original *signed* noisy y vs the *clipped* y+ target; fully charge generation of all four candidates. GT only for post-selection outcome.

| A16 arm | GT oracle K4 | Signed minimum-residual K4 | Clipped minimum-residual K4 | Signed Good25 |
|---|---:|---:|---:|---:|
| lf_fast_post | 30.415 | **30.387** | **23.619** | 8/8 |
| lf_native | 30.231 | 30.179 | 30.154 | 8/8 |
| lf_fast_prescore | 30.191 | 30.177 | 30.117 | 8/8 |
| lf_beam2 | 30.221 | **30.153** | **22.952** | 8/8 |
| s2_native | 25.719 | 25.390 | 21.534 | 6/8 |
| lf_150_fast | 27.004 | 26.646 | 21.447 | 7/8 |

Particularly, clipped ranking among lf_fast_post K4 candidates returned low raw PSNR images **66631 10.62dB**, **67174 13.16dB**, **67293 11.66dB** despite good alternatives already available; signed ranking picked good raw candidates on all eight and within-method average oracle regret only **0.0277dB**. For beam2, signed regret ~0.0682dB vs clipped catastrophic failures. Mathematical explanation: with nonnegative A(x) and signed y, letting y+=max(y,0), the difference ||A(x)−y||²−||A(x)−y+||² equals sum over negative coordinates of **2 A_i(x)|y_i|+y_i²**, so clipping removes a *candidate-dependent* term, not only a constant. Original NP sampling itself uses clipped y in most arms; final signed-y reranking exploits the still-available originally observed y. It is **not automatically evidence to replace NP's internal observation treatment**; signed-vs-clipped correction during diffusion is a separate intervention.

Symmetry and calibration caveats: 10/260 low-raw candidates had rotated-180 PSNR≥25dB, and exact Fourier magnitude cannot distinguish identical-magnitude conjugate-reflection without additional asymmetry; raw oriented PSNR remains primary. A1.3 pooled minimum-residual cross-method choice catastrophically favored SITCOM, so strong *within-family* signed residual ranking does not prove a universally comparable or calibrated selector. In A1.6 the 17-arm outcome-dependent promotions and 8 exposed images invalidate population significance/held-out claims. Unknown methodological overlap with Xue ISCAS 2025 remains.

## 5. Full provenance and PAC artifact locator — read only, not an instruction to run

**Common model and frozen input**:
- PAC workspace root: /egr/research-pac/huang248
- FFHQ model: /egr/research-pac/huang248/models/ffhq_10m.pt ; SHA 81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa
- A1.2 frozen 8-input manifest: /egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1/A1_2_INPUT_FREEZE_COMPLETE.json ; SHA fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490
- Historical A1.2 oracle summary (DAPS, NP-LF, NP-S2, SITCOM): /egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_k4_v1/A1_2_OFFLINE_ORACLE_SUMMARY.json; source GPU execution f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd
- Original A1.3 source execution b55c5a47f010fad019270b77c4c9366a9c8b99cc, retrospective doc f002643fbaff0b0541886e625258f428cac3ea90. Included only as necessary predecessor to A14 selector discussion.

**A1.4 desk-only (NO PAC outputs)**:
- https://github.com/Cheng-HanHuang/pr_diffusion/tree/codex/prdiff-track-a-a14-publication-gate-20261020/docs/track_a/executors/a14-publication-gate

**A1.5 user-owned PAC source/results**:
- A15 source checkout: /egr/research-pac/huang248/pr_diffusion_track_a_a15_fastdips ; Git source d51170f77ee672990c22f1703c5997e949dda9d6
- Native FAST source checkout: /egr/research-pac/huang248/external/FAST_DIPS_A15 ; HEAD a9758b38906b7c7950fa04491f37f6634948db26
- Native FAST GPU interpreter: /egr/research-pac/huang248/conda-envs/fast_dips_a15_blackwell/bin/python
- Output root: /egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/
- A15 Stage0 frozen physics/model CPU receipt: a15_0_cpu_preflight_v1/A15_PREFLIGHT_PASS.json ; SHA 963a9e4b2c3ed7b2c8557ef5ff759d86df22425a00bf4ce2cf3752984ccbb8aa
- Native 1-image smoke: a15_1_single_native_fast_smoke_v1/A15_SMOKE_COMPLETE.json ; unmodified original smoke source
- All eight×four FAST reconstruction roots and per-run receipts: a15_2_native_fast_k4_exposed_v1/<image-id>/rep<0-3>/A15_SMOKE_COMPLETE.json
- K1 paired report: a15_3_report_k1_v1/A15_PAIRED_SUMMARY.json
- K2 paired report: a15_5_report_k2_v1/A15_PAIRED_SUMMARY.json
- K4 paired report: a15_4_report_k4_v1/A15_PAIRED_SUMMARY.json
- M0 CPU mechanism checkout: /egr/research-pac/huang248/pr_diffusion_track_a_a15_mechanism_audit ; HEAD a9b5dedb0c5586d55991a7f9cb82f13aea847fff
- M0 receipt: a15_6_mechanism_cpu_audit_v1/A15_MECHANISM_AUDIT_COMPLETE.json, SHA b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74
- M0 per-candidate CSV: a15_6_mechanism_cpu_audit_v1/A15_PER_CANDIDATE_MECHANISM.csv
- Saved-root residual selector checkout: /egr/research-pac/huang248/pr_diffusion_track_a_a15_residual_selector ; HEAD 0bf8cb750068ef56e8fbec919a193bf6ab4ea58c
- Saved-root selector receipt: a15_8_fast_residual_selector_cpu_v1/A15_RESIDUAL_SELECTOR_COMPLETE.json ; SHA dac02309c91c985a32b6784bdeb047b900ec582ea0457c85c20ad546e4d2aa91
- Derived source-desk continuation only: https://github.com/Cheng-HanHuang/pr_diffusion/blob/453ed54883290267225cf132bc44182b1f4d9020/docs/track_a/executors/a15-mechanism-audit/A15_M0_VERIFIED_SCIENTIFIC_RETURN_AND_M1_DECISION.md

**A1.6 user-owned PAC source/results**:
- A16 pinned execution checkout: /egr/research-pac/huang248/pr_diffusion_track_a_a16_np_fast_autopilot ; HEAD 66fad4a6fafcf7f224b2e8cb6e388137b18d7569
- NP interpreter: /egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python
- Output root: /egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021/
- CPU plan: A16_CPU_PLAN_PASS.json ; SHA b7a87c185a88cb1fee1d86c18aae456c58dbd97461be3c01fa1f6a6e720e5db6
- FAST residual-rank CPU diagnostic: A16_SAVED_FAST_SELECTOR_FORENSICS.json
- Every NP arm candidate and timing: runs/<arm>/<image-id>/rep<0-3>/{A16_CANDIDATE_COMPLETE.json,A16_SUBPROCESS_WALL.json,reconstruction.pt,reconstruction.png}; logs/<arm>__<image-id>__rep<r>.log
- Stage summaries: A16_STAGE0_COMPLETE.json, A16_STAGE1_COMPLETE.json, A16_STAGE2_COMPLETE.json; ultimate completion A16_AUTOPILOT_FINAL.json (all PASS as audited)
- Metadata-only archive: A16_COMPLETE_METADATA_RETURN.zip ; independent SHA 12768e9eedd49b73ba6608c81a8e7521414e666e650b81f6760c0fd307d897df (uploaded by user). No original GT tensor/images/checkpoint are inside the metadata ZIP.
- Independent A16 analysis branch: codex/prdiff-track-a-a16-results-audit-20261021, final scientific audit HEAD 584052f75d0a564e56d6d1ab8ac8e759ad678f01. Full per-image tables, signed/clipped selection and corrected smoke overhead documented there.
- A16 actual physical GPU UUIDs: GPU0 8c9c6250-7b65-20d8-5c81-d6cb618810c3; GPU1 883c037a-34d2-48c4-467f-9a352fd8fdff; GPU2 c381c0f4-1dbc-004f-7d3a-1d7f7794dffe; GPU3 7d65c050-d7e8-5a6b-ee38-1d72d7a5696a (prefixes all preceded by GPU-). All four used for Stage0–2; lease released after completion.

**Current operation ledger:** A1.4 zero hardware. A1.5 FAST 32/32 K4 plus initial separate single-image smoke; M0 32/32 saved reconstructions inspected CPU; A15 selector 32-root metadata CPU. A1.6 260/260 successful independent NP candidate runs across GPUs0–3, completed and packed. No confirmed further outstanding GPU job/lease by Track A. All scripts, data, outputs, physics and source pinned as above. No B24 confirmation305 (or its registry/membership) was accessed in these stages, as defined by receipts and run source; cannot claim anything about its test performance.

## 6. What the planner must resolve before another experiment

**A. Publication positioning and stop criterion.** A1.4 was explicitly a no-go for a simple NP tradeoff paper. A1.5 direct paired FAST baseline reinforces that no NP hybrid yet provides quality–compute superiority. A1.6 +.184 K4 FAST-post gain is too small/conditional, not method-level novelty. Generic finite ADMM correction, particle branching, noise proposal adaptivity, orientation ambiguity, and normalized residual scores have substantial pre-existing literature. The planner is free to **STOP NP publication**; retaining these clear reproducibility results is scientifically legitimate.

**B. Highest-value independent validation opportunity: signed versus clipped final residual candidate selection.** Potentially test a *pre-frozen within-method* final signed-y selector vs clipped and native selectors for NP, FAST, DAPS, SITCOM under exact common y, all candidate costs, raw PSNR orientation, root-level and image-level failure rates and cross-method calibration. However A1.3 found DAPS sign preference REVERSED, and pooled cross-family min residual chose SITCOM all eight, so a universal signed selector claim is already contradicted on development data. A selector may be well behaved within families but not cross-method or under 180° ambiguity. Full Xue ISCAS2025 method must be reviewed before claiming failure-detection novelty. No new dataset can be called independent if membership/exposure overlaps B22/B24; protected B24 confirmation305 must not be inspected.

**C. Cost-matched multi-basin exploration factorial, only if a distinctive causal hypothesis exists.** Minimum controlled comparisons should separate (i) sparse K schedule off/on, (ii) two-state branch persistence off/on, (iii) FAST correction off/on (at least a constrained 2×2 plus sparse/no-branch control) with NFE and actual subprocess wall matched or Pareto frontier reported. Report K1/K2/K4, per-root tail, raw orientation failure and GT-free choice. A16 observed branch survives only as *compound package*, with no independent attribution, and has worse mean individual-root quality. Any experiment must compare against native FAST strong baseline at equivalent total compute. Track B owns new basin-transition research; avoid overlapping Track B algorithm agendas without planner/metaplanner consent.

**D. Native FAST mechanism check (if potentially useful for simplification).** Controlled numerical replay with original and mathematically shortened K2 ADMM, exact source operator, common frozen x0hat, check first zero-gradient loop, compare final returned x0hat vs x0y, and plain residual-gradient control matched FFT/VJP calls. Do not treat source algebra or native hard-ball infeasibility alone as a proven correction mechanism, bug, or publication novelty. CPU algebra has been verified; GPU causal experiment NOT DONE.

**E. Prospective-data gate and resource governance.** Before any confirmatory conclusion, identify an independently sourced/licensed cohort that is not user-tuned B22 eight/B24 DEV80; freeze exposure/measurement/model SHA and metric; register a non-overlapping new DATA/RUN reservation in issue #41 with Track B priority and exact PAC storage. Set effect size and futility criteria, sample-size precision target, and statistical unit *image* (not 4 K correlated roots). Do NOT silently reuse eight exposed outcomes as an independent final test or use unavailable GT in runtime candidate selection. The planner should decide which, if any, specific action warrants renewed GPU allocation.

## 7. Explicit handback

**Formal status:** A14_DESK_SCIENCE_COMPLETE / A15_NATIVE_FAST_K4_AND_M0_AND_RESIDUAL_CPU_COMPLETE / A16_FOUR_GPU_260_ROOTS_COMPLETE_AND_AUDITED / ALL_EXPOSED_ENGINEERING_DATA / NO_PROSPECTIVE_VALIDATION / FAST_QUALITY_COMPUTE_FRONTIER_STILL_STRONG / PLANNER_DECISION_REQUESTED / NO_ADDITIONAL_GPU_AUTHORIZED.

Please record a new **Track A scientific planner decision** on planner-owned branch codex/prdiff-track-a-np-paper-planner-20261015 and canonical issue https://github.com/Cheng-HanHuang/pr_diffusion/issues/41. Planner may stop or reframe, seek missing literature, design a true independent test, or delegate a new executor; this handoff does not appoint new executors, claim fresh significance, or reserve devices.
