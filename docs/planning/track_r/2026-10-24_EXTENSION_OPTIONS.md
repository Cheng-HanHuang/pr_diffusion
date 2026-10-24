# Extension options for the audit paper — planner note (2026-10-24, late morning)

> **Update, 2026-10-24 evening, after the extension research pass "Scoping Extensions for a EUSIPCO 2027 Audit…".**
>
> **What changed:**
> - **X3 is narrowed.** TGD (arXiv 2605.03712) already shows that residual-selected DAPS with N=4 beats N=1 at matched wall-clock on FFHQ and ImageNet phase retrieval, with an A-TGD particle sweep up to N=16. X3 therefore becomes **FAST-DIPS K-scaling with failure-rate (Good25) frontiers and our symmetry-aware GT-free selector**, explicitly framed as extending TGD. A 60–100-image subset is acceptable.
> - **X4 is narrowed.** PC-DAPS already reports ImageNet orientation prevalence (DAPS 42.0% flipped).
>   - **X4a (new, minutes of GPU):** run the frozen-style CLIP test on 100 ImageNet ground-truth images and their 180° rotations. Compare a generic prompt, a prompt ensemble and the ImageNet diffusion prior's denoising loss. No prior work compares these.
>   - **X4b (optional):** FAST-DIPS K = 4 on 100 ImageNet images (≈2 GPU-h). No DAPS on ImageNet.
> - **X5 and X6 are deferred** to any journal version.
> - **SITCOM's own rule:** "we report the best results from 4 independent runs, following the convention in (Chung et al., 2023b; Zhang et al., 2025)", criterion undefined (arXiv 2410.04479 §4.1). Its runtime statement is internally inconsistent (§4.1 vs App. G).
>
> **Unchanged:** X1 and X2 are open, zero-GPU, and first. The EUSIPCO 2025/2026 rule is 5 pages including references.
>
> **Theory question** (Cheng, evening): a research pass on theoretical analyses of DPS/DAPS/SITCOM/FAST-DIPS was launched. See §5.

**Trigger.** Cheng asked whether the gap to EUSIPCO 2027 (deadline 2027-02-06, verified) can be used to extend the study, and whether an observation pass or research pass is needed.

**Status of this note.**
- Options and a recommendation only.
- **Authorized now:** CPU work (R0-N.2 revision C, §A7).
- **Not authorized until a planner pre-registration and Cheng's go:** any GPU extension (X3–X6).

**Inputs:**
- LETTER_PLAN v1.3;
- decision §10;
- the 2026-10-24 research pass, "Prior Art, Scoop Risk and Venue Choice";
- the planner's code and paper checks in §1 below.

---

## 0. Planner's view for Cheng

### How I view the current results

**The core audit is complete and verified on FFHQ.** Four solver families, two cohorts. Its findings:
- GT-free selection recovers 93–129% of the K-run gain;
- 93 of the 99 failures still left after selection are generation failures;
- SITCOM's failures fit y in 57% of cases;
- generic orientation tests suffice.

**Its weak points, as a reviewer will see them:**
1. **Generality:** FFHQ only, one operator setting.
2. **"Residual selection is classical":** prDeep, deepECpr and TGD already use it. Our contribution is the measurement, so the measurement has to be sharp.
3. **No practical "so what" in compute terms.**

The extensions below target exactly these three points. Nothing else is worth the time.

### My plan

1. **Now, CPU only:** an observation pass folded into the pending R0-N.2 block (amendment 2, revision C, §A7). It needs no extra block from Cheng.
2. **Now:** a research pass on extension directions.
3. **By 2026-11-07:** the planner pre-registers at most two GPU extensions, using the observation table and the research report. Cheng decides go or no-go.
4. **Drafting continues in parallel.** K4 (2026-12-11) remains the draft-ready date, and EUSIPCO is 2027-02-06.

### Publication image (current)

- **Format.** A 5-page EUSIPCO audit paper. The EUSIPCO 2026 rule is 5 pages including references; confirm the 2027 rule.
- **Content:**
  - the protocol ladder, with a compute axis if X3 runs;
  - cross-solver anatomy;
  - the post-selection decomposition;
  - the protocol-inflation table (X1);
  - orientation tests;
  - one generality result (X4 or X6), if run.
- **Odds** (planner judgment): about 35–45% as is, perhaps 45–55% with X1 + X3 + one generality result.
- **Fallback:** ICIP 2027.

---

## 1. Observations already made this morning (desk and code, no compute)

**"Best-of-4" is not one protocol.**

| Source | What it actually does | Evidence |
|---|---|---|
| DPS (ICLR 2023) | Main text: "we generate 4 different samples for all the methods, and report the metric based on the best samples". The criterion is **unspecified** for DPS. Appendix D.2: classical ER/HIO/OSS are repeated four times, and "the best one with the smallest mean squared error between the measurement and amplitude of the estimation in the Fourier domain" is reported, i.e. **residual selection** | [MAIN] arXiv 2209.14687v4 PDF |
| DAPS code (`e7a77d0`, `eval.py` `display`) | Reports, for **each metric separately**, the best value over runs: PSNR max, SSIM max, LPIPS min (`get_eval_fn_cmp`). The three numbers can come from **different runs** | [CODE] |
| FAST-DIPS code (`a9758b3`, `misc.py` `display`) | The same per-metric rule, copied from DAPS | [CODE] |
| PC-DAPS (paper; code `59b2f04`) | Its own numbers are the mean over the bag's particles. Baselines are "closest to the ground-truth image", citing DPS and DAPS | [FULL] [CODE] |
| deepECpr; TGD | DPS/DOLPH and DAPS baselines selected by lowest measurement residual | research pass [MAIN-excerpt] / [2nd] |

**Consequence.** The audit can state exactly which protocol each number comes from. A **per-metric oracle** is a third, stronger inflation that no paper we found quantifies; X1 measures it. SITCOM's own selection rule still needs checking from its code or text; Cheng may know it.

## 2. Extension menu (ranked for this paper)

| ID | Extension | Question answered | Compute | Value for the paper | Main risk |
|---|---|---|---|---|---|
| **X1** | **Protocol-inflation decomposition**: per-metric oracle vs single-pick oracle vs GT-free (E1) vs mean-of-K vs first run, for PSNR, SSIM and LPIPS. Plus a literature protocol table (§1) | How much of a reported number is GT, how much is selection, how much is K | CPU (SSIM/LPIPS on 3,240 runs) | **High**: sharpens the main claim; new; cheap | None material |
| **X2** | **Observation and hardness pass**: seed-vs-image variance split; cross-solver agreement on which images fail; GT-free hardness predictors from y; what SITCOM's measurement-consistent failures look like; contact sheets | Are generation failures intrinsic to the image? Are they predictable without GT? | CPU | **Medium–high**: explains "what remains"; may yield a GT-free reliability flag | May find nothing crisp. Then it becomes one sentence |
| **X3** | **FAST K-scaling and compute frontier**: 12 more FAST runs per image on the same 180 ids (K → 16), GT-free selection + CLIP; then Good25 vs seconds per image for all four solvers | Does GT-free best-of-K keep paying? Can a cheap solver × K beat an expensive solver × 1? Where does it saturate? | ≈2 GPU-h FAST + seconds of CLIP | **High**: the practical message, and it separates seed luck from intrinsic hardness | Cross-solver timing comparability; must record GPU type per source (§A7) |
| **X4** | **ImageNet audit**: FAST K = 8 on 100 ImageNet images; optional DAPS K = 4 on a 25-image subset. Orientation by CLIP with generic prompts vs the ImageNet diffusion prior (2-forward and S_mid). PC-DAPS-style labels plus PSNR thresholds, all pre-registered | Does the anatomy change where symmetry errors are common (PC-DAPS: 42% of DAPS runs flipped)? Does a generic test still suffice? | ≈4 GPU-h FAST, +≈9 GPU-h with the DAPS subset | **High for generality**; reviewers' first question | Data access (`/data/imagenet` contents unknown; DAPS's 100-image test set may need downloading); low ImageNet PSNRs (FAST-DIPS reports 19.7 dB best-of-4) |
| **X5** | **PC-DAPS as a fifth solver** on B22 (bag of 4, released configuration), audited with our labels, plus GT-free selection among its particles | Does in-sampler pruning change the anatomy? Does post-hoc GT-free selection on plain DAPS runs reach PC-DAPS? | ≈2–4 GPU-h + environment work | **Medium**: currency against the closest work | Reproduction variance (its README warns about PRNG/GPU sensitivity) |
| **X6** | **Operator robustness with FAST**: oversampling 1.5 / 4, σ ∈ {0.01, 0.1}, on B22 | Is the anatomy specific to the 2× / 0.05 setting? | ≈1.5 GPU-h; **new measurements** | Medium; better suited to a journal version | Pre-registration of thresholds per setting |

**Not recommended for this paper:**
- **Early-abort / restart policies** using the signed residual. This reopens the method line and needs D4's reopening procedure.
- Latent-diffusion solvers.
- Real optical data.
- The T1 theory note.

**Recommendation:**
- **X1 + X2 now** (CPU, inside R0-N.2).
- **X3 very likely**: cheap, and it carries the practical message.
- **Then one of X4 (generality) or X5 (currency).** I lean X4: reviewers ask "only FFHQ?" more often than "did you run the latest preprint?", and the user's norm already lets us cite PC-DAPS without running it.
- **X6** belongs in any later journal version, e.g. IEEE OJSP, where Crafts & Villa (2025) published a DAPS-family benchmark.

## 3. Page budget

Planning assumption: 5 pages including references.

| Block | Space |
|---|---|
| Introduction, setup, protocol table (X1, literature part) | 1.3 pages |
| Fig. 1: anatomy | 0.4 page |
| Table 1: ladder with X1 inflation columns | 0.5 page |
| Fig. 2: K-scaling / compute frontier (X3) | 0.5 page |
| Table 2: orientation tests | 0.3 page |
| Generality (X4 or X6) | 0.4 page |
| Discussion and limitations | 0.4 page |
| References | 0.8 page |

The X2 hardness results get one paragraph only if they are crisp.

## 4. Process, dates, guardrails

**Dates:**

| Date | Step |
|---|---|
| Now | R0-N.2 revision C (CPU, including §A7 observation export); research pass launched |
| By 2026-11-07 | Planner reads both, then writes `X_PREREG_v1` for the chosen GPU extension(s), with frozen seeds, labels, thresholds, prompts and stop rules. Cheng: go / no-go |
| November – early December | At most **two** GPU blocks, each one launch command, run by Cheng |
| 2026-12-11 | Draft ready (K4) |
| 2027-02-06 | EUSIPCO |

**Guardrails:**
- Every extension is **evaluation-only**: no new method, pre-registered before the data exist, reported whatever the outcome.
- Any extension not returned by its date is dropped.
- No new FFHQ ids: X3 reuses the 180 B22 + DEV80 ids. confirmation305 stays locked.
- **Burnout guard:** if Cheng's time is short, X1 + X2 alone still improve the paper, and nothing else is needed.

## 5. Theory (Cheng's question, 2026-10-24 evening) — planner's preliminary view, pending the research pass

Two different things go by "theory":

**P1 — a probabilistic model of the protocol.** This is in scope for the EUSIPCO paper.
- **The model:**
  - Image i yields a GOOD run with probability p_i.
  - It yields a measurement-consistent wrong run with probability q_i.
  - Otherwise it yields an inconsistent failure.
- **The selection rule:** GT-free selection ranks inconsistent failures last. Consistent failures are exchangeable with GOOD runs under the residual.
- **What follows:**
  - the oracle success curve 1 − E(1 − p)^K;
  - the GT-free curve and the oracle gap, as closed forms in (p_i, q_i);
  - saturation from intrinsically hard images (p_i ≈ 0).
- **Testing:** fit it on K = 1, predict K = 2/4/8, check it against the ladder, and use it to read X3.
- **An analogy to check:** in LLM sampling, GT best-of-K is pass@k and GT-free selection is an imperfect verifier. SITCOM's consistent-wrong runs play the role of verifier false positives.
- **Cost:** CPU only; it fits X2/X3.

**T2 — algorithm-internal theory.** This is a separate study and is not recommended for this paper.
- **The view:** read DAPS, FAST-DIPS and SITCOM as different solvers of one inner problem: sampling (DAPS), projection (FAST-DIPS), optimization through the network (SITCOM).
- **The method:** analyze them in toy models with exact symmetry and a known prior, together with speciation and symmetry-breaking results for diffusion. The aim is to predict orbit errors (early commitment) and consistent-wrong outputs (strong measurement fitting on an ill-conditioned problem).
- **Cost and status:** months of work and idealized assumptions. It needs D4's reopening procedure.

The research pass will check what already exists for both. The planner will rule after reading it.
