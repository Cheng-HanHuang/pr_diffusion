# PC-DAPS and field positioning — planner note (2026-10-24)

> **Update, 2026-10-24 morning: K2 triggered** (decision §10).
>
> **Result.** CLIP zero-shot fixes 84/84 global orbit errors with 0/2,040 good outputs harmed, matching the diffusion prior at lower cost. So the diffusion-prior repair is **no longer a contribution**.
>
> **Superseded in this note:**
> - every "still ours" item about repair (§0 item 3, §2's prior column, §6 claim 3);
> - the related-work sentence in §7. Its replacement:
>
> > Concurrent work [PC-DAPS] also observes that ground-truth best-of-k reporting masks rotated and color-inverted phase-retrieval reconstructions, and modifies DAPS to prune them during sampling. Our study is complementary: a post-hoc audit of four solver families showing that ground-truth-free selection with a generic orientation test recovers nearly all of the ground-truth best-of-K gain, and that the remaining failures are generation failures.
>
> **Still valid:** the reconciliation (§3), the field survey (§5), and the anatomy and ladder claims.
>
> **Withdrawn:** N4 (§9).
>
> The current plan is LETTER_PLAN v1.3.

**Author:** planner (Claude).

**Trigger:** Cheng attached arXiv 2610.31592 (PC-DAPS, posted 22 Oct 2026). He asked for:
- a full read;
- what other work in this setting teaches us;
- the gap we fit.

He noted that preprints can informally be ignored in our paper, but that this one means we should submit fast.

**Effect on binding documents:**
- **`LETTER_PLAN.md` → v1.2:** title, pitch, claim order, must-not-claim list, timeline.
- **New executor amendment:** [`R0N_AMENDMENT_2_20261024.md`](../../track_r/executors/r0n/R0N_AMENDMENT_2_20261024.md), the R0-N.2 follow-up block, CPU only.
  - Required: N1h and A3 identities (owed from amendment 1), N2 on CPU, and the N1d miss identity.
  - Optional: the N1i crosswalk and protocol ladder, and the N3b bounded inventory.
- **Proposed, not authorized:** N4, a clean-image orientation probe (≤0.75 GPU-hours, §9). Needs Cheng's explicit go.
- **Unchanged:** R0 v1, R0-N v1 definitions, decisions D1, D2 and D4, and kill criteria K1–K4.

**Evidence labels:**
- **[FULL]** read in full.
- **[CODE]** checked in the official code.
- **[MAIN]** main-text passages read.
- **[ABS]** abstract or metadata only.
- **[2nd]** secondhand.
- **[REPO]** our own audited records.

---

## 0. Planner's view for Cheng

### How I view the current results

**Our numbers are unchanged; how much of the framing is still ours has changed.**

**Pre-empted: "GT best-of-k hides failures."**
- PC-DAPS is titled *Beyond Best-of-k*.
- It states that the protocol masks rotated and color-inverted reconstructions.
- It reports per-run outcome frequencies and a mean-of-k table.
- Our working title "What best-of-K hides…" must go.

**Partly pre-empted: using the solver's own denoiser to reject rotated faces without GT.**
- PC-DAPS does this inside DAPS by pruning particle clusters.
- Its released code also contains an exact 180° twin test with paired noise. That is essentially our global-only repair, applied mid-sampling rather than to final outputs. It runs only for ImageNet and is not described in the paper (§1.3).

**Still ours, and now the lead:**
1. **A cross-family anatomy under GT-free selection.**
   - Repairable orbit errors are 7.1% of failures pooled (FAST 3.6%).
   - SITCOM's failures fit y in 57% of cases, so no residual test can see them.
2. **A post-hoc GT-free protocol** (signed-residual selection + exact-orbit repair) that matches the field's GT best-of-K.
   - Going from the first run to GT best-of-K gains 2.2–5.5 dB.
   - The GT-free protocol recovers 93–129% of that gain in every family (§3.2).
   - What remains is generation failure.
3. **Solver-agnostic repair with harm accounting:** 0/2,040 good outputs harmed, versus 220 for UnivFD, with a cost curve.
4. **The complexity explanation** for why the same score is a near-perfect orbit selector but a mediocre cross-image failure detector.

**The R0-N N1 return (same day) strengthens the parts that are ours** (rulings in decision §9):
- K3 passes and K1 is not triggered.
- The pre-registered cheapest repair is **2 UNet evaluations per candidate**: identity vs global 180° at t = 300, one paired draw. It scores 49/49 with 0/1,423 harm on held-out B22. PC-DAPS's code-path twin test uses 24 evaluations per particle.
- After GT-free selection plus repair, **93 of the 99 still-failing image-strata are generation failures**.

**The numbers reconcile in principle.**
- PC-DAPS counts 14.5% of DAPS runs on FFHQ as upside-down/color-inverted.
- We count 3.5% (B22) and 5.3% (DEV80) of DAPS runs as *repairable* orbit errors.
- Their label has no quality threshold: a 15 dB rotated output counts. Ours requires an orbit member ≥ 25 dB.
- The configurations and images also differ. N1i measures the label part directly (§3.1).

**Adversarial summary.**
- The letter is still worth writing, as an evaluation letter that uses repair as a tool, not as a repair-method letter.
- SPL odds: about 25–35%, down from 30–40%.

### My plan

1. **Reposition now:** this note, plus LETTER_PLAN v1.2 with a new title, claim order and must-not-claim list.
2. **One more CPU block, R0-N.2** (amendment 2).
   - Required:
     - N1h (membership) and A3 identities, owed from amendment 1;
     - N2 on CPU, for K2, because the GPU was unavailable.
   - Optional:
     - PC-DAPS's outcome label computed on our 3,240 runs;
     - the protocol ladder (mean-of-K → E0 → E1 → O_raw → O_orbit);
     - an exploratory prior-only selection policy.
3. **Submit faster.**
   - SPL submission target: 2026-11-24.
   - K4 (2026-12-11) stays the hard stop.
   - Starting now, I draft the parts that don't depend on R0-N.
4. **No PC-DAPS head-to-head before submission.** It is a revision-stage option; its code is public under MIT.
5. **Optional N4, needs your go (≤0.75 GPU-hours).** A clean-image probe on held-out ImageNet images, plus held-out faces if PAC has any. It is the cheapest way to answer two reviewer questions:
   - "Does this work only because the prior memorized FFHQ?"
   - "What about ImageNet, where PC-DAPS is weak?"

### Publication image (current)

**Title candidates** (LETTER_PLAN v1.2 §0):
- A: "Failure Anatomy and Exact-Symmetry Repair for Diffusion-Prior Phase Retrieval";
- B: "Ground-Truth-Free Selection for Diffusion Phase Retrieval: Symmetry Repair and What Remains".

**Venue:** IEEE SPL (4 + 1 pages).
- Fallback EUSIPCO 2027, paper deadline 6 Feb 2027.
- Fallback ICIP 2027, 31 Mar 2027 per the SPS event page (verify).

**Expected content:**
1. Anatomy across 4 families × 2 cohorts, crosswalked to PC-DAPS's labels.
2. Proposition 1 (posterior ratio = prior ratio on the exact orbit) with the paired estimator and cost curve.
3. The protocol ladder showing GT-free parity with GT best-of-K.
4. Repair versus UnivFD versus CLIP, with harm.
5. The complexity mechanism.
6. Recommendations and limitations: FFHQ only, σ = 0.05, 2× oversampling, checkpoint membership, ImageNet.

### Decisions for Cheng (the default applies if you don't reply)

| # | Decision | My recommendation | Default |
|---|---|---|---|
| 1 | Mention PC-DAPS in the letter? | Yes: one concurrent-work sentence (§7) | cite |
| 2 | Run N4 (≤0.75 GPU-h)? | Yes if N3/N3b find the assets; otherwise no | not run |
| 3 | Title | A | A |
| 4 | Post our own arXiv preprint at submission | Yes, if coauthors agree | ask coauthors |

---

## 1. PC-DAPS, read in full [FULL] [CODE]

### 1.1 Identity

- **Paper:** Kocabas*, Burns*, Geiger, Gauen, Fridovich-Keil, "Beyond Best-of-k: Prior-Consistent DAPS for Ambiguous Nonlinear Inverse Problems," arXiv 2610.31592v1, 22 Oct 2026.
- **Affiliations:** Tübingen / MPI-IS / Georgia Tech. ICLR-style preprint; venue not stated.
- **Code:** github.com/voilalab/PC-DAPS. MIT license; read at commit `59b2f04`. It builds directly on the DAPS code.

### 1.2 Setting, which is ours

- **Operator:** DAPS phase retrieval (oversampling 2.0, β_y = 0.05, the same operator definition as ours).
- **Checkpoints:** DAPS's FFHQ DDPM (`ffhq_10m.pt` lineage) and ImageNet DDPM.
- **Test sets:** DAPS's 100-image FFHQ and ImageNet test subsets (`start_id 0`, `end_id 100`).
  - Our research pass found that the field's "first 100" FFHQ images are in FFHQ's training category [2nd].
- **Main baseline:** DAPS-4K (4,000 NFE per run) as GT best-of-4. The other baselines (DPS, PnP-DM, DPnP, RED-Diff, HIO) are also GT best-of-4.

### 1.3 Method

**Paper (§4, App. A.1/A.4):**
- **Bag.** K = 4 particles annealed in one DAPS run.
- **Denoiser posterior.** DAPS's Gaussian is replaced by a Gaussian mixture built from the bag (Eqs. 10–13).
- **One-shot consensus pruning.**
  - **Trigger:** fires when r̄ = mean_k ‖A(x_k) − y‖² < r⋆ = ‖A(x̄) − y‖². This is a Jensen-gap test for "the bag sits in several likelihood wells".
  - **Clustering:** must return M > 1 clusters. Particles are downsampled to 32², projected onto PCA components explaining 90% of the variance, and clustered with a GMM chosen by BIC.
  - **Score:** each cluster is scored by Eq. 9, s(C) = −mean ‖x − D(x + σε; σ)‖² over σ ∈ S = {0.5, 1, 2} (EDM scale).
  - **Pruning:** clusters scoring below (1 + γ)·max are replaced by copies of survivors. γ is the 95th percentile of within-cluster score gaps on 10 calibration images.
- **Other changes.** Adaptive ODE (1 Euler step for the first 65% of annealing, then 5), and the final Langevin is skipped.
- **Cost.** PC-DAPS-1K costs 732 NFE per particle (Table 5), about 2,928 per image for K = 4. DAPS-4K best-of-4 costs 16,000.

**Code facts not in the paper** (`sampler.py`, `cores/pcdaps.py`, `posterior_sample.py` @ `59b2f04`):
- **Score noise.** Eq. 9 uses one noise draw per σ, **shared by all particles of a bag** ("common random numbers"). Squared errors are averaged over pixels.
- **ImageNet twin test.** For phase retrieval on ImageNet only (`posterior_sample.py` L220–222 sets `sampler.orient = True` when the data root contains "imagenet"), an **exact global 180° twin test** runs once, when the annealing σ ≤ 0.3:
  1. `prior_consistency(..., num_samples=4, twin=True)` scores each particle's PF-ODE estimate and its 180° rotation;
  2. the twin's noisy input is the rotated noisy input (paired noise, R ε);
  3. the post-Langevin particle is flipped if the twin scores higher.

  This is global only (no per-channel members), runs inside the sampler, and is off for FFHQ.
- **Released pruning is two-stage.** A Tweedie-residual stage (γ = 0.15) is checked every step. A noisy-residual stage (γ = 0.065) is active only when σ ≤ 1. This is slightly more elaborate than the paper's one-shot description.
- **Outcome labels** come from `pr_outcomes.py`; the exact rule is in amendment 2, A4.1.

### 1.4 Results relevant to us

**Main metrics (PSNR / SSIM / LPIPS):**

| Phase retrieval | FFHQ | ImageNet |
|---|---|---|
| PC-DAPS, mean of its 4 particles (Table 1) | 30.09±1.77 / 0.834 / 0.152 | 21.75±6.10 / 0.469 / 0.373 |
| DAPS-4K, GT best-of-4 (Table 1) | 29.66±4.17 / 0.770 / 0.177 | 20.98±6.72 / 0.461 / 0.387 |
| DAPS, mean-of-4 (Table 4) | 25.11±5.73 / 0.666 / 0.283 | 15.70±3.90 / 0.317 / 0.516 |

**Outcomes (Table 2), DAPS 4 runs → PC-DAPS bag of 4:**

| Outcome | FFHQ | ImageNet |
|---|---|---|
| Faithful | 79.5% → 97.5% | 38.5% → 45.0% |
| Upside-down / color-inverted | 14.5% → 1.5% | 42.0% → 30.5% |
| Failed | 6.0% → 1.0% | 19.5% → 24.5% |
| Unrecoverable images | 4% → 1% | 25% → 15% |

**Other results:**
- More particles help, saturating at K = 8–16 (Fig. 7).
- Ablations (Table 3) use 10 images only.

### 1.5 Their stated limitations

- On ImageNet, "the multi-class ImageNet prior provides weaker disambiguating signals."
- Even with the twin test enabled there, 30.5% of ImageNet particles stay flipped, and failures rise from 19.5% to 24.5%.
- The discussion invites "future work to further explore our proposed paradigm of leveraging the prior score to assess candidate reconstructions at inference time." Our letter is that exploration, post hoc and across solvers.

## 2. Side by side

| | PC-DAPS | Our letter |
|---|---|---|
| Where it acts | Inside DAPS (one run, K particles) | Post hoc, on any solver's final outputs |
| Solvers studied | DAPS (others as GT best-of-4 baselines) | DAPS, SITCOM, FAST-DIPS, NP |
| Prior signal | Cluster-mean denoising residual at σ ∈ {0.5, 1, 2} ≈ DDPM t ≈ 144 / 258 / 396; 1 draw. ImageNet code path: global twin, 4 draws | Paired-noise ε-MSE over the exact 8-member per-channel orbit. S: t ∈ {50…700}; S_mid: t ∈ {300…600}. Any single t ≥ 200 suffices (R0 per-t) |
| What is compared | Different particles (clusters) for the same y | Exact orbit members of one candidate: identical likelihood, identical rotation-invariant statistics |
| Trigger and cost | Jensen-gap test plus BIC clustering; ImageNet twin test = 2 members × 3 σ × 4 draws = 24 evaluations per particle | Always. Dev-selected configuration: 2 evaluations per candidate (t = 300, 1 paired draw, identity vs global). Full 8-member S: 256 |
| Harm accounting | None reported | 0/2,040 good outputs harmed; UnivFD harms 220 |
| Labels | Orientation only (faithful / flipped / failed), no quality threshold | Quality-aware: GOOD ≥ 25 dB / ORBIT / CONSISTENT / INCONSISTENT |
| Protocol compared | PC-DAPS mean-of-4 vs baselines' GT best-of-4 and mean-of-4 | First run, mean-of-K ⟨N1i⟩, E0/E1/E2 (GT-free) vs O_raw/O_orbit (GT) |
| Images | DAPS 100-image test sets (FFHQ: likely training category) | FFHQ B22 (random, validation category) + DEV80 (hard-enriched) |
| Datasets | FFHQ + ImageNet | FFHQ only (ImageNet: N4 proposal) |

## 3. Reconciling the numbers

### 3.1 DAPS orientation-error rates

**Per-run rates [REPO, R0.7 P1]:**
- DAPS on B22 (200 runs, K = 2): GOOD 82.5%; repairable ORBIT 3.5%; measurement-inconsistent 14.0%; measurement-consistent 0%.
- DAPS on DEV80 (320 runs, K = 4): GOOD 53.1%; ORBIT 5.3%; consistent 5.6%; inconsistent 35.9%.

**PC-DAPS, DAPS on its FFHQ images:** faithful 79.5%, flipped 14.5%, failed 6.0%.

**Three differences, in likely order of size:**
1. **Labels.**
   - PC-DAPS calls a run "flipped" if every channel is > 1 dB closer to some orientation, at any quality. Our ORBIT requires an orbit member ≥ 25 dB.
   - A misoriented 15 dB run is "flipped" for them and a non-orbit failure for us.
   - Our B22 DAPS non-orbit failures (14%) are enough to cover the gap if many of them are misoriented.
2. **Configuration.**
   - Theirs is DAPS-4K: 10 ODE × 400 annealing steps, 4,000 NFE.
   - The planner reads our historical B21/B24 launch defaults as 5 ODE × 400 annealing steps (2,000 NFE). N1i confirms this from receipts.
3. **Images.** Their test subset versus B22/DEV80.

N1i applies their exact rule to our 3,240 runs and cross-tabulates it against our labels. Expected outcome (planner hypothesis, not a claim): most of their "flipped" excess is misoriented generation failure. That supports our "what remains is generation failure".

### 3.2 GT best-of-K is mostly recoverable without GT [REPO, R0.8 P4; planner arithmetic, to be reproduced by N1i]

| Stratum (max K) | First run (K=1) | GT best-of-K (O_raw) | GT-free E1 | Gain O_raw − first run | E1 − O_raw | Share of the gain E1 recovers |
|---|---|---|---|---|---|---|
| B22 DAPS, K=2 | 27.23 dB (80%) | 29.42 (92%) | 30.07 (95%) | +2.19 | +0.65 | 129% |
| B22 SITCOM, K=4 | 22.97 (71%) | 26.66 (95%) | 26.64 (94%) | +3.69 | −0.02 | 100% |
| B22 NP, K=8 | 25.58 (75%) | 29.40 (96%) | 29.12 (94%) | +3.82 | −0.28 | 93% |
| DEV80 DAPS, K=4 | 23.29 (59%) | 28.83 (85%) | 28.82 (86%) | +5.54 | −0.00 | 100% |
| DEV80 SITCOM, K=4 | 19.59 (43%) | 24.12 (69%) | 24.70 (73%) | +4.53 | +0.58 | 113% |
| DEV80 NP, K=4 | 22.32 (51%) | 26.07 (71%) | 25.86 (70%) | +3.75 | −0.21 | 94% |
| FAST (180 images), K=4 | 23.81 (57%) | 29.00 (86%) | 28.99 (86%) | +5.19 | −0.01 | 100% |

Percentages in parentheses are Good25.

**Meaning:**
- The gain from running K times is real (2.2–5.5 dB).
- Almost none of it depends on GT. A residual pick plus ≤ 32 forwards of repair recovers it.
- PC-DAPS presents GT best-of-k as a protocol that "uses information unavailable at test time". Our data say the information it uses is mostly recoverable.
- The honest cost of best-of-K is K× compute, not GT.

This is the cleanest one-line message we own.

"First run" is a single realization. The mean-of-K row from N1i replaces it in the letter.

## 4. What we learn and can use

1. **Their label as a crosswalk** (N1i). Readers will compare to PC-DAPS Table 2, so report our runs in their taxonomy as well as ours.
2. **Mean-of-K as a ladder rung** (N1i). It makes the "first run → GT best-of-K → GT-free" ladder honest.
3. **Their σ set corroborates our per-t finding.**
   - σ ∈ {0.5, 1, 2} maps to DDPM t ≈ 144 / 258 / 396 (linear β schedule; planner computation).
   - That is inside or next to our working range (t ≥ 200 gives 100%; t = 100 gives 80–86%; t = 50 fails).
   - Both avoid low noise, where the CCLR paper (Goodier & Campbell, arXiv 2310.17432) finds the complexity bias concentrates.
4. **Common random numbers are their design too.** This supports our paired estimator as the natural choice. We claim its validation, not its invention.
5. **ImageNet is the stress test.**
   - 42% of DAPS runs are flipped on ImageNet, and in-sampler pruning plus a global twin test leaves 30.5%.
   - This is where a post-hoc 8-member test could be "better in some metric", or could fail for the same reason. N4 (clean-image probe) is the cheap first look.
6. **Premature mode selection** (their §3.2 and toy in Fig. 2) and Burns & Fridovich-Keil's finite-sample analysis give a mechanism story for why orbit and consistent-but-wrong modes arise. Cite it; don't re-derive it.
7. **Their Jensen-gap trigger** (r̄ < r⋆) is a GT-free "the K outputs disagree in a non-trivial way" test. We don't need it, because the repair is cheap enough to always run. It could gate repair at scale; mention at most one clause.
8. **Their reported DAPS numbers** (29.66 GT best-of-4 versus 25.11 mean-of-4) quantify the same ladder on their images. Quote them only as their numbers.

## 5. How other recent work studies this setting

| Work | What it does in this setting | What we take | Relation |
|---|---|---|---|
| DPS, DAPS, SITCOM, FAST-DIPS (2023–2026) | GT best-of-4 reporting | The protocol we measure | Baseline practice |
| DiffFPR (ICML 2024) [CODE] | Enumerates the 8-member orbit and aligns to GT | Orbit enumeration is not ours | Prior art for enumeration |
| DDRM-PR, I2I-PR (2025–2026) [MAIN per research pass] | GT-based conjugate-inversion correction in evaluation | Evidence the field corrects symmetry with GT | Motivation |
| FA-FPR (ISCAS 2025) [FULL] | GT-free measurement-match selection; UnivFD-triggered restarts | Residual selection is classical; UnivFD is our negative baseline | Direct predecessor |
| deepECpr (arXiv 2407.09687) [2nd snippet] | Aligns HIO channels to channel 1 by correlation: GT-free, heuristic | Prior art for per-channel alignment without GT | Cite with the 8-member orbit |
| TReg (ICLR 2025) [MAIN] | Says image statistics alone may not break the symmetry; uses text | Both PC-DAPS (FFHQ) and our R0 contradict it for face priors | Claim we refute (FFHQ only) |
| PC-DAPS (arXiv 2610.31592) [FULL][CODE] | In-sampler prior pruning; outcome frequencies; mean-of-k; ImageNet twin test in code | Label, ladder, σ choice, ImageNet stress test | Concurrent and closest |
| DAPS++ (arXiv 2511.17038; ECCV 2026 per PC-DAPS refs) [MAIN, App. S2.2] | Phase-retrieval failure rate = PSNR < 25 on the "top 100" FFHQ test images. Attributes failures to "the nonlinear frequency-domain measurement operator, which interacts unfavorably with the pixel-domain initialization." Rotations not discussed | Our Good25 threshold matches field practice; cite for it | Supports anatomy framing |
| Burns & Fridovich-Keil (arXiv 2605.30330) [ABS + main-text excerpt + README] | Finite-sample reference shows DPS/ΠGDM/TMPD hallucinate "likelihood modes not supported by the prior". Synthetic priors only; no phase retrieval; no real-image test | Mechanism citation for orbit and consistent-but-wrong modes | Theory background |
| RPU (Jin et al., arXiv 2606.02331) [ABS] | Robust prior update inside DPS; inpainting and deblurring; human faithfulness study | Not phase retrieval | Hallucination context only |
| TGD / A-TGD (Makris, Fearnhead, Nemeth, arXiv 2605.03712; NeurIPS 2026 poster page) [ABS] | Annealed SMC with likelihood reweighting; A-TGD prunes to one high-likelihood trajectory | In-sampler likelihood selection; phase-retrieval numbers seen only secondhand | Our E0 is its zero-cost post-hoc analogue |
| SCG (ICML 2024), CPS (AAAI 2026), FPS-SMC, TDS | Particle or per-step selection by measurement fit | Selection inside samplers is common | Context |
| Serrà et al. (ICLR 2020); Nalisnick et al. (ICLR 2019); CCLR (arXiv 2310.17432) [SNIP/ABS] | Likelihoods track image complexity; for DDPMs the bias concentrates at low noise | Our mechanism, and why t = 50 fails | Mechanism prior art. "Cancels on the exact orbit" not found in prior work. |
| DiME (ICLR 2026); score-KSD; PosteriorBench; InverseBench; OJSP 2025 benchmark [ABS] | Evaluation of priors and posteriors, not per-sample phase-retrieval selection | Shows evaluation is an active topic | Context |
| FFHQ README; ILVR App. C.2 [primary per research pass] | First 60k images are training; `ffhq_10m.pt` likely trained on all 70k (inferred) | Memorization limitation; motivates N4 | Validity |

**Not found in this pass or the 2026-10-24 research pass:**
- a cross-solver failure anatomy;
- the residual's AUROC as a phase-retrieval failure detector;
- measurement-consistent-but-wrong rates per solver;
- a post-hoc, solver-agnostic orbit repair evaluated with harm.

## 6. The gap we fit

**One sentence:** no paper quantifies, across diffusion phase-retrieval solvers, what is left after a GT-free selection and symmetry repair, or validates a post-hoc symmetry repair with harm accounting. PC-DAPS changes one sampler and reports orientation-only outcomes.

**Revised claim order** (LETTER_PLAN v1.2 §2):
1. anatomy;
2. GT-free parity with GT best-of-K (the ladder);
3. post-hoc, harm-accounted exact-orbit repair;
4. complexity mechanism;
5. negative findings on realness detectors and the clipped residual.

**New must-not-claims** (LETTER_PLAN v1.2 §3):
- first to show best-of-k hides orientation errors;
- first GT-free prior-based orientation fix;
- novelty of the paired twin test;
- "orbit errors are rare" without "repairable";
- anything about PC-DAPS's cluster score being confounded (untested);
- head-to-head numbers against PC-DAPS.

## 7. Cite or ignore PC-DAPS?

Cheng's point stands: a preprint posted about a month before our submission is concurrent work, and we owe no comparison. I still recommend a single concurrent-work sentence:
1. **It shares our setup.** Same task, operator and checkpoint lineage, and overlapping framing. Reviewers in this area are likely to know it; Fridovich-Keil's group works on imaging inverse problems.
2. **Its code contains the twin test.** Silence would read as missing or hiding it.
3. **Citing costs nothing we need.** We give up "first to critique best-of-k", which we never needed. We gain a frame: their numbers become one more case our anatomy explains.

**Draft sentence (related work):**

> Concurrent work [PC-DAPS] also observes that ground-truth best-of-k reporting masks rotated and color-inverted phase-retrieval reconstructions, and modifies DAPS to prune such particles during sampling with a denoising-residual prior score; its released implementation additionally applies a global 180° twin comparison on ImageNet. Our study is complementary: it is post hoc and applies to the final outputs of any solver, compares only exact symmetry-orbit members of one candidate (identical likelihood, so the prior ratio is the posterior ratio), measures harm on correct outputs, and shows across four solver families that once selection is ground-truth-free and symmetry errors are repaired, the remaining failures are generation failures.

**Draft framing sentence (introduction):**

> Diffusion-prior phase retrieval is reported as the best of four runs selected with the ground truth. We ask how much of that number a deployable, ground-truth-free procedure recovers, and what kind of failure remains.

## 8. Speed and submission

- **Target.** SPL submission on 2026-11-24. K4 (2026-12-11) is unchanged as the hard stop. Full timeline in LETTER_PLAN v1.2 §6.
- **Parallel drafting.** The planner drafts the introduction, setup, method and related work now. Results wait for R0-N.
- **Drop rule.** Any optional item (N1i, N2, N4) not returned by 2026-11-06 is dropped from the submission.
- **Preprint at submission.** IEEE permits posting the submitted version on arXiv, so this is possible if coauthors agree. It timestamps our claims; it is Cheng's and the coauthors' call.
- **Scoop watch (W1).** Before submission, check arXiv weekly for a PC-DAPS revision and for "diffusion" + "phase retrieval" + "symmetry/orientation" work. If someone publishes post-hoc exact-orbit repair with cross-solver evaluation first, the planner re-rules. The anatomy and the ladder still carry the letter.

## 9. Experiments: add, propose, defer

**Added (authorized, CPU): amendment 2, the R0-N.2 block.**
- **Required:** N1h, A3 identities, N2 on CPU, and the N1d miss identity.
- **Optional, non-blocking:**
  - **N1i:** PC-DAPS outcome rule on our runs; cross-tab; ladder with mean-of-K; prior-only selection E_P (exploratory); DAPS configuration record.
  - **N3b:** bounded inventory of held-out face data and of the ImageNet validation layout.

**Proposed (needs Cheng's go): N4, a clean-image orientation probe, ≤0.75 GPU-hours, one block run by Cheng.**
- **Question.** On clean upright images the solver's prior has *not* seen, does the frozen paired test pick the identity orbit member?
- **N4a (memorization).** FFHQ checkpoint on up to 100 held-out face images from a non-FFHQ dataset, if N3b finds one. Control: the 100 B22 GT images. Do not draw new FFHQ ids; confirmation305 stays untouched.
- **N4b (ImageNet).** ImageNet DDPM checkpoint (from N3) on the 100 DAPS ImageNet test images, or the first 100 ImageNet validation images selected by fixed file name (no directory enumeration).
- **Score.**
  - The frozen S and S_mid, plus the N1d-selected cheap configuration (`single_300|J1|global2`), with paired R_s ε noise.
  - Metric: identity-win rate over the 8 members and global-only (000 vs 111), with Wilson CIs.
- **Pre-registered reading:**
  - **N4b:** identity-win ≥ 0.95 → one Discussion sentence ("separates orientation on held-out ImageNet images at X%"); otherwise a limitation consistent with PC-DAPS.
  - **N4a:** within 3 points of the control and ≥ 0.95 → "no evidence repair relies on memorization"; otherwise state the gap as a limitation.
- **Not a claim about ImageNet reconstructions.**
- **Cost estimate.**
  - ImageNet: about 100 × 256 forwards. The ImageNet model is roughly 6× the FFHQ model's compute per forward, so ≈ 25–30 min.
  - FFHQ: ≤ 200 × 256 forwards, ≈ 9 min at 2.755 s per candidate.
  - Total ≈ 0.6 GPU-hours. Stop if the projection exceeds 0.75 GPU-hours.
- **Why now rather than after the draft (D3):** PC-DAPS made ImageNet the obvious reviewer question, and N4 is cheaper than the FAST-only ImageNet probe D3 anticipated.

**Deferred: a PC-DAPS head-to-head on B22/DEV80.**
- Cost: about 2 GPU-hours plus environment work.
- Reasons to defer: speed, and Cheng's norm on preprints. It is the natural experiment if a reviewer asks: run their released configuration with K = 4 and apply our labels and post-hoc repair to its particles.

## 10. Sources

**Papers:**
- PC-DAPS: arXiv 2610.31592 (full PDF read from Cheng's upload); code https://github.com/voilalab/PC-DAPS @ `59b2f04`.
- DAPS++: arXiv 2511.17038, App. S2.2.
- Burns & Fridovich-Keil: https://arxiv.org/abs/2605.30330; code https://github.com/voilalab/diagnosing-posterior-sampling.
- RPU: https://arxiv.org/abs/2606.02331.
- TGD / A-TGD: https://arxiv.org/abs/2605.03712.

**Research reports and repo records:**
- Earlier planner research pass "Diffusion-Prior Fourier Phase Retrieval: SOTA, Evaluation Practice, and Novelty Threats" (2026-10-24): FFHQ split, ILVR provenance, venues, deepECpr, CCLR, DAPS, SOCS.
- R0 numbers: `R0_8_PLANNER_RETURN.zip` (SHA-256 `13c008f3…efab`): R0.7 P1 and the R0.8 P4 matrix.
