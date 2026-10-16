# TRACK_B_PLANNER_RETURN — Basin-aware diffusion phase retrieval

**Date:** 2026-10-15  
**Owner:** Track B autonomous scientific research lead (planner; no PAC access)  
**Planner branch:** \`codex/prdiff-track-b-basin-planner-20261015\`  
**Read-only inspection base:** \`1793b67f9990842f57ac2e948921453e2a61436e\` (the branch later acquired the metaplanner autonomy amendment)  
**Status:** SCIENTIFIC PLAN; no new solver implementation, FFHQ inference, measurement generation or PAC run performed as part of this return.  
**Scientific decision:** B1 synthetic falsification first; B2 possible fallback; B3 deferred. Not a validated method or novelty result.

## 1. The scientific question and non-negotiable boundaries

Can a diffusion phase-retrieval trajectory *discover and preserve* prior-compatible reconstruction basins when visually wrong alternatives fit Fourier magnitudes, without an oracle selector among independent restarts? The distinction is basin **generation** rather than retrospectively selecting the best of existing candidates. Exact Fourier ambiguities must not be described as errors recoverable from the observation alone. The runtime cannot read GT, optimize PSNR, or pick orientation using the true reconstruction.

The shared charter, autonomy amendment and live coordination issue #41 apply. Historical datasets are diagnostic exposures, not fresh confirmation. B24 confirmation305 remains locked. Track A owns NP publication/selector fairness; B26 completed STOP and is a read-only scientific dependency. Track B owns new basin-transition operators, toy validation, and separately qualified native implementations.

## 2. Complete historical evidence: positives, negatives, corrections

| Evidence | Frozen result | Consequence |
|---|---|---|
| B19.20 and subsequent B21.0 correction | 100-image diagnostic panel originally described as 1000 image×measurement rows; most saved metric groups degenerate across seeds | DO NOT treat 1000 rows as independent; candidate-generation and detector-failure taxonomy remains diagnostic |
| B20.11 hard image / B20.12A held-out hard images | LF guidance changes basin coverage: held-out base 119/256 and union with LF 153/256 seed-cases | Structured intervention can alter basin entry, but oracle union is not executable best-of-one and costs must be accounted |
| B21.7 matched comparison | Fresh2 70/80 Good25 vs base+LF 63/80; both approximately two DAPS runs | Old LF intervention not the default second arm |
| B21.9 disjoint validation | Fresh3 75/80 vs Fresh2 74/80 | Blind extra restarts not a promising mechanism |
| B21.11 prospective Fresh2 | 92/100 executable Good25; 8/100 lacked a good candidate within two DAPS runs | Candidate-generation bottleneck, not merely a poor final selector |
| B21.12 taxonomy | Of eight raw failures, three oracle 180-degree recoveries and five persistent under identity/reversal; latter include chromatic, twin/ghost, complex-prior collapse | Separate symmetry, ghosts and low-quality floors; offline orientation oracle forbidden at runtime |
| B22 fixed 100-image baseline | Fresh2 mean 29.299, q05 15.860, Good25 92/100, 312.6 GPU-s/image; SITCOM-4S mean 26.442, q05 23.415, Good25 93/100, 196.1; NP-8-RS mean 29.269, q05 25.146, Good25 95/100, 411.8 | No equal-budget dominance; 99/100 cross-family GT oracle is NOT deployable |
| B22 shared failure | Image 65003 fails Fresh2, SITCOM4S, NP8RS; structured ghost/photometric failures differ by parent | Need new basin generation without fitting retrospective hard images |
| B23.1 adapter audit | Four native replays PASS, **zero NP/SITCOM cross-family adapter donors qualified** | Conceptual operator needs distinct native insertion, not transplanted state |
| B24 DEV80 | NP4 55/80; EPP321 50/80; PE3-score 52/80; PE3-random 53/80; STOP_B24_METHOD_REFINEMENT | No disguised NP early pruning or protected exploration |
| B25 mechanism study | Hard iid selection has O(sqrt(h)) directional displacement versus weighted O(h); posterior TV need not improve single-truth MSE | A posterior improvement and a one-truth reconstruction gain are different claims; finite NP proposal density matters |
| B26 closed | 640 GPU root-arm trajectories, 393216 CPU toy trajectories; H Good25 55/80 vs signed raw R 53/80; mean R-H -0.2215 dB; recommendation STOP | No benefit established for aggregate NP H/R correction; do not duplicate B26 |

Detailed sources: [B19 correction](../../b19/b19_20_correction_2026_07_measurement_seed_audit.md), [B21.11](../../b21/b21_11_fresh2_final_benchmark.md), [B21.12](../../b21/b21_12_failure_atlas_decision.md), [B22 closeout](../../b22/b22_3_scientific_closeout.md), [B23.1](../../b23/B23_1_CHECKPOINT_REPORT.md), [B24 correction](../../b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md), [B25 return](../../b25/B25_FINAL_PLANNER_RETURN.md), [B26 accepted return](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/planner-b26-resume-handoff-20261015/docs/handoffs/B26_PLANNER_RETURN_2026-10-15.md).

## 3. Identification, exact symmetry, basin and target definitions

Let \(\mathcal A(x)=|\mathcal F(Px)|\) be the *actually implemented* support/padding/FFT magnitude operator and \(y=\mathcal A(x_\star)+\epsilon\). In a discrete synthetic prior with centers \(\mu_k\), define likelihood \(L_k(y)\propto\exp[-\|\mathcal A(\mu_k)-y\|^2/(2\sigma_y^2)]\) for a specified Gaussian amplitude noise model. The posterior mass is
\[
\pi_k(y)=\frac{w_k L_k(y)}{\sum_j w_j L_j(y)}.
\]
If two centers have exactly the same magnitude, their likelihoods coincide for *every* observation and the posterior odds equal the prior odds. The GT member is not identifiable from magnitude alone. A single-truth reconstruction objective can conflict with posterior fidelity. In the real FFHQ model signed noisy measurements must not silently be clamped across parents: NP's historical clamp_min(0) differs from DAPS/SITCOM raw treatment.

Use the exact symmetry group \(G\) admitted by the specified finite-support operator, not an assumed unlimited transformation group. In the 1D toy, use identity/reversal (no nontrivial within-support translation for full-span endpoints); track discrete orbit label \(\{A,RA\}\) versus \(\{B,RB\}\). Distinguish symmetry switches **within** a mode orbit from transitions **between** orbit classes.

On FFHQ, define a *diagnostic* basin as a reproducible cluster of native converged candidates under a frozen, symmetry-quotiented distance/feature representation, with separate prior-compatibility and observation diagnostics. Neither early pixel similarity nor mode separation proves correctness. Raw PSNR/Good25 remains the primary prospective reconstruction evaluation.

## 4. Source-level native interface / compatibility map

From [parent semantics](../../b23/02_PARENT_SEMANTICS_AND_COMPATIBILITY.md), [native replay](../../b23/B23_1_CHECKPOINT_REPORT.md), frozen \`configs/b23/*_frozen.yaml\` and direct source inspection:

| Parent | Native state and measurement | Parent behavior | Tentative insertion boundary | Qualification gate |
|---|---|---|---|---|
| **DAPS Fresh** | \`DAPSNativeState\`, annealing sigma/index, raw signed observation; FFHQ256 oversample2 sigma_y=.05 ann400 diff5 | Denoising/prior prediction, measurement-conditional Langevin/optimization to \`x0y\`, native re-noising | Conditional clean-space candidate proposal *before* native correction/renoising; alternatively after \`x0y\` with new native conditional refinement | Verify exact patched external DAPS source, replay, RNG and output; not a new independent restart |
| **SITCOM-ODE official** | \`SITCOMNativeState\`: annealing sigma, diffusion substep, LGVD optimizer input/state; raw observation | Optimize denoiser input \(v\), obtain \(f_\theta(v,t)\), apply forward noising | Initialize a *new valid native* LGVD solve from bridge-informed \(v\), never directly transplant DAPS/NP \`x0\` | Solve feasibility of obtaining denoiser-representable bridge candidate, check backward and forward consistency, charge UNet-backprop |
| **NP canonical** | \`NoisePickingNativeState\`, DiffFPR timestep, in-memory clipped observation; 5 preproj/1 postproj candidates | Re-noise selected denoised estimate, optionally reuse incumbent noise, denoise/score proposals, select, late LF projection | Propose within its own timestep/lineage, not as foreign latent state | Reconcile proposal law, retained noise, candidate-index asymmetry and B26; not first integration target |

The simple \`prdiffusion/algorithms/sitcom.py\` repository prototype and \`hybrid_np_sitcom.py\` are NOT the official frozen SITCOM parent. Actual SITCOM implementation is the official external \`SITCOM_ODE\` at \`275ab67efbd8146bffca20155171ba6be1169c09\`, whereas DAPS depends on a tracked patch and NP on the clean DiffFPR checkout. A source-level native operation must be verified on PAC before any future real-image run.

## 5. Prior-art novelty audit (not a patentability or exhaustive novelty finding)

| Primary source | Already contains | Why it matters |
|---|---|---|
| [DAPS, CVPR 2025](https://arxiv.org/html/2407.01521v3) | Decoupled annealing, MCMC conditional updates and intentionally large nonlocal corrections | Generic reinjection, annealing or re-noising does not count |
| [SITCOM, ICML 2025](https://arxiv.org/html/2410.04479v2) | Stepwise triple consistency including denoiser-input optimization | Generic measurement consistency/prior regularization does not count |
| Native NP and [B25/B26](../../b26/B26_NATIVE_ESTIMATOR_SPEC.md) | Score/rank finite proposals, hard versus weighted conditional selection | No disguised NP proposal-count/selection variation |
| [TDS](https://arxiv.org/abs/2306.17775) | Twisted SMC and conditional sampling | Particle weighting / proposals as such are old |
| [MGDM](https://arxiv.org/abs/2502.03332) | Mixture-based guidance and Gibbs approaches | Multimodal conditional sampling alone not new |
| [NCS](https://arxiv.org/abs/2510.23633) | Noise combination/subspace search for linear inverse problems | Noise-coordinate perturbations alone not new; differs in forward operator |
| [DAPS++](https://arxiv.org/abs/2511.17038) | Decoupled posterior annealing and separation of stages | Repackaging initialization vs refinement is insufficient |
| [CLAMP, ICML 2026](https://proceedings.mlr.press/v306/shin26b.html) | Geometry-correct denoiser-pullback curvature/Gauss–Newton guidance | Generic Hessian/preconditioned steps especially weak novelty |
| [Repulsive Latent Score Distillation, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html) | Kernel particle repulsion across nonlinear inverse problems including phase retrieval | Plain particle diversity not new |
| [DDRM-PR, Applied Optics 2025](https://doi.org/10.1364/ao.545150) plus classical HIO/Fienup | Diffusion prior + phase-retrieval alternating Fourier projections | Fourier phase replacement + denoiser not new |
| [Classical homometric/Bloom turnpike example](https://infoscience.epfl.ch/bitstreams/f1e42b4b-cd81-4d38-9332-943caf0b5466/download) | Non-congruent supports with the same differences/autocorrelation | Our toy is a recognized counterexample, not an invented result |

**Novelty risk:** B1 medium-high (specific phase/prior coupling is the only candidate novelty delta); B2 high (CLAMP and curvature literature); B3 very high (TDS/MGDM/repulsion). None is currently an established new result; a narrow theory-only finding is a valid fallback.

## 6. Three falsifiable proposals

### B1: prior-aligned Fourier-phase bridge — **first pilot**

Use a native clean-space candidate \(x\) and a meaningful native diffusion-prior direction \(g_t(x)\). Put \(Z=\mathcal F(Px)=a e^{i\phi}\), \(G=\mathcal F(Pg_t)\). The derivative of the log-prior along Fourier-phase angle \(\phi_\omega\) is proportional (subject to Hermitian-real constraints) to
\[
d_\phi(\omega)=\frac{\operatorname{Im}(\overline{Z_\omega}G_\omega)}{|Z_\omega|^2+\varepsilon}.
\]
The denominator is a **chosen preconditioner**, not a general exact Riemannian projection. Propose \(\phi'=\phi+\delta\,\mathcal H(d_\phi)+\xi_\phi\) with a Hermitian-compatible angular direction \(\mathcal H\) and a pre-registered symmetric perturbation \(\xi_\phi\). Construct \(Z'=a e^{i\phi'}\), and \(u=P^\ast\operatorname{Re}\mathcal F^{-1}(Z')\); reapply the **native** parent conditional refinement and native re-noising.

**Major mathematical caveat:** \(P^\ast\) crops/restricts to the real image support; generically \(|\mathcal F(Pu)|\ne a\). Constant amplitude in unrestricted Fourier space does **not** imply support-feasible measurement consistency. Record the *actual* residual after crop/refinement and stop if bridge phases never remain feasible. Real/Hermitian constraint also needs care at DC/Nyquist and near-zero Fourier coefficients.

Conservative exploratory objective (not posterior exact): in the analytic toy where \(q_{\eta}(x\mid y)\) is available, accept an image only if \(-\log q_\eta(u\mid y)\) decreases by a fixed margin **and** the post-support measurement residual stays under a frozen bound; otherwise keep the incumbent. In FFHQ the exact density is unavailable; a prior surrogate must be independently audited before copying this acceptance rule. Do NOT claim invariant posterior law without the correct proposal density and Metropolis/reversibility proof.

Hypothesis: the **specific prior-aligned Fourier phase direction**, not only larger steps or extra prior-gradient evaluations, improves crossings into the preferred prior-compatible orbit compared with same-work random-phase, image-space perturbations, and score-directed image-space moves. Failure to beat those controls falsifies the mechanism.

### B2: conditional energy weak/saddle-curvature bridge — **fallback only**

For a well-defined parent-specific energy \(E_t(x)=\ell_y(x)+R_t(x)\), estimate low Hessian eigenvectors \(v_j\), propose \(u=x\pm r v_j\), re-refine under native parent, accept under a frozen non-GT criterion. At a positive-definite minimum even \(v_{\min}\) is not an escape certificate; the needed move may be nonlocal. Must beat Hessian/CLAMP-like preconditioning and isotropic step at equal curvature-vector/gradient work, not just unpreconditioned baseline. Expensive Hessian-vector products make novelty and tractability unfavorable unless a precise saddle/bifurcation phenomenon appears.

### B3: symmetry-quotient mode-preserving particles — **diagnostic/deferred**

For particle estimates \(x_i\), use a symmetry-invariant distance \(d_G(x_i,x_j)=\min_{g\in G}\|x_i-gx_j\|\) and identify distinguishable modes. An interaction kernel should preserve posterior mass, not force equal occupancy; proper importance/SMC weights need proposal density and trajectory bookkeeping. TDS, MGDM and repulsive sampling substantially preempt a stand-alone method here. B24 excludes simply pruning/reallocating NP particles by surrogate scores. Only consider as a way to study B1/B2's generated candidates, not as the lead method.

## 7. TB-0 synthetic design — precise pilot and falsification

**Objective:** establish or reject a Fourier-phase *orbit-to-orbit* transition mechanism under exact known prior scores and observation likelihood, without using FFHQ or pretrained weights.

**Homometric test pair:** \(A=\{0,1,4,10,12,17\}\), \(B=\{0,1,8,11,13,17\}\); \(\mu_A,\mu_B\in\mathbb R^{18}\) are indicator vectors and \(R\) reverses indices. Nontrivial orbit classes are \(\{A,RA\}\) and \(\{B,RB\}\). Full-span endpoints fix translations inside this chosen support. The pair has identical aperiodic autocorrelation and zero-padded FFT magnitude. A direct numerical sanity check (36/64/128 FFT) gives equality at ~1e-15 and the reversal check is false. Executor must check **exact integer autocorrelation** independently, verify all four supports, and verify FFT normalization/padding; a floating-point near-tie must not define ground-truth basin identity.

**Prior:** \(\Pr(X_0=\mu_{A},R\mu_A,\mu_B,R\mu_B)=(.46,.46,.04,.04)\). Let \(Z_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\epsilon_t\). For any measurement \(y\),
\[
p(z_t\mid y)=\sum_k \pi_k(y)\,\mathcal N(z_t;\sqrt{\bar\alpha_t}\mu_k,(1-\bar\alpha_t)I).
\]
The exact annealed posterior score and Tweedie denoiser are finite sums; all methods get identical access and billed evaluations. Here, because all four ideal homometric centers have exactly equal measurement predictions, \(\pi_k(y)=w_k\) regardless of observed \(y\), including noisy observations. This is an **identifiability stress test**, not a demonstration that the measurement supplies a preference for the true A center. The declared practical objective is to move low-prior-orbit starts toward high-prior-orbit probability without damaging high-prior-orbit starts. Measure posterior calibration separately; a MAP-like operator is not an unbiased posterior sampler.

**Four equal-budget arms:**
- S0: native analytic annealing/conditional-step baseline, no bridge.
- S1: same parent + matched isotropic clean-space perturbation and re-refinement.
- S2: same parent + random Hermitian-compatible Fourier-phase perturbation and re-refinement.
- S3: same parent + prior-aligned Fourier-phase bridge and re-refinement.
**Mandatory fifth control if S3 uses an extra exact score:** S4 = score-directed image-space perturbation with identical score calls, proposal attempts, refinements, RNG budget, and target tests. Otherwise S3 improvements may be simply the result of using the prior, not of Fourier phase geometry. Compare S3 against the BEST of S1/S2/S4, not only S0.

Matched seeds: pair initial states, draw common predeclared RNG substreams for baseline, proposal, refinement; each method receives the same number of proposal opportunities, score calls and refinement gradients (pad with logged dummy evaluations only for work-count comparison; also report actual needed calls separately). Count FFTs, energy/likelihood calls and wall time, and compare both equal-score-work and equal-wall budgets. No uncounted branches, no image-truth choice at runtime.

**Parameters:** first fix \(N=18\), FFT length 64 (cross-check 36 and 128 only as operator sanity), \(\sigma_y=0.05\) where needed, exact mixture weights above, a small pre-declared annealing ladder (e.g. three \(\bar\alpha\) levels), 1-2 candidate bridge points per transition and at most four native conditional-refinement iterations per proposal. Use a small **development** grid of proposal scale (e.g. \(0.2,0.5,1.0\) radians after angular-direction normalization), then FREEZE the chosen operator and all params before a distinct synthetic-check set. Do not grid-search the confirmation seeds.

**Truth-free runtime acceptance:** fixed conditional log-density/energy and measurement discrepancy only; with discrete prior, modes are labeled offline by nearest symmetry orbit at a declared tolerance. The executor should assess whether log-density is used consistently at the noisy checkpoint; it must not evaluate an undefined discrete \(-\log p_0(x)\) for off-atom candidates. For the canonical exact test the annealed posterior is available; for later FFHQ no density-correctness claim transfers automatically.

**Primary readouts:** paired probability of transitioning \(B\)-orbit to \(A\)-orbit; net \(A\)-retention/harm rate; accepted transition displacement; symmetry-only switches; final measurement residual *after support projection*; conditional energy; posterior component-weight TV/calibration if treating outputs as samples; work counters. Require 200 independent paired synthetic CHECK starts from low-prior orbit and 200 from high-prior orbit, plus 200 starts drawn from prior weights to reveal distributional distortion. Split development/check seeds using named deterministic namespaces; record all raw CSV rows. Use paired bootstrap CIs or exact paired discordance test; no pretending start repeats are independent images in later FFHQ.

**Hard STOP gates on the check split:** S3 minus best cost-matched S1/S2/S4 \(\ge 10\) percentage points in genuine B-to-A reach, with paired 95% interval lower endpoint \(>0\); excess A-to-B harm \(\le2\) percentage points; actual post-support measurement residual meets the predeclared noise-calibrated bound in accepted candidates; no extra hidden score/FFT/refinement work. If S3 fails S2, alignment is not supported. If S2/S3 both fail image-space controls, phase-space hypothesis rejected. Repeated phase projection reverts to original basin -> STOP.

**No exact posterior-sampling claim:** greedy target-density improvement by construction changes stationary mass. Any posterior TV plot is a warning/diagnostic. Secondary near-homometric tests may perturb one center's amplitude and recompute **true** component likelihoods; this is NOT part of the original exact-equality identity gate and must be designated separately.

**Resource ceiling:** Zero GPU, zero pretrained inference, zero real images/measurements, \(\le2\) CPU-core-hours, target RAM <2 GiB, compact CSV/JSON/report under track-owned namespace only. No PAC session necessary; deterministic small local CPU arrays suffice. The executor may stop early on a decisive mathematical impossibility and submit proof/STOP instead of consuming the budget.

**Advance to FFHQ only after independent scientific review:** real-cohort reservation through issue #41, native source SHA/model SHA, matched DAPS baseline/restarts/random-phase/score-space controls, UNet/FFT cost and wall audit, a 20-32-image novel DEV pilot (not B21/B22/B24 reused as prospective), clean-free selector frozen, paired harm/rescue and raw Good25/q05 with image-level uncertainty. Transfer to official SITCOM only with a new native LGVD representability gate. Neither prospective test cohort nor PAC GPU allocation is authorized *by this document*.

## 8. Failure criteria, risk and proposed governance

The proposal fails if support constraints negate phase-space moves; if the prior aligned direction is not a real additional transition over score-guided image-space changes; if accepted candidates collapse to same orbit; or if native DAPS already does equally well at matched cost. A four-atom discrete prior has an analytically exact score but is not a credible natural-image manifold; success there is a feasibility condition, never FFHQ evidence. Exact observed symmetry can produce legitimate alternate modes; no algorithm can infer orientation from magnitude in those cases without extra prior information.

Scientific next action: one TB-0 CPU synthetic executor handoff with explicit branch and STOP criteria, followed by a critical Track B review of its committed return. Nothing in this document delegates ownership of historical NP, B26 or Track A datasets.

## 9. Persistent references and TRACK_B_PLANNER_RETURN fields

- Governance: [shared charter](../metaplanner/2026-10-15_SHARED_CHARTER.md), [autonomy amendment](../metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md), [Track B mandate](2026-10-15_METAPLANNER_NEXT_MANDATE.md), [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).
- Planner branch: \`codex/prdiff-track-b-basin-planner-20261015\`.
- Native contract: DAPS/SITCOM/NP only via separately qualified native boundaries; no cross-family donor transfer.
- Preferred B1: prior-aligned Fourier-phase bridge (unvalidated); fallback B2 conditional saddle/weak-direction bridge; deferred B3 symmetry quotient particles.
- Authorized scope under autonomy amendment: local zero-GPU TB-0 proof of concept and delegation; no PAC GPU, new FFHQ measurements, or locked panels.
- Branch/output namespace: \`codex/prdiff-track-b-<study-id>\`, \`docs/track_b/executors/<study-id>/\`, \`scripts/track_b/\`, \`configs/track_b/\`, future PAC \`/egr/research-pac/huang248/outputs/pr_diffusion/track_b_basin/\`.
- Source exactness: this report records the reviewed GitHub scientific base; each future executor must record **its own** exact pre-run SHA, model/operator identity and all work receipts.
- Output: proof-of-concept, falsifier and credible negative-result acceptance, **not** a promised positive novel method.
