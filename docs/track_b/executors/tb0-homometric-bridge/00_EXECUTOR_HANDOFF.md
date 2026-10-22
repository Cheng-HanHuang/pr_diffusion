# TB-0 Executor Handoff — Synthetic homometric Fourier-phase basin-transition test

**Date:** 2026-10-15  
**Owner:** Track B scientific lead; delegated TB-0 implementation executor  
**Branch:** \`codex/prdiff-track-b-tb0-homometric-bridge\` (created from scientific planner return commit \`007b7cc2121333ed4930e4d7ae9c79bd4f23277c\`)  
**Base scientific report:** [complete Track B planner return](../../../planning/track_b/2026-10-15_TRACK_B_PLANNER_RETURN.md)  
**Governance:** [metaplanner autonomy amendment](../../../planning/metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md), [shared charter](../../../planning/metaplanner/2026-10-15_SHARED_CHARTER.md), [coordination ledger #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).  
**Status:** AUTHORIZED CPU-only wholly synthetic proof of concept. This is not an FFHQ implementation/GPU permit.

## Mission

Build the *smallest reproducible, falsifiable* experiment that establishes whether a diffusion-prior-aligned Fourier-phase proposal transitions between **non-congruent, Fourier-magnitude-equivalent** reconstruction orbits at matched work, while protecting incumbents already in a plausible mode.

The role is **executor**, not a separate research track: you implement, test, analyze, and propose a scientific GO/STOP under the gates here, returning to the Track B lead for review. You may recommend a mechanistic pivot if the mathematical premise breaks; do not quietly change the primary acceptance/gates after looking at frozen check results.

You have no direct PAC authority. Do not access /home, real FFHQ imagery, pretrained model checkpoint, B21/B22/B24/B26 outputs, confirmation305, or any GPU. The small synthetic test may run in an isolated CPU environment without the user running PAC commands. You may use existing local Python only after checking packages; no default-PAC-Python assumption. No broad data scans.

## 1. Formal toy and mandatory identity preflight

Set \(n=18\) and define binary supports
\[
A=\{0,1,4,10,12,17\}, \qquad
B=\{0,1,8,11,13,17\}.
\]
Let \(a,b\in\mathbb R^{18}\) be support indicator vectors and \(R\) reversal \( (Rx)_j=x_{17-j}\). Check that \(a,b,Ra,Rb\) form **two distinct symmetry orbits** (A-orbit and B-orbit) and no orbit duplicates are falsely counted as new basin discovery.

Define zero-pad \(P:\mathbb R^{18}\to\mathbb R^{64}\), orthonormal FFT \(F_{64}\), and
\[
\mathcal A(x)=|F_{64}Px|.
\]
Do not silently change support, FFT normalization or loss to match real FFHQ. As preflight before any solver run, compute integer aperiodic autocorrelations for all four atoms, assert exact equality, assert no reversal/congruence A=B, and check padded magnitude mismatch \(\le 10^{-12}\) (64-point orthonormal FFT). Test 36 and 128-point transforms only as identity sanity, not as extra tuning data. Derive why equal autocorrelation implies equal Fourier magnitudes. Record check in machine-readable JSON and a unit test.

The example is classical homometric/turnpike (Bloom); **do not claim discovery of this pair**.

## 2. Prior, exact reference and score

Prior on four CLEAN atoms \(\mu=(a,Ra,b,Rb)\) with
\[
w=(0.46,0.46,0.04,0.04).
\]
For noisy observations \(y=\mathcal A(\mu_k)+e\), with \(e\sim N(0,\sigma_y^2 I)\), \(\sigma_y=0.05\), compute
\[
L_k(y)=\exp\left[-\frac{\|\mathcal A(\mu_k)-y\|^2}{2\sigma_y^2}\right],\quad
\pi_k(y)=\frac{w_kL_k(y)}{\sum_jw_jL_j(y)}.
\]
Because the four centers are homometric, they have equal likelihoods for each y, and \(\pi=w\) theoretically. Check stable log-sum-exp probabilities numerically. Do NOT use ground-truth index at runtime to score, accept or orient candidates.

Use the exact forward-noised mixture
\[
q_t(z\mid y)=\sum_k\pi_k(y)\,\mathcal N(z;\sqrt{\alpha_t}\mu_k,(1-\alpha_t)I)
\]
for a fixed \(\alpha_t\in (0,1)\). Derive exact \(\nabla_z\log q_t(z\mid y)\) and conditional mean \(\mathbb E[X_0\mid z,y]\), both available by normalized Gaussian mixture responsibilities. Verify the score by finite-difference tests (small fixed input). Avoid undefined off-atom \(-\log p_0(x)\).

For the bridge, work in clean-scale noisy coordinates \(u=z/\sqrt{\alpha_t}\): then the mixture \(q^\mathrm{clean}_t(u\mid y)=\sum_k\pi_k(y)N(u;\mu_k,\tau_t^2 I)\) with \(\tau_t^2=(1-\alpha_t)/\alpha_t\). The gradient \(g_t(u)=\nabla_u\log q^\mathrm{clean}_t(u\mid y)\) is exact. Evaluate \(-\log q^\mathrm{clean}_t\) as the **annealed posterior energy** for screening. Treat this as a model-knowledge upper bound for synthetic science, NOT a readily available FFHQ density.

Pre-register three reasonable levels e.g. \(\alpha_t=(0.25,0.50,0.80)\), or defend a different three-point ladder in code *before* generating CHECK outputs. At high noise, discrete orbit assignment of u may be ambiguous; the output must be refined toward an atom and classified with a strict radius instead of counting a vague cluster.

## 3. Fully specified phase bridge and support check

For \(u\in\mathbb R^{18}\), \(Z=F_{64}Pu\), \(G=F_{64}Pg_t(u)\). For \(|Z_\omega|\) safely nonzero, define phase-alignment direction
\[
d_\omega = \frac{\mathrm{Im}(\overline{Z_\omega}G_\omega)}{|Z_\omega|^2+\varepsilon}.
\]
This represents a PRECONDITIONED spectral-phase prior directional derivative (not a guaranteed exact tangent on the supported magnitude level set). Project/construct a phase direction with the necessary conjugate Hermitian angular symmetry for the inverse to be real. Handle DC/Nyquist consistently (zero phase change for self-conjugate bins). For near-zero amplitudes, do not amplify arbitrary numerical noise: freeze those components, use a reported denominator floor, and skip the move when the directional norm is below a stated threshold.

Candidate:
\[
Z'_\omega=|Z_\omega|\exp(i[\arg Z_\omega+\delta \hat d_\omega+\xi_\omega]),\quad
u_\mathrm{raw}=P^\ast\operatorname{Re}F_{64}^{-1}Z'.
\]
Refine \(u_\mathrm{raw}\) through an explicitly documented fixed-step analytic conditional updater applied identically to controls. Do not call this implementation a bitwise native DAPS reproduction; it is a DAPS-inspired toy annealer with an analytically known prior.

**Important:** truncation \(P^\ast\) generically destroys \(|Z|=\)constant. Record residuals BEFORE and AFTER support crop and AFTER all refinement. A proposal is not observation-preserving merely because Fourier amplitude was held fixed before inverse/crop. The proper test is \(\|\mathcal A(u_{\rm final})-y\|\). Do not temporarily allow out-of-support pixels as uncounted extra unknowns.

Use a fixed non-GT acceptance rule at the chosen t:
1. \(-\log q^\mathrm{clean}_t(u_\mathrm{final}\mid y)\le -\log q^\mathrm{clean}_t(u_\mathrm{inc}\mid y)-\Delta_E\) with predeclared \(\Delta_E\);
2. signed raw-observation magnitude residual \(\|\mathcal A(u_\mathrm{final})-y\|^2/\sigma_y^2\) lies under a precomputed noise-law threshold, such as \(\chi^2_{64,0.99}\) (declare quantile implementation);
3. otherwise return incumbent exactly.

The energy criterion is **greedy MAP-like**, not posterior invariant; no sampling exactness claim. Alternative acceptance / objective is permitted only as an openly reported new *development* hypothesis, and the first specification is preserved for comparison.

## 4. Required work-matched arms

All share the same synthetic parent, initial states, annealing ladder, observation, score access, conditional refinement, checkpoint count and non-GT acceptance. Use separate named RNG streams and paired seeds. Keep a separate work ledger for actual and billed-equivalent work.

- **S0 — Parent only:** same analytic conditional annealer, with equal-work additional native updates to provide a meaningful cost control.
- **S1 — Isotropic move:** random image-space perturbation + identical refinement.
- **S2 — Random phase:** Hermitian-compatible random Fourier phase perturbation of identical angular norm distribution + identical support crop/refinement.
- **S3 — Prior-aligned phase bridge:** phase direction above + same perturbation and refinement.
- **S4 — Prior-score image-space move:** a normalized score-guided clean-space displacement + random part, *with identical exact prior-score evaluations* and native refinement.

S4 is MANDATORY: a bridge that only beats unguided random phases has not isolated the geometric contribution.

If proposal scale normalization differs (phase rad vs image norm), report both displacement norms and define a fair scale-matching rule *before* check data; include norm-matched ablation if feasible. No extra independent starts or selecting the best GT-PSNR candidate. Every rejected proposal and score/likelihood evaluation belongs to method cost.

## 5. Frozen develop/check design

Use deterministic independent namespaces \`tb0/dev/*\`, \`tb0/check/b-start/*\`, \`tb0/check/a-start/*\`, \`tb0/check/prior-start/*\`, \`tb0/noise/*\`. Never re-use a dev seed for frozen check. Commit configuration (test counts, alpha ladder, phase scales, gate tolerances, selector, seed scheme) before using CHECK results.

- **Preflight:** no stochastic performance experiment until operator, score gradient and symmetry tests PASS.
- **DEV:** up to 30 paired low-prior B-orbit starts and up to 30 high-prior A-orbit starts for checking numerical stability and choosing one fixed phase scale from \((0.2,0.5,1.0)\) radians after angular normalization. Do not use held-out check outcomes to reselect.
- **CHECK:** 200 paired B starts, 200 paired A starts, 200 genuine-prior starts, all independent within predefined classes and all paired across S0–S4. Use one observation per declared observation-seed group and correctly report correlations if multiple starts share y. Avoid treating them as separate independent images.
- Optional a priori near-homometric variant modifies one center, recomputes exact likelihoods, and must be labeled as SECONDARY; never silently weaken strict homometric test.

Truth diagnostics and symmetry-orbit labels computed **only after** trajectory choice. Classify as A/B only if within a predeclared normalized radius around one of the four exact atoms and measurement-feasible; else label MIXED/UNRESOLVED. Never count simply moving farther from B as entering A. Separately measure orientation-only transitions within G.

## 6. Freeze and STOP gates

The Track B lead must decide if the method is scientifically useful, not only whether some statistic improves.

**STOP if ANY:**
- identity, exact score or conjugate/supported output tests fail;
- S3 does not beat best work-matched S1/S2/S4 B-to-A discovery by >=10 percentage points on CHECK with paired 95% CI lower bound >0;
- excess damage to good A starts (A-to-B or lost A) over best matched control >2 percentage points;
- accepted post-crop candidates fail the declared observation-consistency gate;
- advantage disappears under equal-score/FFT/refinement work or represents only symmetry reversals;
- unsupported numerical amplification at near-zero Fourier magnitudes drives transitions.

The 10pp and 2pp are *synthetic advancement thresholds*, not predictive effect sizes for FFHQ. If statistical precision is insufficient to conclude, report INCONCLUSIVE instead of assuming PASS. A proof of impossibility or a clear negative result can replace full stochastic CHECK execution, with explicit equations/tests and STOP.

If S3 seems positive, first audit whether score normalization, mixture exactness, acceptance objective or metric definition grants it a hidden advantage. No FFHQ or native DAPS/SITCOM implementation is authorized by a toy GO alone.

## 7. Repository and execution permissions

- Write only to **this executor branch** \`codex/prdiff-track-b-tb0-homometric-bridge\`, using \`scripts/track_b/tb0_*/\`, \`configs/track_b/tb0_*/\`, \`docs/track_b/executors/tb0-homometric-bridge/\`.
- Keep a source-frozen pre-run commit and record its SHA in output metadata; results/results-summary in the same track-owned docs folder, with test and numerical receipts.
- Small synthetic outputs can be attached under track-owned repo paths if compact; no giant tensors or archives.
- Use CPU only. Cap scientific work at 2 CPU-core-hours and RAM target under 2 GiB. No actual GPUs; no pretrained model inference.
- No PAC outputs, worktrees or shell commands should be assumed to have run. If local source execution is not possible, give a reproducible user-run CPU command, not a claim of completion.
- Do not touch Track A, B26, B24 confirmation305, historical branches/PRs, main, or any downloaded FFHQ artifacts.
- No \`set -e\`, \`set -u\`, \`set -o pipefail\` or terminal-ending commands in user-facing shell scripts. No \`/home\` paths in PAC-related instructions. No unstated GPU work.
- No new real-image data/cohort reservation is required for wholly first-principles toy data. This executor must nevertheless register itself as [EXECUTOR] in issue #41; Track B lead will do initial registration.
- Do not merge or retarget PRs, force push or reset history. Keep all changes reviewable.

## 8. Deliverables / executor return

Implement well-tested module with exact homometric toy, exact noised score, measured FFT operators, frozen sampler/control definitions and deterministic RNG. Tests must cover integer autocorrelation identity, Hermitian realness, posterior likelihood weight equality, finite-difference score, phase truncation mismatch (not necessarily zero!), retention acceptance, energy monotonicity, work counters, seed namespaces, orbit classification, and no GT use during runtime decisions.

Produce compact tables plus:
- \`docs/track_b/executors/tb0-homometric-bridge/TB0_EXECUTOR_RETURN.md\` with source commit, chosen frozen config, tests PASS/FAIL, all arm results + paired CIs, work, diagnostic plots if compact, artifacts SHA, numeric limitations and single GO/STOP recommendation;
- \`docs/track_b/executors/tb0-homometric-bridge/TB0_STOP_REPORT.md\` instead if concept dies at preflight, preserving negative proof/test receipts;
- a human-readable executive summary to Track B lead explaining whether Fourier-phase geometry adds value *beyond the prior score itself*, and whether support truncation is fatal.

Commit and push the source/results on **executor** branch; do not merge to planning branch. Track B lead reviews, posts accepted result to issue #41 and decides next scientific hypothesis.

## Ready-to-use executor instruction

You are the TB-0 CPU synthetic executor for the PRDiffusion Track B project. Read the entire handoff in this directory, the complete Track B planner return, the metaplanner autonomy amendment and issue #41 before changing code. Work only on \`codex/prdiff-track-b-tb0-homometric-bridge\`. Implement the exact homometric symmetry-orbit basin-transition toy, including S0–S4 work-matched controls, frozen seeds, exact mixture score, phase support audit and strict GO/STOP gates. Run only small local CPU synthetic tests. Commit a complete source-backed TB0_EXECUTOR_RETURN or STOP_REPORT on your executor branch. Do not use PAC/GPU/FFHQ/historical confirmation datasets. Return the commit and scientific verdict to Track B lead.
