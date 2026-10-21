# A1.5 post-M0 scientific return and conditional next-stage decision
**Date:** 2026-10-21; **owner:** Track A / FAST-DIPS mechanism audit. **Document status:** GitHub source-backed and user-reported PAC stdout; next GPU work NOT implemented or authorized by this document.

## Observations verified by actual user-operated PAC

The earlier GPU K1/K2/K4 eight-image exposed FFHQ experiment remained pinned at **A1.5 execution HEAD** `d51170f77ee672990c22f1703c5997e949dda9d6`, unmodified official FAST-DIPS HEAD `a9758b38906b7c7950fa04491f37f6634948db26`, and original signed noisy measurements. With four independent roots per image, mean offline raw PSNR FAST 31.525134 dB versus DAPS 31.517951 dB, NP-LF 30.388019, NP-S2 30.221876, SITCOM 27.438794. **Eight already-exposed development images, not holdout**. FAST sampling-only four roots 14.851145 s/image.

Following this, the user ran source-isolated **A1.5 M0** CPU audit from commit `a9b5dedb0c5586d55991a7f9cb82f13aea847fff`. Observed log: `A15_CPU_MECHANISM_AUDIT_PASS`, RC0, 8 images, all 32 candidate terminals SHA verified; **all eight frozen signed observations numerically certified infeasible** for FAST native global measurement Euclidean ball radius epsilon=0.05; average rigorous lower bound `||min(y,0)||_2 = 19.085433188982034`. Mean sum of four root *program wall after module imports* `32.439573873067275 s/image`, versus 14.851145s/image *sampling only*. Reconstructed image GT **not** read in M0. Receipt SHA `b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74`, at `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/a15_6_mechanism_cpu_audit_v1/A15_MECHANISM_AUDIT_COMPLETE.json`; root CSV `A15_PER_CANDIDATE_MECHANISM.csv`.

**What remains unseen:** User has NOT provided the 8 individual feasibility lower bounds, terminal residuals, or root-level M0 CSV. Do not claim individual values from the average or invent a correlation with PSNR. M0 itself computed and wrote them, ready for read-only inspection.

## A sharp source-level algebraic observation for K_ADMM=2

See official source `main.py` (pinned) functions `FAST_DIPS.sample` and `x_update_autograd`, lines ~245–287 and 55–113.

Denote denoised anchor `p = D_theta(x_t,sigma)`, initial `x=p`, `v=A(p)`, `u=0`, and measurement `y`. Data radius `epsilon=0.05`.

The first ADMM image update minimizes `||x-p||^2/(2 sigma^2) + (rho/2)||A(x)-(v-u)||^2`. At `x=p`, `v=A(p)`, `u=0` its gradient vanishes exactly, so **all S=5 iterations in first x-update are mathematically redundant** in exact arithmetic. They use forward/adjoint FFT work but no diffusion-net calls.

Then `v_1=P_{B(y,epsilon)}(A(p))`. Let `r=A(p)-y`, `d=||r||_2`, `alpha=min(1,epsilon/d)`. Then
`v_1 = y+alpha*r`; `u_1=A(p)-v_1=(1-alpha)r`; the second x-update receives target `b_2=v_1-u_1=2v_1-A(p)`.
At `x=p`, the second image-update gradient is
`g_2(p)=rho J_A(p)^T(A(p)-b_2)=2rho J_A(p)^T(A(p)-v_1)=2rho(1-alpha)J_A(p)^T(A(p)-y)`.
This is an *exact formula for the local initial gradient*, not a claim that the entire five-step nonlinear x-update equals an ordinary single gradient step.

As `A(p)>=0` for Fourier magnitude, `d>=||min(y,0)||_2` for **every** possible denoised anchor. On all 8 actual observations `||min(y,0)||_2 > epsilon`, so `alpha<1` always: the projection is necessarily active on the first ADMM iteration. Characteristic `epsilon / mean lower bound = 0.00262`, but this **does not bound each image's alpha**; calculate individual bounds from M0 before reporting per-image values. For small alpha, the first effective correction acts like a prior-regularized, scaled measurement gradient. A K_ADMM=1 control does **not** perform a substantive image correction at any diffusion step under this initialization, so it is an effectively unconditional/prior-only **negative control**, not an honest one-correction comparison.

Native final timestep has `if is_last_step: xt=x0hat`, ignoring final corrected `x0y`; previous 149 timestep corrections still matter. The native measurement-ball projection ensures `v in B(y,epsilon)` even when no `x` yields `A(x)` in this ball. Nothing in M0 demonstrates the finite-step output is feasible or that this approximate behavior is a proven cause of quality.

## What is and is not novel

- **Already published by FAST-DIPS (ICLR2026):** its diffusion prior anchored optimization, finite-K ADMM, analytic/FD step and backtracking, projection and resampling, runtime/quality.
- **Verifiable source/experiment mechanics to test:** redundant first S image-gradient subloop; effective 2nd-step measurement-gradient formula; hard ball infeasibility for signed observations, final corrected iterate discarded.
- **Not proven:** that physical infeasibility causes superior reconstruction, that discarding last correction harms quality, that nonstandard residual units are a bug, or that a trivial loop deletion forms a publishable method contribution.

## Decision and execution gates

**P0 now (no GPU):** inspect M0's existing `per_image` values for `strict_nonnegative_forward_residual_floor_l2`, `floor_to_epsilon_ratio`, `min_output_residual_l2`, `best_of_4_raw_psnr_db`, `total_4_root_program_wall_after_imports_seconds`. Inspect per-root CSV specifically for original raw K1 failure images 67174 and 66631, all rep0..3. This allows correlation inspection **without** generating any new candidate; no GT tensor read since saved PSNR metadata is sufficient.

**M1 tiny GPU ablation, not yet authorized/implemented:** preserve original code+weights+measurements and PR trajectories; choose *before observing M1 outcomes* 69073 success and 67174 failure from the already-exposed development set (a sensitivity / mechanism demonstration only). Copy the native method into a side-by-side *source-versioned* adapter with exact equivalence guard. Test:
1) baseline original K=2 as replicated control;
2) skip the mathematically redundant first S=5 x-update **but retain the first auxiliary v projection and u update**, expecting bitwise or tiny-roundoff identity and a measurable FFT/AD saving;
3) capture both last-step `x0hat` and `x0y` from the **same native trajectory** for retrospective comparison; do not alter earlier steps. This avoids confounding candidate randomness and adds no extra diffusion call;
4) K=1 prior-only negative control *only if useful*: clearly label it non-conditioned as per source algebra.
Pre-register seeds, raw PSNR and 180-degree diagnostic, measurement residual RMS/total L2, exact U-Net calls, CUDA timing and external wall, and implement CPU structural tests before reserving GPU0. Do not tune on eight exposed results and claim prospective significance.

**M2 conditional:** ε-scale ablation (`ε=.05` vs `ε≈σ sqrt(m)`, where m=442368) and signed-y vs clipped-y, but these change the measurement optimization constraint and/or observation model, must be labeled changes, and must not be conflated with the published native baseline. Separate score/quality and physically feasible residual metrics.

**Hard governance:** Track B priority, all GPU jobs require new issue41 [RUN RESERVATION] with time/device/identity; native execution A15 old branch `d51170f` and original M0 execution source `a9b5ded` stay immutable. Do not access B24 confirmation305. Future research paper needs clear novelty beyond published FAST and true fresh prospective phase-retrieval cohort.

## Sources
- Native FAST-DIPS pinned source: https://github.com/ququlza/FAST-DIPS/blob/a9758b38906b7c7950fa04491f37f6634948db26/main.py
- FAST-DIPS paper: https://arxiv.org/pdf/2603.01591
- A1.5 M0 source + prereg: https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-a15-mechanism-audit-20261021/docs/track_a/executors/a15-mechanism-audit/A15_FAST_NATIVE_MECHANISM_FINDINGS.md
- Issue41 main cross-track ledger: https://github.com/Cheng-HanHuang/pr_diffusion/issues/41
