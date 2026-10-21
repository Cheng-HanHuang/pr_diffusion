# A1.6 — Four-GPU FAST-informed Noise Picking research autopilot

Date: 2026-10-21. STATUS: implementation committed, NO PAC GPU work executed by the executor. Plan, test, exact replay smoke, and GPU reservation are mandatory before four-GPU execution.

Source branch: codex/prdiff-track-a-a16-np-fast-autopilot-20261021
Reserved checkout: /egr/research-pac/huang248/pr_diffusion_track_a_a16_np_fast_autopilot
Reserved outputs: /egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021
Original A15 FAST GPU source d51170f77ee672990c22f1703c5997e949dda9d6, M0 source a9b5dedb0c5586d55991a7f9cb82f13aea847fff, frozen historical NP source f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd: DO NOT MODIFY.

## 1. Critical review of attached deep-research synthesis

Strong starting hypotheses:
- FAST inexpensive post-denoiser measurement updates may supply greater phase-retrieval improvement per additional model evaluation than NP's five candidate UNet evaluations per early step. FAST-DIPS ICLR2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html
- Sparse early branch survival could rescue wrong basins; compare true two-survivor propagation with a one-path compute comparator. DAPS CVPR2025: https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.html
- Noise-aware scoring is plausible; basic moment scoring here is NOT an implementation of the full Distributional Consistency loss (ICLR2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html

Novelty caution: particle filtering for diffusion inverse problems (PFLD, arXiv:2408.13868), SNR-adaptive candidate counts (Diffusion Posterior Proximal Sampling, arXiv:2402.16907), FAST correction, and phase-retrieval Fourier ambiguity handling (DiffFPR ICML2024) already exist. A combination alone is not publication-ready. A noise-normalized global squared residual merely rescales scores by the constant 1/(m*sigma^2) and cannot change rankings. Therefore this trivial arm is omitted. The report's broad speculation about quality is NOT validated by eight exposed images.

A15 prior evidence: FAST K4 signed residual selector achieves 31.483358 dB mean raw PSNR versus offline GT oracle 31.525134; 5/8 selected same root, mean regret 0.041776 dB. These are 32 already-exposed roots only and do NOT validate an independent GT-free selector. A15 M0 shows 8/8 signed high-dimensional Fourier observations infeasible for native epsilon=.05 global L2 ball: mean provable negative-part bound 19.085. Native output residuals ~33.1, as expected from Gaussian sigma=.05 vector noise. A16 CPU-only early analysis measures residual margin and Spearman on SAME saved 32 roots.

## 2. Exact NP / FAST distinctions

Frozen B22 NP (original selector scripts/pr_external_difffpr_np_guided_lf_s2_selector.py plus scripts/pr_external_difffpr_np_benchmark.py):
- NP LF and S2: 1000 reverse scheduler steps, five noise candidates early and one late, ~2200 UNet calls/root; low-frequency magnitude ranking; S2 adds previous-image regularization lambda .01 before late projection.
- Hard late projection is low-frequency Gerchberg-Saxton Fourier-magnitude replacement, not FAST's image-gradient correction.
- Original NP B22 runner CLIPS negative measured amplitudes to zero in memory. Native FAST uses the original SIGNED saved observations. A16 default preserves NP clipping; signed target is a separately labeled variant.
- FAST native pixel PR has 150 UNet calls/root, cheap FFT/VJP-based local corrections. Its nominal K2 ADMM initial image update is zero-gradient exact arithmetic; the first auxiliary v/u projection makes the second update effective. A16 embeds source-derived effective second correction, without importing the incompatible original FAST environment. Gamma is adapted to NP's VP scheduler as (1-alpha_bar)/alpha_bar: a modeling choice, not a copy of native FAST's EDM grid.

## 3. Frozen 17 arms / GPUs

All arms share eight exposed B22 image IDs 69073 68539 67948 60140 67293 67174 61101 66631, the original FFHQ checkpoint bytes (SHA 81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa) and frozen signed-y bytes (manifest SHA fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490). New fixed root seeds are 16160000 + int(image_id) + rep and are reused across experimental arms. Differing proposal counts induce different later RNG streams even with the same initial seed.

GPU0: lf_native, lf_fast_post, lf_grad_post, lf_fast_replace. Isolate post-selection correction vs ordinary residual-gradient vs replacing late GS projection.

GPU1: s2_native, lf_fast_prescore, lf_fast_both, lf_fast_residual. Correct candidate BEFORE ranking with predeclared sparse checkpoints including the EARLY five-candidate portion. Disentangle ranking improvement versus propagation improvement. This lane has a selected-only negative control in GPU0.

GPU2: lf_sparse, lf_adaptive, lf_beam2, lf_budget2. Test sparse K5, SNR/progress-dependent K1/2/3, a true two-state survivor maintained for six steps, and simple K2 baseline. Record total UNet counts to evaluate matched efficiency (these four arms are NOT automatically budget-matched).

GPU3: lf_500_fast, lf_250_fast, lf_150_fast, lf_250_signed, lf_250_moments. Shorter schedules 500/250/150, one deliberate signed-vs-clipped changed-observation arm, and a cheap standardized residual moment-ranking hypothesis (NOT original DC loss).

Config is immutable within this branch: configs/track_a/a16_np_fast_autopilot_v1.json. After the four-GPU investigation, research with fresh data requires new user/planner approval; B24 confirmation305 is not accessed.

## 4. Mandatory run gates and staged autopilot

A. CPU tests and PLAN, zero GPU:
Run python -m unittest discover -s tests -p test_a16_np_fast_autopilot.py -v from the new checkout with CUDA_VISIBLE_DEVICES empty. Then run module scripts.track_a.a16_autopilot --mode plan --expected-head EXACT_SHA. Plan checks pinned source and clean checkout, hashes model, eight saved y and ground-truth FILES (does not load GT tensors), A15 selector JSON hash, all Python AST. Generates exactly 17 immutable arm configs and A16_CPU_PLAN_PASS.json, plus zero-GPU A16_SAVED_FAST_SELECTOR_FORENSICS.json with per-image rank Spearman and residual gaps. No modification of any A15 or A1 historical output.

B. Native NP exact-replay SMOKE, GPU0 only:
Controller --mode smoke with recorded issue41 lease string A16_NP_FAST_FOUR_GPU_RECORDED_ISSUE41. First it runs ORIGINAL frozen NP 1000-step LF selector for original model/observation/seed, then A16's reconstructed baseline loop under the exact same seed. Guard: max absolute reconstruction difference <=5e-4 or STOP. One completed image 69073 root rep0 saved write-once, GPU0 physical UUID checked. This baseline no-hybrid equivalence gate is more important than any later PSNR result.

C. FOUR-GPU execution:
After successful smoke, user checks all four GPU UUIDs, current idle compute processes, Track B lease priority. Controller --mode execute with same source/lease and --max-hours 8:
  Stage0: 17 arms x 4 pre-exposed diagnostics (69073 61101 67174 66631) x 1 root, reusing smoke baseline. Outcome-screen for mechanics and severe failures.
  Stage1: preserve native LF and S2 controls, promote 2 candidate arms per lane by PRE-REGISTERED development-only score (mean raw PSNR, Bad20 penalty, logged NFE and wall cost), evaluate all 8 exposed images x reps0/1, reusing stage0 receipts.
  Stage2: best mechanism per lane plus native controls, evaluate all 8 exposed x K4 roots. Report GT oracle upper bound and GT-free min signed/min clipped full residual selection per arm, with mean regret and Good25, at charged K4 candidate cost.
One worker per physical GPU, no concurrent jobs on the same GPU. Hardware guard >=90,000 MiB available, <=1024 MiB used and <=10% utilization, exact expected GPU UUID, no existing GPU compute process at launch. Eight-hour aggregate guard and 600-second per root timeout, write-once outputs, STOP on first failed lane and prevent launching new jobs in other lanes. Ctrl+C terminates ONLY the controller's own child process groups; no set -e or shell closure. Partial results remain untouched for forensic diagnosis; no silent automatic retry.

D. Metadata return:
python -m scripts.track_a.a16_pack_metadata (CPU-only) packs per-candidate result JSON, stage summaries, console logs and arm config into A16_COMPLETE_METADATA_RETURN.zip. It excludes tensor/Png/GT measurements and model weights, and prints SHA256 for upload to chat.

## 5. What is recorded

Per candidate: raw oriented RGB8 PSNR primary; rot180 PSNR diagnostic; Good25/Good30/Bad20; exact UNet calls and correction VJPs; correction forward magnitude-operator evaluations; *estimated* score/GS FFT call counts clearly differentiated from measured; CUDA sample event and synchronized wall; model load, whole Python after-import and per-root external subprocess wall; peak GPU reserved memory; original input/checkpoint/source hashes, output PNG/float SHA; raw signed and clipped full forward measurement residual computed AFTER sampling and excluded from inference time. GT loaded ONLY after completing the reconstruction. No runtime GT selector.

Per stage: number of valid roots, per-arm cohort raw mean/min/failures, actual candidate time/NFE. Stage2 also records full-K GT oracle upper PSNR and signed/clipped min-residual-selected PSNR and oracle regret for each image, all 4 roots charged. Promotions use GT only on ALREADY EXPOSED engineering cohort, not deployable adaptive selection.

## 6. Interpretation / kill criteria

Hard stop if native replay mismatch, source/config/model/measurement SHA drift, GPU contention/Track B lease, nonfinite output, occupied incomplete receipt, or root timeout. Kill scientifically weak arms after Stage0; only stage survivors use larger budgets. Even a successful stage2 is development-only. If NP-fast cannot approach FAST's K4 quality-compute frontier at an honestly comparable candidate and UNet budget, no standalone NP method claim is justified. If complementary correction/branching survives, freeze a *new* prospective protocol with independent images and source snapshot before seeking a publication claim. Particle and SNR-adaptive selection are already known; claim must identify a new mechanism with causal controls.

Relevant sources: FAST-DIPS ICLR2026 https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html ; Diffusion Posterior Proximal Sampling https://arxiv.org/abs/2402.16907 ; PFLD https://arxiv.org/abs/2408.13868 ; Distributional Consistency https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html ; DiffFPR https://proceedings.mlr.press/v235/li24bj.html .

The executor has NO direct PAC access. All run receipts must be returned by the user; code commit and issue reservation are NOT evidence that GPU experiments have been run.
