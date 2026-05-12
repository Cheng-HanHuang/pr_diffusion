# PR Diffusion: diffusion-based phase retrieval experiments

This repository studies diffusion-prior solvers for phase retrieval, with current emphasis on **FFHQ 256×256 phase retrieval** under DiffFPR/SITCOM-style oversampled Fourier magnitude measurements.

## Main benchmark

FFHQ-25 is the active benchmark. CelebA-HQ experiments are historical/development-focused.

## Current practical NP setting

- `score_radius=0.6`
- `proj_radius=0.2`
- `proj_start=300`
- `soft_k=5`
- `hard_k=1`
- `oversample=2`

At sigma_y=0.05 on FFHQ-25: best-of-4 PSNR ≈ 29.42, SSIM ≈ 0.832, LPIPS ≈ 0.224.

## Core conclusions

1. Small low-frequency projection is essential.
2. Broader projection radius hurts.
3. More candidates do not reliably replace independent restarts.
4. Score design is a key bottleneck.
5. Reliability/failure reduction is the next objective.

## Main scripts

- `scripts/pr_external_difffpr_np_guided_benchmark.py`
- `scripts/pr_external_difffpr_np_benchmark.py`

## Docs

- `docs/progress_report.md`
- `docs/current_experiment_plan.md`
- `docs/runbooks/tmp_runner_record.md`
- `docs/repo_audit_notes.md`
- `docs/historical/`
