# PR Diffusion: diffusion-based phase retrieval experiments

This repository studies diffusion-prior solvers for phase retrieval.  The current active direction is **FFHQ 256×256 phase retrieval** under DiffFPR/SITCOM-style oversampled Fourier magnitude measurements.

The project goal is not only to maximize one best-of-k benchmark number.  The goal is to develop a **reliable phase retrieval method** that produces good reconstructions per run, has controlled failure modes, and can be justified as one coherent solver rather than a post-hoc oracle over many methods.

## Current active benchmark

The active benchmark is FFHQ-25.

```text
data root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

split:
  /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt

guided diffusion FFHQ checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt

external SITCOM-ODE repo:
  /egr/research-pac/huang248/external/SITCOM_ODE

external DiffFPR repo/model utilities:
  /egr/research-pac/huang248/external/DiffFPR
```

CelebA-HQ experiments are now historical/development experiments.  FFHQ is the main benchmark for current work.

## Current practical Noise Picking setting

The current practical NP setting on FFHQ is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

At `sigma_y=0.05` on FFHQ-25, this setting gives roughly:

```text
best-of-4 PSNR ≈ 29.424
SSIM ≈ 0.832
LPIPS ≈ 0.224
```

A slightly higher top-8 confirmation setting was:

```text
score_radius = 0.2
proj_radius  = 0.2
proj_start   = 500
soft_k       = 8
hard_k       = 1
best-of-4 PSNR ≈ 29.440
```

but the gain was only about `0.016 dB`, so the cheaper `score=0.6, proj=0.2, start=300, soft=5, hard=1` setting remains the practical baseline.

## Main conclusions from the May 12 update

1. Small low-frequency projection is essential.
2. `proj_radius=0.2` works; `proj_radius=0.4` or broader is much worse.
3. Late radius broadening still hurts: the best tested schedule was about `26.71 dB` best-of-2 versus about `29.13 dB` for the matching constant-radius baseline.
4. SITCOM-ODE remains stronger at low noise in PSNR/LPIPS.  At `sigma_y=0.05`, SITCOM-ODE was about `29.690 dB` PSNR, `0.733` SSIM, and `0.185` LPIPS, while NP was about `29.424 dB` PSNR, `0.832` SSIM, and `0.224` LPIPS.
5. NP has complementary failure behavior: at `sigma_y=0.05`, NP rescued a SITCOM weak/failure image around `21.643 dB` to about `30.141 dB`, while SITCOM was usually stronger on the remaining non-failure images.
6. NP becomes stronger in PSNR/SSIM under higher measurement noise.  In the noise sweep, NP beat SITCOM-ODE in PSNR at `sigma_y >= 0.10`, although SITCOM-ODE retained better LPIPS.
7. Increasing candidate count does not reliably replace independent restarts.  `soft=10, hard=1` rescued some failures but created new catastrophic failures; `soft=10, hard=2` reduced catastrophic failures but reduced quality.
8. The original low-frequency score is fragile.  `prev_l2` scoring contains useful signal, especially around `lambda=0.01`, but a fixed global lambda still moves failures between images.
9. The next phase should pursue one reliable solver: timestep/adaptive S2 scoring, candidate-memory/noise-bank selection, NP-in-SITCOM, or robust soft measurement weighting.

## Main scripts

External FFHQ guided NP runner:

```bash
python scripts/pr_external_difffpr_np_guided_benchmark.py --help
```

DiffFPR-style helper utilities:

```bash
python scripts/pr_external_difffpr_np_benchmark.py --help
```

Recent analysis scripts include:

```text
scripts/analyze_ffhq_np_confirm_top8_nopandas.py
scripts/analyze_ffhq_np_schedule_screen_nopandas.py
scripts/analyze_ffhq_np_bestof2_candidate_ablation_nopandas.py
scripts/analyze_ffhq_np_score_mode_s1_s4_nopandas.py
scripts/analyze_ffhq_np_s2_lambda_sweep_nopandas.py
scripts/compare_np_sitcom_all_metrics_pac_nopandas.py
```

Older phase/grid scripts remain in `scripts/` for historical continuity.  Many `slurm_neurips_*` scripts correspond to earlier experiment phases and should be treated as historical unless explicitly reused.

## Current docs

```text
docs/README.md
  Navigation page for the active docs after the May 12 update.

docs/progress_report.md
  Detailed May 12 results and conclusions.

docs/current_experiment_plan.md
  Recommended next experiments.

docs/runbooks/tmp_runner_record.md
  Temporary shell runners and command patterns used during FFHQ testing.

docs/repo_audit_notes.md
  Script/doc inventory and cleanup checks.

docs/historical/
  Archived plans, old progress reports, and previous CelebA/NeurIPS-stage notes.
```

## Evaluation convention

Always specify whether a result is:

```text
all-run mean over every image/seed;
image-level best-of-2;
image-level best-of-4;
raw alignment;
rot180 alignment;
resolve alignment.
```

Do not treat `condition_level_summary.psnr_best` as a method-level best-of-k score.  That value is usually the single best reconstruction among all images/seeds, not the image-level best-of-k average.

## Repo hygiene notes

The repository may contain historical backup scripts and old experiment plans.  Before final release or paper artifact packaging, review and remove accidental backup files such as:

```text
*.bak*
*_patched.py
```

unless they are intentionally archived.