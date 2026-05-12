# PR Diffusion: diffusion-prior phase retrieval experiments

This repository studies diffusion-prior solvers for phase retrieval. The current active direction is **FFHQ 256×256 phase retrieval** under DiffFPR/SITCOM-style oversampled Fourier magnitude measurements.

The project goal is not only to maximize one best-of-k benchmark number. The goal is to develop a **reliable phase retrieval method** that produces good reconstructions per run, has controlled failure modes, and can be justified as one coherent solver rather than a post-hoc oracle over many methods.

## Current active benchmark

The active benchmark is FFHQ-25.

```text
data root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

split:
  /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt

guided diffusion FFHQ checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt
```

CelebA-HQ experiments are now historical/development experiments.

## Current practical Noise Picking setting

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

## Main scripts

- `scripts/pr_external_difffpr_np_guided_benchmark.py`
- `scripts/pr_external_difffpr_np_benchmark.py`

## Current docs

- `docs/progress_report.md`
- `docs/current_experiment_plan.md`
- `docs/runbooks/tmp_runner_record.md`
- `docs/repo_audit_notes.md`
- `docs/README.md`
