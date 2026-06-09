# Phase retrieval experiments: 2026-06-08

This folder contains launchers for the next NP research step:

1. keep the existing LF/S2 selector as a no-refinement control;
2. test NP-selected candidates plus a small anchored measurement-refinement stage;
3. estimate candidate-generation reliability `p_x` on hard FFHQ images;
4. keep external public solvers as official baselines until their entrypoints are inspected.

Default convention follows the current project setting: **oversample=2** and
**best-of-4** through `SEEDS=100,101,102,103`.

## Files

- `prepare_phase_retrieval_20260608.sh` creates
  `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608`,
  copies the old FFHQ/ImageNet-25 split files, and creates a hard-image split.
- `run_np_selector_ffhq_one_gpu.sh` runs the existing LF-vs-S2 selector control.
- `run_np_refine_ffhq_one_gpu.sh` runs the new NP + anchored refinement benchmark.
- `run_np_refine_hard_ffhq_four_gpu.sh` launches four seed groups on hard FFHQ images.
- `run_np_refine_imagenet_one_gpu.sh` is an ImageNet pilot wrapper.
- `summarize_phase_retrieval_20260608.py` aggregates all `run_level.csv` and
  `selected_image_level.csv` outputs under the dated output root.
- `inspect_external_solvers_20260608.py` scans external DiffFPR/SITCOM_ODE folders
  and writes a manifest of likely official-code entrypoints.

## First commands

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
bash scripts/phase_retrieval_20260608/prepare_phase_retrieval_20260608.sh
```

Run the selector control:

```bash
bash scripts/phase_retrieval_20260608/run_np_selector_ffhq_one_gpu.sh 0 selector_full25_s100_103
```

Run NP + refinement on the same setting:

```bash
bash scripts/phase_retrieval_20260608/run_np_refine_ffhq_one_gpu.sh 0 refine_full25_s100_103
```

Hard-image reliability sweep:

```bash
GPUS="0 1 2 3" NOISES="0,0.01,0.05,0.08,0.10" \
  bash scripts/phase_retrieval_20260608/run_np_refine_hard_ffhq_four_gpu.sh
```

Summarize:

```bash
python scripts/phase_retrieval_20260608/summarize_phase_retrieval_20260608.py \
  --root /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608
```

## What the new refinement runner tests

For each image/noise/config/seed, it runs:

```text
NP LF candidate
NP S2-preprojection candidate
  -> no refinement control
  -> Adam measurement refinement with anchor weights 0, 0.01, 0.05
```

The refinement loss is:

```text
mean((|F_pad(x)| - y)^2)
+ anchor_weight * mean((to01(x) - to01(x_np))^2)
+ tv_weight * TV(to01(x)).
```

This is not intended to be a faithful SITCOM/DAPS/DiffFPR reimplementation.  It
is a minimal test of the hypothesis that NP is a good basin/mode selector and
that local measurement optimization should improve candidates once NP puts them
near a good solution.

## External solvers

Before wrapping official DiffFPR/SITCOM_ODE code, run:

```bash
python scripts/phase_retrieval_20260608/inspect_external_solvers_20260608.py
```

This writes:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/manifests/external_solver_entrypoints.json
```

The reason is practical: public solver repos often use different image
normalization, measurement scaling, checkpoints, and CLI conventions.  For now,
use official external runs as baselines and use the in-repo refinement runner for
algorithm-development experiments.
