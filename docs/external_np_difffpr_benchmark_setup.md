# External benchmarking setup: NP vs DiffFPR table setting (FFHQ + ImageNet)

This note sets up an **initial external benchmarking protocol** for Noise Picking (NP), aligned to the numerical setting reported in the DiffFPR paper/review note in `docs/`.

## 1) Paper-aligned target setting (what we reproduce)

The related comparison table setting is:

- dataset: `FFHQ 256x256`, `ImageNet 256x256`
- oversampled Fourier phase retrieval with **oversampling ratio `r^2 = 4.0`**
- measurement noise levels: `sigma_y in {0.00, 0.01, 0.05}`
- metrics in the paper table include PSNR/SSIM/LPIPS

In this repository, we now mirror the same core measurement setting with NP:

- two NP solvers from Phase 10:
  - `np_canonical` (`soft=5`, `hard=1`, `proj_start=400`)
  - `np_fixedk_lateproj` (`soft=5`, `hard=5`, `proj_start=400`)
- same 1000-step denoising trajectory (`np_steps=1000`)
- same oversampling setting (`oversample=4.0` in code, equivalent to `r^2=4.0`)
- same noise-level sweep (`0.00`, `0.01`, `0.05`)

## 2) Pretrained-model adaptation for NP (FFHQ and ImageNet)

The benchmark script now supports explicit paper presets:

- `--paper_preset ffhq` → model `google/ddpm-celebahq-256`
- `--paper_preset imagenet` → model `google/ddpm-ema-256`

And it enforces a shape sanity check:

- if a preset is selected but the loaded model is not `256x256`, the run fails early.

This avoids silently using a mismatched prior resolution for NP.

## 3) How to run the benchmark matrix

### Single run (one dataset / one noise level)

```bash
python scripts/pr_external_difffpr_np_benchmark.py \
  --paper_preset ffhq \
  --data_root /path/to/ffhq \
  --image_list_file /path/to/ffhq_eval_1000.txt \
  --outdir ./outputs/external_np/ffhq_sigma005 \
  --variants np_canonical,np_fixedk_lateproj \
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 \
  --np_steps 1000 \
  --late_start 400 \
  --fixed_k 5 \
  --measurement_noise_std 0.05 \
  --clip_noisy_magnitude
```

### Full FFHQ + ImageNet matrix (paper noise levels)

Use:

```bash
bash scripts/pr_external_difffpr_np_paper_matrix.sh
```

Required environment variables:

- `FFHQ_DATA_ROOT`
- `IMAGENET_DATA_ROOT`

Optional:

- `FFHQ_LIST_FILE`, `IMAGENET_LIST_FILE`
- `OUT_ROOT`, `SEEDS`, `VARIANTS`, `NP_STEPS`, `LATE_START`, `FIXED_K`

## 4) Output files

Each run writes:

- `configs.csv` (all reconstruction and measurement settings)
- `run_level.csv` (per image / per seed metrics)
- `image_level_summary.csv` (image-level mean/median/max aligned PSNR etc.)

## 5) Practical caveats for faithful paper comparison

This setup is intentionally strong for **initial external benchmarking**, but strict apples-to-apples comparison with the paper still depends on missing details:

1. exact image IDs / split protocol used in the paper for FFHQ/ImageNet table,
2. exact pretrained diffusion checkpoint family used by paper authors on ImageNet,
3. any RAAR-side implementation details used inside DiffFPR (damping / iteration internals),
4. exact evaluation details for SSIM/LPIPS preprocessing and ambiguity alignment.

So, we should treat this as:

- **paper-setting-aligned NP benchmarking**, suitable for internal comparison and external trend checking,
- not yet a claim of exact reproduction parity with the published table.

