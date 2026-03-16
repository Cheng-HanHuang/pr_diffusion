# Latest Experiment Record (from latest experiment setting files)

## How "latest" was identified
- From git history, the most recent experiment-setting commit is:
  - `c35a25e` — **"Add low-frequency radius ablation compare script and Slurm job"**.
- Files added/used there:
  - `scripts/slurm_compare_subset5_lowfreq_ablation.sh`
  - `scripts/compare_methods_lowfreq_ablation.py`

So the latest experiment we did is a **low-frequency radius ablation** comparing **SITCOM vs Noise Picking** on a 5-image subset, with repeated seeds per image/radius.

---

## 1) Experiment objective
Evaluate and compare reconstruction quality/time of:
- **SITCOM** (`sitcom_reconstruct`)
- **Noise Picking** (`noise_picking_reconstruct`)

under **low-frequency constrained measurements**, while ablating the low-frequency mask radius:

`radius ∈ {0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}`.

For each image and radius, run multiple random seeds and report PSNR / full-frequency magnitude error / low-frequency magnitude error and runtime.

---

## 2) SLURM experiment setup (explicit)

### 2.1 Cluster resources and scheduling
From `scripts/slurm_compare_subset5_lowfreq_ablation.sh`:
- `--job-name=prdiff_lowfreq5`
- `--partition=general-long-gpu`
- `--nodes=1`
- `--ntasks=1`
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=06:00:00`
- `--gpus=h200:1`
- `--array=0-4%1` (5 tasks, one running at a time)
- Log files:
  - stdout: `logs/prdiff_lowfreq5_%A_%a.out`
  - stderr: `logs/prdiff_lowfreq5_%A_%a.err`

### 2.2 Runtime environment
- Working directory: `${SLURM_SUBMIT_DIR}`
- Creates `logs/`
- Sets HuggingFace caches:
  - `HF_HOME=$HOME/.cache/huggingface`
  - `TRANSFORMERS_CACHE=$HF_HOME`
  - `DIFFUSERS_CACHE=$HF_HOME`
- Conda environment: `dip`

### 2.3 Dataset/model/output paths
- `DATA_ROOT="$HOME/data/prdiff_subset5"`
- `OUT_ROOT="$HOME/out_hpc_compare_lowfreq_subset5"`
- `MODEL_ID="google/ddpm-celebahq-256"`

### 2.4 Image selection and job-array mapping
Image list in script (fixed subset):
1. `00004.jpg`
2. `09375.jpg`
3. `09671.jpg`
4. `10277.jpg`
5. `19500.jpg`

`SLURM_ARRAY_TASK_ID` indexes this list; each task processes one image.

### 2.5 Seeding plan and replicate count
- `BASE_SEED=100`
- `--n_runs 10`
- In Python script, seeds are generated as:
  - `seeds = [base_seed + i for i in range(n_runs)]`
  - i.e., `{100, 101, ..., 109}`.

### 2.6 Slurm-launched command (actual experiment command)
```bash
conda run -n "$CONDA_ENV" python scripts/compare_methods_lowfreq_ablation.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUT_ROOT" \
  --model_id "$MODEL_ID" \
  --n_runs 10 \
  --base_seed "$BASE_SEED" \
  --radius_list "$RADIUS_LIST" \
  --sitcom_outer_steps 20 \
  --sitcom_inner_steps 20 \
  --noise_picking_steps 1000
```

---

## 3) Parameter choices and ablations (comprehensive)

## 3.1 Explicitly ablated parameters
1. **Low-frequency radius** (main ablation)
   - Values: `0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`
   - Applied to:
     - SITCOM `meas_radius`
     - Noise Picking `score_radius` and `proj_radius`

2. **Random seed replicate dimension**
   - Values: 10 runs (`100..109`) per (image, radius)
   - Not an algorithmic ablation knob, but a repeated-trial dimension for variance assessment.

## 3.2 Manually chosen (fixed in this latest run, not swept)
### Shared experiment-level
- `images`: one image per SLURM task from fixed 5-image list
- `data_root`: `$HOME/data/prdiff_subset5`
- `outdir`: `$HOME/out_hpc_compare_lowfreq_subset5`
- `model_id`: `google/ddpm-celebahq-256`
- `n_runs`: `10`
- `base_seed`: `100`

### SITCOM settings in this run
- `num_steps` / outer steps: `20` (via `--sitcom_outer_steps 20`)
- `K` / inner steps per outer step: `20` (via `--sitcom_inner_steps 20`)
- `lr_inner`: `0.05` (script default, not overridden)
- `lam`: `0.1` (script default, not overridden)
- `eta_scale`: `1.0` (script default, not overridden)
- `stop_meas_l2`: `None` (default, no early stop by l2 threshold)
- `backprop_unet`: `True` (hardcoded)
- `inner_optim`: `"adam"` (hardcoded)

### Noise Picking settings in this run
- `num_steps`: `1000` (via `--noise_picking_steps 1000`)
- `num_candidates_soft`: `5` (default, not overridden)
- `num_candidates_hard`: `2` (default, not overridden)
- `proj_start`: `400` (default, not overridden)
- `use_lowfreq_score`: `True` (hardcoded)
- `use_lowfreq_projection`: `True` (hardcoded)
- `score_radius`: set from ablated `radius`
- `proj_radius`: set from ablated `radius`

## 3.3 Defaults/fixed parameters from library/algorithm internals (unstudied here)

### Diffusion/model loading
- UNet loaded by `UNet2DModel.from_pretrained(model_id)`
- Scheduler loaded by `DDPMScheduler.from_pretrained(model_id)`
- Scheduler timestep grid defined by `scheduler.set_timesteps(cfg.num_steps, device=device)` per method

### Device policy
- Uses CUDA if available, else CPU:
  - `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### Input preprocessing
- Ground-truth image is loaded as RGB and resized to `256 x 256`
- Pixel range mapped to `[-1, 1]`

### Seeding mechanics / determinism behavior
- `seed_everything(seed, deterministic=True)` per reconstruction call sets:
  - Python `random`, NumPy, torch CPU and CUDA seeds
  - cuDNN deterministic flags:
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`

### SITCOM internal fixed behavior in this run
- Initial latent sample scale: `init_scale=1.0` (default)
- Logging interval: `log_every=200` (default)
- Loss:
  - Measurement term:
    - If `meas_radius` set (true here), uses **lowfreq l2**, then squares it for optimization objective
  - Regularizer term: `lam * mean((v - x_t)^2)`
- Early stop:
  - `stop_meas_l2=None` and `stop_meas_mse=None` => no early stopping active
- Resampling noise for step transition uses a dedicated generator seeded as `seed + 12345`

### Noise Picking internal fixed behavior in this run
- Start latent sampled as standard normal
- At first iteration uses direct denoiser estimate of `x0_hat`; later iterations use previous candidate result
- Projection details:
  - Projection active only when iteration index `i >= proj_start` (here 400)
  - Projection uses low-frequency magnitude replacement with `num_iter=1` and `eps=1e-8`
- Candidate schedule:
  - before `proj_start`: soft candidate count (`5`)
  - after `proj_start`: hard candidate count (`2`)
- Logging interval: `log_every=100` (default)

## 3.4 Evaluation outputs (what is compared)
Per (image, radius, seed), script computes and stores:
- `sitcom_psnr`, `noise_picking_psnr`
- `sitcom_magerr_l2`, `noise_picking_magerr_l2` (full-frequency magnitude L2)
- `sitcom_lowfreq_magerr_l2`, `noise_picking_lowfreq_magerr_l2` (radius-specific low-freq L2)
- `sitcom_time_s`, `noise_picking_time_s`
- reconstructed image paths for both methods

Also stores config snapshots in `configs_*.csv` under each radius folder.

---

## 4) Method comparison framing in this experiment
- **Compared methods:** SITCOM vs Noise Picking.
- **Shared measurement target:** Fourier magnitude of the same ground-truth image.
- **Shared randomization protocol:** same seed list for both methods at each setting.
- **Main study axis:** low-frequency radius.
- **Secondary setup axis:** 5 fixed image identities as separate SLURM array tasks.
- **Primary readouts:** quality (PSNR, magnitude consistency) and runtime.

---

## 5) Suggested structure for future comprehensive studies
To make this line of experiments more systematic, keep this exact 4-level hierarchy:
1. **Method:** {SITCOM, NoisePicking}
2. **Data slice:** image ID (or dataset split)
3. **Ablation knobs:** e.g., radius, steps, candidate counts, lr, lambda
4. **Seed replicate:** base-seed offset

And log every run with:
- full config dump (already done here per-radius and in summary csv)
- software versions (torch/diffusers/cuda) and git commit hash
- aggregate mean/std over seeds per (method, image, radius)

This will support structural and comprehensive experimental analysis across algorithmic and systems-level factors.
