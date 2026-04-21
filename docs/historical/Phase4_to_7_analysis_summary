# Phase 4–7 analysis summary for `pr_diffusion` (updated)

This note summarizes the uploaded CSV outputs together with the current repo logic in:

- `docs/neurips_prdiffusion_experiment_plan.md`
- `docs/progress_report.md`
- `scripts/neurips_grid_experiments.py`
- `scripts/neurips_postprocess_grid.py`
- `scripts/neurips_canonical_compare.py`
- `prdiffusion/algorithms/noise_picking.py`
- `prdiffusion/algorithms/sitcom.py`

The goal of this updated note is to make the summary **self-contained** by recording the **exact settings used in Phases 4–7**, clarifying what **score** and **projection** mean in the Phase 5 mechanism ablation, and then interpreting the resulting tables.

---

## 1. Global settings shared across Phases 4–7

From the launcher script `scripts/pac_launch_phases_4_7_bg.sh`, the following were shared across the runs:

- `RADIUS = 0.5`
- `SEEDS = 100,101,102,103,104,105,106,107,108,109`
- model id left at default: `google/ddpm-celebahq-256`

Split files:

- **Phase 4**: `validation_25.txt`
- **Phase 5**: `validation_25.txt`
- **Phase 6**: `test_50.txt`
- **Phase 7**: `test_20.txt`

So all four phases used:

- low-frequency radius `r = 0.5`
- 10 restarts per image

---

## 2. Exact settings used in each phase

## Phase 4: budget study

This phase used:

- script: `scripts/neurips_grid_experiments.py`
- mode: `budget`
- split: `validation_25`
- images: 25
- seeds: 10 per image
- radius: `0.5`
- methods: **both SITCOM and Noise Picking**

### Phase 4 Noise Picking settings

In `budget` mode, the NP step counts are swept over:

- `num_steps ∈ {250, 500, 750, 1000}`

All other NP parameters stayed at the grid-runner defaults:

- `score_radius = 0.5`
- `proj_radius = 0.5`
- `proj_start = 400`
- `num_candidates_soft = 5`
- `num_candidates_hard = 2`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

Therefore the four NP Phase 4 settings were:

1. `(num_steps=250, soft=5, hard=2, proj_start=400, score on, projection on)`
2. `(num_steps=500, soft=5, hard=2, proj_start=400, score on, projection on)`
3. `(num_steps=750, soft=5, hard=2, proj_start=400, score on, projection on)`
4. `(num_steps=1000, soft=5, hard=2, proj_start=400, score on, projection on)`

### Phase 4 SITCOM settings

In `budget` mode, the SITCOM compute pairs are swept over:

- `(num_steps, K) ∈ {(20,20), (50,10), (100,5)}`

All other SITCOM parameters stayed at the grid-runner defaults:

- `lr_inner = 0.05`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `meas_radius = None` (unmasked)

Therefore the three SITCOM Phase 4 settings were:

1. `(outer=20, inner=20, lr=0.05, lam=0.1, eta=1.0, init=1.0, unmasked)`
2. `(outer=50, inner=10, lr=0.05, lam=0.1, eta=1.0, init=1.0, unmasked)`
3. `(outer=100, inner=5, lr=0.05, lam=0.1, eta=1.0, init=1.0, unmasked)`

### Phase 4 results

#### Noise Picking
| config | avg_image_mean_psnr | avg_image_median_psnr | avg_image_max_psnr | avg_image_mean_mag_l2 | avg_image_mean_runtime_s |
|---|---:|---:|---:|---:|---:|
| np_steps=250 | 12.346 | 12.116 | 15.562 | 76.329 | 17.116 |
| np_steps=500 | 14.398 | 14.451 | 18.866 | 3.883 | 30.225 |
| np_steps=750 | 14.971 | 14.271 | 23.629 | 3.741 | 37.137 |
| np_steps=1000 | 19.651 | 20.232 | 26.359 | 3.604 | 44.061 |

#### SITCOM
| config | avg_image_mean_psnr | avg_image_median_psnr | avg_image_max_psnr | avg_image_mean_mag_l2 | avg_image_mean_runtime_s |
|---|---:|---:|---:|---:|---:|
| sitcom=20x20 | 10.677 | 9.878 | 17.942 | 7.311 | 15.169 |
| sitcom=50x10 | 12.026 | 10.743 | 22.681 | 6.128 | 19.892 |
| sitcom=100x5 | 13.784 | 12.487 | 25.339 | 7.945 | 20.777 |

### Phase 4 read

- NP at 250 steps is not just worse; it is qualitatively broken on measurement error.
- 500 steps already restores measurement consistency strongly.
- 1000 steps is the best NP point by a clear margin.
- SITCOM improves as the outer-step budget increases.

### Important implementation caveat for Phase 4

This is **not** a pure step-budget study for Noise Picking.

Because `proj_start = 400` stayed fixed:

- `num_steps = 250` never enters the late projection regime at all.
- `num_steps = 500` enters projection only near the end.

So the Phase 4 NP curve mixes two effects:

1. **less total compute**, and
2. **less time spent in the projection regime**.

---

## Phase 5: mechanism ablation

This phase used:

- script: `scripts/neurips_grid_experiments.py`
- mode: `mechanism`
- split: `validation_25`
- images: 25
- seeds: 10 per image
- radius: `0.5`
- method: **Noise Picking only**

### Phase 5 fixed NP parameters

These stayed fixed across the mechanism variants:

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `proj_start = 400`
- `num_candidates_soft = 5`
- `num_candidates_hard = 2`

### Phase 5 four mechanism settings

1. **full**
   - `use_lowfreq_score = True`
   - `use_lowfreq_projection = True`

2. **score_only**
   - `use_lowfreq_score = True`
   - `use_lowfreq_projection = False`

3. **projection_only**
   - `use_lowfreq_score = False`
   - `use_lowfreq_projection = True`

4. **no_masking**
   - `use_lowfreq_score = False`
   - `use_lowfreq_projection = False`

### What “score” means here

In Noise Picking, each step samples several candidate noises and converts them into candidate reconstructions `x0_hat`.
The method then **chooses** one candidate according to a measurement-consistency score.

- When **score is on**, candidate selection uses the **low-frequency masked magnitude error**.
- When **score is off**, candidate selection uses the **full magnitude error**.

So **score** is a **selection rule**.
It decides **which candidate trajectory to keep**.
It does **not** directly overwrite the reconstruction.

### What “projection” means here

Later in the denoising process, once the iteration reaches `proj_start`, Noise Picking may explicitly enforce low-frequency measurement consistency by replacing the Fourier magnitude inside the mask while keeping phase.

- When **projection is on**, the algorithm applies this late masked magnitude-enforcement step.
- When **projection is off**, this explicit correction never happens.

So **projection** is a **hard correction / enforcement step**.
It directly modifies the current reconstruction estimate.

### Phase 5 results

| variant | avg_image_mean_psnr | avg_image_median_psnr | avg_image_max_psnr | avg_image_mean_mag_l2 | avg_image_mean_lowfreq_mag_l2 | avg_image_mean_runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| full | 19.651 | 20.232 | 26.359 | 3.604 | 3.108 | 44.134 |
| projection_only | 19.610 | 20.350 | 25.517 | 3.596 | 3.103 | 43.701 |
| score_only | 11.115 | 11.126 | 13.397 | 112.698 | 112.674 | 44.000 |
| no_masking | 10.836 | 10.906 | 13.176 | 114.526 | 114.500 | 43.581 |

### Phase 5 read

- Masked projection is the main mechanism.
- Masked score alone does not rescue the method.
- Full and projection-only are nearly tied on average.
- Score appears to provide, at most, a secondary effect in some better trajectories rather than the main gain.

### Important caveat for Phase 5

This phase used the `mechanism`-mode defaults from `neurips_grid_experiments.py`, so it is **not** exactly the later frozen canonical setting.
In particular, it used:

- `num_candidates_soft = 5`
- `num_candidates_hard = 2`
- `proj_start = 400`

rather than the canonical main-comparison NP setting with `hard = 1`.

---

## Phase 6: main held-out benchmark (`test_50`, canonical)

This phase used:

- script: `scripts/neurips_canonical_compare.py`
- split: `test_50`
- images: 50
- seeds: 10 per image
- radius list: `{0.5}`
- methods:
  - **Noise Picking masked**
  - **SITCOM unmasked**

### Phase 6 SITCOM settings

- `sitcom_variant = unmasked`
- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `meas_radius = None`
- `backprop_unet = True`
- `inner_optim = adam`

### Phase 6 Noise Picking settings

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `proj_start = 400`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

### Phase 6 results

| method | avg_image_mean_psnr | avg_image_median_psnr | avg_image_max_psnr | avg_image_mean_full_mag_l2 | avg_image_mean_lowfreq_mag_l2 | avg_image_mean_runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| Noise Picking masked | 19.104 | 18.874 | 28.583 | 3.599 | 3.191 | 36.060 |
| SITCOM unmasked | 13.526 | 12.071 | 23.394 | 7.539 | 7.379 | 15.162 |

Image win rates for Noise Picking over SITCOM:

- mean PSNR: `42 / 50`
- median PSNR: `40 / 50`
- max PSNR (best-of-10): `35 / 50`

Average deltas (NP − SITCOM):

- avg image mean PSNR: `+5.579 dB`
- avg image median PSNR: `+6.804 dB`
- avg image max PSNR: `+5.188 dB`

### Phase 6 read

- This is the strongest headline result.
- Noise Picking is substantially better in quality and consistency than unmasked SITCOM on the held-out test set.
- The runtime cost is about `2.38×` higher per restart.

---

## Phase 7: secondary masking ablation (`test_20`)

This phase used:

- script: `scripts/neurips_canonical_compare.py`
- split: `test_20`
- images: 20
- seeds: 10 per image
- radius list: `{0.5}`
- methods:
  - **Noise Picking masked**
  - **SITCOM masked**

### Phase 7 SITCOM settings

Same canonical SITCOM configuration as Phase 6, except now masking is enabled:

- `sitcom_variant = masked`
- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `meas_radius = 0.5`
- `backprop_unet = True`
- `inner_optim = adam`

### Phase 7 Noise Picking settings

Same canonical NP setting as Phase 6:

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `proj_start = 400`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

### Important note for Phase 7 interpretation

Phase 7 by itself contains only:

- **SITCOM masked**
- **Noise Picking masked**

To form the intended three-row masking-ablation table, combine it with the Phase 6 unmasked SITCOM results restricted to the same 20 images.

### Combined `test_20` table

| method | avg_image_mean_psnr | avg_image_median_psnr | avg_image_max_psnr | avg_image_mean_full_mag_l2 | avg_image_mean_lowfreq_mag_l2 | avg_image_mean_runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| SITCOM unmasked (Phase 6 common-20) | 12.412 | 11.491 | 20.992 | 7.796 | 7.608 | 15.163 |
| SITCOM masked (Phase 7) | 12.782 | 11.116 | 23.344 | 5.658 | 5.306 | 15.590 |
| Noise Picking masked (Phase 7) | 16.338 | 15.349 | 25.820 | 3.794 | 3.290 | 35.227 |

### Phase 7 read

- Adding masking to SITCOM improves measurement consistency substantially.
- SITCOM masked modestly improves mean PSNR over SITCOM unmasked on this 20-image set.
- SITCOM masked also improves best-of-10 PSNR.
- Noise Picking masked still remains best overall.

### Reproducibility check

On the overlapping 20 images, the Noise Picking metrics match between Phase 6 and Phase 7 up to runtime jitter.
That is a good sign that the canonical comparison pipeline is stable.

---

## 3. Main conclusions

1. **Phase 6 strongly supports the main paper claim**: masked Noise Picking beats unmasked SITCOM on the held-out benchmark.
2. **Phase 5 shows the main mechanism is late masked projection**, not masked scoring by itself.
3. **Phase 7 suggests masking also helps SITCOM**, but not enough to close the gap to Noise Picking.
4. **Phase 4 should be interpreted carefully**, because fixed `proj_start = 400` makes the NP budget curve partly a projection-budget curve.
5. For a final paper presentation, the natural role of each phase is:
   - **Phase 6**: main headline table
   - **Phase 5**: mechanism table
   - **Phase 7**: secondary masking ablation
   - **Phase 4**: runtime-quality study with explicit caveat about absolute `proj_start`

---

## 4. One compact paper-style takeaway

A concise way to summarize the current evidence is:

> Under the current CelebA-HQ 256 phase-retrieval protocol at radius `r=0.5`, masked Noise Picking substantially outperforms unmasked SITCOM on the held-out test set. Mechanism ablation indicates that the dominant source of the gain is the late masked projection step, while masked candidate scoring alone contributes little by itself. Masking also improves SITCOM, but the masked SITCOM variant still remains below masked Noise Picking.
