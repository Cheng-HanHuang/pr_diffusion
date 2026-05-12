---

## 1. Run this from the repo root

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo

git status --short
mkdir -p docs/historical docs/runbooks

STAMP=$(date +%Y%m%d_%H%M%S)

# Archive current docs if present.
for f in \
  docs/progress_report.md \
  docs/current_experiment_plan.md \
  docs/ffhq25_guided_np_pilot_comparison_plan.md
do
  if [[ -f "$f" ]]; then
    base=$(basename "$f" .md)
    git mv "$f" "docs/historical/${base}_archived_${STAMP}.md" || mv "$f" "docs/historical/${base}_archived_${STAMP}.md"
  fi
done
```

---

## 2. Write the new progress report

````bash
cat > docs/progress_report.md <<'MD'
# Progress report: FFHQ phase retrieval, tuned Noise Picking, and SITCOM-ODE comparison

This document records the current project state after the FFHQ-25 guided NP benchmarking, parameter tuning, radius ablations, score-mode experiments, and SITCOM-ODE comparisons.

## Current project objective

The project goal is not merely to beat SITCOM-ODE on one averaged table. The goal is to develop a reliable phase retrieval solver that:

1. produces good reconstructions per run, not only after heavy post-hoc selection;
2. has controlled failure modes;
3. uses a single well-defined algorithmic rule, not an oracle selector between multiple methods unless complementarity can be justified mathematically;
4. is evaluated on FFHQ as the main benchmark, with CelebA-HQ treated as historical/development-only.

## Current main benchmark

Main data/model setting:

- Dataset: FFHQ 25-image split.
- Image resolution: 256.
- Guided diffusion FFHQ checkpoint: `/egr/research-pac/huang248/models/ffhq_10m.pt`.
- Image root: `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`.
- Split file: `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt`.
- Measurement: DiffFPR-style centered FFT magnitude after symmetric zero padding.
- Oversampling: 2.
- Main noise level for method tuning: sigma_y = 0.05.
- Main NP evaluation convention:
  - report image-level best-of-k only when k is explicitly stated;
  - also track all-run mean, median, failure counts, and per-image worst cases.

## Best current NP setting

The best practical NP setting found so far is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
````

The tiny numeric winner in one top-8 confirmation was:

```text
score_radius = 0.2
proj_radius  = 0.2
proj_start   = 500
soft_k       = 8
hard_k       = 1
```

but it was only about 0.016 dB better than the cheaper practical setting, so the practical setting is preferred for ongoing development.

## NP parameter tuning and top-8 confirmation

A broad FFHQ tuning screen tested:

* score radius in a small set including 0.2, 0.4, 0.6, and full-ish;
* projection radius in a small set including 0.2, 0.4, 0.6, and full-ish;
* projection starts including 300, 500, 700;
* candidate schedules including soft/hard settings such as 5/1 and 8/1.

The clear finding was:

```text
proj_radius = 0.2 is essential.
proj_radius = 0.4 or larger is much worse.
```

Top-8 confirmation, using 25 images and 4 seeds at sigma_y=0.05, found:

| Config                                         | Best-of-4 PSNR |   SSIM |  LPIPS | Worst best-of-4 PSNR |
| ---------------------------------------------- | -------------: | -----: | -----: | -------------------: |
| score=0.2, proj=0.2, start=500, soft=8, hard=1 |         29.440 | 0.8320 | 0.2242 |               27.243 |
| score=0.6, proj=0.2, start=300, soft=5, hard=1 |         29.424 | 0.8319 | 0.2241 |               27.185 |

These two settings are effectively tied. The second one is much cheaper and became the main practical setting.

Important interpretation:

* The best-of-4 average is respectable but not enough to beat low-noise SITCOM-ODE.
* The all-run mean is much lower because NP has seed-level failures.
* Therefore, method development should reduce per-run failure probability rather than only increasing best-of-k.

## Projection-radius and radius-schedule ablations

A two-stage projection-radius schedule was tested, motivated by the idea that NP may need small low-frequency projection early but might benefit from broader projection late.

Schedules included:

```text
500:0.2,700:0.4
500:0.2,800:0.4
500:0.2,900:0.4
500:0.2,900:0.72
300:0.2,700:0.4
300:0.2,800:0.4
300:0.2,900:0.4
300:0.2,900:0.72
```

Best schedule result:

```text
score=0.2, start=500, soft=8, hard=1, schedule=500:0.2,900:0.4
best-of-2 PSNR ≈ 26.71
```

Matching constant-radius baseline on same 10 images and same seeds:

```text
score=0.2, proj=0.2, start=500, soft=8, hard=1
best-of-2 PSNR ≈ 29.13
```

Conclusion:

```text
Late broadening of the hard projection radius is decisively harmful in the tested form.
```

Even broadening only at step 900 from radius 0.2 to 0.4 lost roughly 2.4 dB. Full-ish projection at 0.72 was worse.

Mechanistic conclusion:

```text
The diffusion prior tolerates a small low-frequency measurement correction.
Broader hard measurement replacement conflicts with the learned prior, even late.
```

## SITCOM-ODE comparison at sigma_y = 0.05

A same-split SITCOM-ODE run gave:

```text
SITCOM-ODE PSNR ≈ 29.690
SITCOM-ODE SSIM ≈ 0.733
SITCOM-ODE LPIPS ≈ 0.185
```

Current practical NP setting at sigma_y=0.05:

```text
NP best-of-4 PSNR ≈ 29.424
NP SSIM ≈ 0.832
NP LPIPS ≈ 0.224
```

Conclusion:

* NP does not beat SITCOM-ODE in PSNR or LPIPS at sigma_y=0.05.
* NP is much stronger in SSIM.
* NP is close enough that failure-mode analysis matters.

Per-image comparison at sigma_y=0.05:

* SITCOM-ODE had a major weak/failure image around PSNR 21.643.
* NP reconstructed the same image around PSNR 30.141.
* On the remaining non-failure images, SITCOM-ODE was typically stronger.

Interpretation:

```text
NP is not uniformly stronger, but it has complementary failure behavior.
It can avoid some SITCOM-ODE failure cases.
```

This supports failure-recovery or hybrid directions, but does not by itself justify an oracle selector.

## Noise-level comparison against SITCOM-ODE

A noise sweep used NP practical setting:

```text
score_radius=0.6, proj_radius=0.2, start=300, soft=5, hard=1
oversample=2
best-of-4 over 4 seeds
```

Noise levels:

```text
sigma_y ∈ {0.00, 0.01, 0.05, 0.10, 0.20, 0.50}
```

NP raw best-of-4 vs SITCOM-ODE:

| sigma_y | NP raw PSNR | SITCOM-ODE PSNR | NP - SITCOM | NP SSIM | SITCOM SSIM | NP LPIPS | SITCOM LPIPS |
| ------: | ----------: | --------------: | ----------: | ------: | ----------: | -------: | -----------: |
|    0.00 |      32.372 |          35.852 |      -3.480 |   0.924 |       0.935 |    0.125 |        0.067 |
|    0.01 |      33.444 |          35.736 |      -2.292 |   0.947 |       0.933 |    0.100 |        0.067 |
|    0.05 |      29.424 |          29.690 |      -0.266 |   0.832 |       0.733 |    0.224 |        0.185 |
|    0.10 |      24.203 |          23.302 |      +0.901 |   0.626 |       0.411 |    0.491 |        0.395 |
|    0.20 |      19.606 |          17.429 |      +2.177 |   0.400 |       0.168 |    0.731 |        0.629 |
|    0.50 |      13.034 |          11.121 |      +1.913 |   0.143 |       0.036 |    0.834 |        0.777 |

Conclusion:

* SITCOM-ODE dominates low noise.
* NP becomes better in PSNR/SSIM at sigma_y >= 0.10.
* SITCOM-ODE retains better LPIPS at all tested noise levels.
* This supports a noise-robustness story for NP, not a clean-setting SOTA story.

Per-image pattern:

* At sigma_y=0.05, NP wins only a small number of images in PSNR but rescues a major SITCOM failure.
* At sigma_y=0.10, NP wins PSNR on most images.
* At sigma_y=0.20, NP wins PSNR on all images, though absolute quality is low.

## Candidate count ablation

Goal:

```text
Test whether doubled candidates per reconstruction can replace independent restarts.
```

Old setting:

```text
soft=5, hard=1, best-of-4
```

New settings:

```text
soft=10, hard=1, best-of-2
soft=10, hard=2, best-of-2
```

Results at sigma_y=0.05:

| Setting             | Best-of-k | PSNR mean | Median |    Min |  SSIM | LPIPS | Images <20 | Images <25 |
| ------------------- | --------: | --------: | -----: | -----: | ----: | ----: | ---------: | ---------: |
| old soft=5, hard=1  | best-of-2 |    28.531 | 29.280 | 17.528 | 0.808 | 0.249 |          1 |          1 |
| old soft=5, hard=1  | best-of-4 |    29.424 | 29.723 | 27.185 | 0.832 | 0.224 |          0 |          0 |
| new soft=10, hard=1 | best-of-2 |    27.656 | 29.356 | 13.727 | 0.777 | 0.278 |          3 |          3 |
| new soft=10, hard=2 | best-of-2 |    26.855 | 26.824 | 24.902 | 0.790 | 0.194 |          0 |          1 |

Interpretation:

* Doubling candidates did not replace independent restarts.
* soft=10, hard=1 rescued some previous failures but created new catastrophic failures.
* soft=10, hard=2 reduced catastrophic failures but substantially lowered reconstruction quality.
* More candidates can amplify a flawed selection criterion.

Per-image findings for soft=10, hard=1:

Major rescues vs old best-of-2:

```text
00018: +11.47 dB
00028: +4.15 dB
00034: +3.02 dB
00027: +2.88 dB
```

New failures:

```text
00005: -16.30 dB
00000: -16.02 dB
00007: -11.34 dB
```

Conclusion:

```text
More candidates change which images fail, but do not reliably lower failure probability.
Independent restarts remain more reliable than simply increasing candidates.
```

## Score-mode experiments

Motivation:

If more candidates make the result worse, the candidate-selection score is likely not well aligned with final reconstruction quality.

Four score modes were tested:

```text
S1: lf
    original low-frequency magnitude score.

S2: prev_l2
    low-frequency score plus penalty for moving too far from previous x0.

S3: consensus_l2
    low-frequency score plus candidate outlier penalty.

S4: huber_lf
    robust Huber low-frequency score.
```

Setup:

```text
sigma_y=0.05
25 images
2 seeds
score_radius=0.6
proj_radius=0.2
proj_start=300
soft=5
hard=1
```

Results:

| Setting                | Best-of-2 PSNR | Median |    Min |  SSIM | Images <20 | Images <25 |
| ---------------------- | -------------: | -----: | -----: | ----: | ---------: | ---------: |
| S1 original LF         |         28.531 | 29.280 | 17.528 | 0.808 |          1 |          1 |
| S2 prev_l2 lambda=0.25 |         28.560 | 29.280 | 13.828 | 0.810 |          1 |          1 |
| S3 consensus_l2        |         27.176 | 29.280 | 11.898 | 0.768 |          3 |          3 |
| S4 Huber LF            |         27.438 | 29.209 | 13.832 | 0.767 |          3 |          3 |
| old S1 best-of-4       |         29.424 | 29.723 | 27.185 | 0.832 |          0 |          0 |

Conclusion:

* S2 had slight positive signal, but was not robust.
* S3 and S4 were clearly worse in this form.
* Score changes can rescue some images but create other failures.

## Focused S2 lambda sweep

Lambdas tested:

```text
0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15
```

Best static lambda:

```text
lambda = 0.01
```

Results:

| Setting         | Mean PSNR | Median |    Min |  SSIM | Below 20 | Below 25 |
| --------------- | --------: | -----: | -----: | ----: | -------: | -------: |
| S1 LF best-of-2 |    28.531 | 29.280 | 17.528 | 0.808 |        1 |        1 |
| S2 lambda=0.005 |    28.460 | 29.582 | 17.739 | 0.801 |        1 |        2 |
| S2 lambda=0.01  |    28.818 | 29.582 | 19.269 | 0.813 |        1 |        1 |
| S2 lambda=0.02  |    28.706 | 29.280 | 19.624 | 0.809 |        1 |        1 |
| S2 lambda=0.05  |    28.544 | 29.280 | 13.836 | 0.809 |        1 |        1 |
| S1 LF best-of-4 |    29.424 | 29.723 | 27.185 | 0.832 |        0 |        0 |

S2 lambda=0.01 rescues:

```text
00018: 17.528 -> 28.963
00028: 25.877 -> 29.996
00034: 27.596 -> 29.690
```

but creates a new failure:

```text
00007: 28.783 -> 19.269
```

Oracle over lambdas:

```text
mean PSNR ≈ 29.294
median ≈ 29.631
min ≈ 25.103
below20 = 0
below25 = 0
```

Conclusion:

```text
S2 contains useful signal, but a fixed global lambda is not robust enough.
The direction is worth exploring only as part of a single principled adaptive score, not as an oracle over lambdas.
```

## Current methodological conclusion

The evidence supports this view:

```text
Noise Picking is useful as a conservative branch-selection / measurement-guidance principle.
The current simple greedy low-frequency score is fragile.
Small low-frequency projection is crucial.
Broad projection is harmful.
Increasing candidate count amplifies score errors.
Score regularization can rescue some failures but may move failures elsewhere.
NP is more robust than SITCOM-ODE under higher measurement noise in PSNR/SSIM.
```

The project should now prioritize one well-defined, reliable solver rather than chasing an averaged best-of-k table.

MD

````

---

## 3. Write recommended next experiments

```bash
cat > docs/current_experiment_plan.md <<'MD'
# Current experiment plan: reliable phase retrieval solver after FFHQ NP/SITCOM study

## Guiding principle

The next phase should optimize for a single solver that reliably produces good reconstructions. Avoid post-hoc two-way oracle selection unless there is a mathematically justified selector or complementarity theorem.

Primary evaluation should include:

1. all-run mean and median;
2. seed-level failure counts;
3. image-level best-of-1 / best-of-2 / best-of-4 curves;
4. per-image variance across seeds;
5. failure cases and not only averages;
6. raw metrics whenever comparing to methods that do not ambiguity-align.

## Direction A: improve candidate score as one algorithm

Current finding:

- S1 low-frequency score is fragile.
- S2 prev_l2 has useful signal, especially lambda around 0.01, but moves failure cases.
- A fixed global lambda is not reliable enough.

### A1. Timestep-dependent S2

Try:

```text
score_i = LF_score + lambda(i) * previous_state_penalty
````

where lambda decays over the trajectory:

```text
lambda(i) = lambda0 * max(0, 1 - i / proj_start)
```

or only applies before projection:

```text
lambda(i) = lambda0 for i < proj_start
lambda(i) = 0 after proj_start
```

Motivation:

* Early branch choices affect global basin.
* Late regularization can over-constrain details.
* Previous fixed lambda sometimes created late/trajectory failures.

Suggested first grid:

```text
lambda0 in {0.005, 0.01, 0.02}
schedule in {pre_projection_only, linear_decay_to_proj_start}
sigma_y = 0.05
25 FFHQ images
2 seeds first
```

Promote only if:

```text
mean PSNR > 28.8
min PSNR > 25
below20 = 0
below25 = 0
```

### A2. Adaptive lambda from score uncertainty

Idea:

Use prev_l2 penalty only when the LF scores are close or unreliable.

Example:

```text
if (best_lf - second_best_lf) / median_lf is small:
    use prev_l2 regularization
else:
    trust LF score
```

This is a single rule, not ground-truth selection.

Rationale:

* When one candidate clearly wins in measurement score, regularization may be unnecessary.
* When candidates are close, the LF score is noisy and stability should matter.

### A3. Candidate-bank / noise-memory selection

User-proposed direction:

At each timestep, keep the winning noise directions from the previous k steps and sample only:

```text
new_candidates = candidate_num - k
```

fresh candidates. The old noise directions must be rescaled appropriately to the current timestep.

Motivation:

* Some timesteps may contain no good fresh candidate.
* A previously successful noise direction may remain useful locally.
* This may reduce catastrophic branch switches.

First version:

```text
soft=5, hard=1
memory_k in {1,2}
reuse previous winning eps and maybe previous second-best eps
remaining candidates sampled fresh
score mode = LF or LF + weak prev_l2
```

Key implementation concern:

* eps_prev is currently kept as one candidate when num_candidates > 1.
* Extend this to a small `eps_memory` queue.
* At each step, include memory candidates first, then fill fresh candidates.
* Keep memory only if candidate is not stale or if it remains competitive.

Evaluation:

```text
sigma_y = 0.05
25 images
2 seeds
compare all-run and best-of-2 failure counts
```

## Direction B: NP inside SITCOM-ODE

Goal:

Build a single solver, not an oracle between NP and SITCOM.

Proposed algorithm:

```text
Early timesteps:
    use NP-style candidate branch selection with conservative low-frequency score.

Late timesteps:
    continue with SITCOM-ODE / DAPS-style consistency optimization.
```

Motivation:

* SITCOM-ODE is stronger in low-noise clean reconstruction.
* NP avoids some SITCOM failure cases and is more robust at high noise.
* NP's value may be early robust branch selection, while SITCOM's value is late consistency refinement.

Staged implementation:

1. NP warm start:

   * run partial or final NP;
   * save x0 estimate;
   * initialize SITCOM-ODE from this estimate if the SITCOM state parameterization allows it.
2. Partial NP to SITCOM continuation:

   * run NP to an intermediate timestep;
   * map x0 estimate into SITCOM's current ODE/noisy state;
   * continue SITCOM from corresponding step.
3. True NP-in-SITCOM:

   * add candidate branch selection inside SITCOM's early sampler loop;
   * score candidates with low-frequency measurement consistency;
   * use SITCOM's normal optimization later.

Success criterion:

```text
Improve per-run reliability and failure rate, not just best-of-k.
Preserve SITCOM's strong low-noise quality while reducing catastrophic failures.
```

## Direction C: robust measurement weighting

Finding:

* Hard projection beyond radius 0.2 is harmful.
* Late broadening to 0.4 or full is also harmful.
* SITCOM-ODE degrades sharply as measurement noise increases.

Hypothesis:

The solver should not hard-enforce noisy or high-frequency measurement components. Instead, use soft, robust, frequency-dependent weights.

Possible forms:

```text
measurement_weight(r, t, sigma_y)
```

with:

* higher weight at low frequency;
* lower weight at high frequency;
* lower overall measurement weight as sigma_y increases;
* robust residual loss such as Huber/Charbonnier instead of squared magnitude error.

First experiment:

```text
Replace hard low-frequency projection with a soft projection step:
x_new = (1 - eta) * x_prior + eta * projected_x
eta in {0.1, 0.25, 0.5}
radius = 0.2
```

or modify score:

```text
score = weighted robust magnitude residual
```

rather than hard projection.

## Direction D: evaluation standard going forward

For every new method, record:

```text
all-run PSNR mean/median/min
image-level best-of-1, best-of-2, best-of-4
number of runs <20 dB
number of images whose best-of-k <25 dB
per-image worst and failure identities
SSIM and LPIPS for final serious runs
runtime and effective candidate-call count
```

A method should be promoted only if it improves reliability without simply moving failures between images.

MD

````

---

## 4. Update README focus

```bash
cat > README.md <<'MD'
# PR Diffusion: diffusion-based phase retrieval experiments

This repository studies diffusion-prior solvers for phase retrieval, with current emphasis on **FFHQ 256×256 phase retrieval** under DiffFPR/SITCOM-style oversampled Fourier magnitude measurements.

The project goal is not only to maximize one best-of-k benchmark number. The goal is to develop a reliable phase retrieval method that produces good reconstructions per run and has controlled failure modes.

## Current main benchmark

The active benchmark is FFHQ-25:

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
````

CelebA-HQ experiments are now historical/development experiments. FFHQ is the main benchmark for current work.

## Current best NP setting

The current practical Noise Picking setting on FFHQ is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

At sigma_y=0.05 on FFHQ-25, this gives about:

```text
best-of-4 PSNR ≈ 29.42
SSIM ≈ 0.832
LPIPS ≈ 0.224
```

SITCOM-ODE on the same split is stronger in low-noise PSNR/LPIPS but weaker in SSIM and can have complementary failure modes. NP becomes more favorable in PSNR/SSIM at higher measurement noise.

## Important conclusions so far

1. Small low-frequency projection is essential.
2. Projection radius 0.2 works; radius 0.4/full hurts.
3. Late radius broadening still hurts.
4. Increasing candidate count does not reliably replace independent restarts.
5. Candidate selection score is fragile; more candidates can amplify bad score choices.
6. A prev-state regularized score has useful signal but is not yet reliable as a fixed global rule.
7. NP is better interpreted as a conservative branch-selection / measurement-guidance principle than as a finished standalone SOTA solver.
8. The next goal is reliability per reconstruction, not only best-of-k mean.

## Main scripts

External FFHQ guided NP runner:

```bash
python scripts/pr_external_difffpr_np_guided_benchmark.py --help
```

DiffFPR-style helper utilities:

```bash
scripts/pr_external_difffpr_np_benchmark.py
```

Older phase/grid scripts remain in `scripts/` for historical continuity. Many `slurm_neurips_*` scripts correspond to earlier experiment phases and should be treated as historical unless explicitly reused.

## Current docs

```text
docs/progress_report.md
  Current detailed results and conclusions.

docs/current_experiment_plan.md
  Recommended next experiments.

docs/runbooks/tmp_runner_record.md
  Temporary shell runners and command patterns used during FFHQ testing.

docs/historical/
  Archived plans, old progress reports, and prior CelebA/NeurIPS-stage notes.
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

Do not treat `condition_level_summary.psnr_best` as a method-level best-of-k score. That value is usually the single best reconstruction among all images/seeds, not the image-level best-of-k average.

## Repo hygiene notes

The repository may contain historical backup scripts and old experiment plans. Before final release or paper artifact packaging, review and remove accidental backup files such as:

```text
*.bak*
*_patched.py
```

unless they are intentionally archived.

MD

````

---

## 5. Record the `/tmp` runners we used

```bash
cat > docs/runbooks/tmp_runner_record.md <<'MD'
# Temporary runner record for FFHQ NP/SITCOM experiments

This file records the temporary shell runners used during the FFHQ-25 NP tuning and benchmarking phase. They were created under `/tmp` and are not guaranteed to persist. Recreate them from this document if needed.

## Main environment variables

```bash
ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
IMAGE_LIST_FILE=$ROOT/splits/ffhq_available25.txt
GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR
REPO=/egr/research-pac/huang248/pr_diffusion_repo
````

## Common conda setup

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prdiff_ffhq
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
```

## Practical NP setting

```text
score_radius=0.6
proj_radius=0.2
proj_start=300
soft=5
hard=1
oversample=2
```

## Runners used

### `/tmp/run_np_noise_one.sh`

Purpose: run practical NP setting at one noise level.

Key options:

```bash
--seeds 100,101,102,103
--late_start 300
--soft_candidates 5
--hard_candidates 1
--score_radius 0.6
--proj_radius 0.2
--oversample_values 2
--measurement_noise_values "$NOISE_STD"
--alignments raw,rot180,resolve
--clip_noisy_magnitude
--max_images 25
```

Used for sigma_y in:

```text
0.00, 0.01, 0.05, 0.10, 0.20, 0.50
```

### `/tmp/run_np_bestof2_candidates_one.sh`

Purpose: test doubled candidates with best-of-2.

Variants:

```text
soft=10, hard=1
soft=10, hard=2
```

Result:

* soft=10/hard=1 created new catastrophic failures.
* soft=10/hard=2 reduced catastrophic failures but reduced quality.
* Keep old soft=5/hard=1.

### `/tmp/run_np_score_mode_one.sh`

Purpose: test score modes S1-S4.

Variants:

```text
lf
prev_l2
consensus_l2
huber_lf
```

Result:

* `prev_l2` had useful signal.
* `consensus_l2` and `huber_lf` were worse in tested form.

### `/tmp/run_np_s2_lambda_one.sh`

Purpose: sweep `prev_l2` lambda.

Lambdas:

```text
0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15
```

Best static lambda:

```text
lambda=0.01
```

but it still creates a failure, so it should not replace S1 directly.

### `/tmp/run_np_schedule_one.sh`

Purpose: test projection-radius schedules.

Schedules included:

```text
500:0.2,700:0.4
500:0.2,800:0.4
500:0.2,900:0.4
500:0.2,900:0.72
300:0.2,700:0.4
300:0.2,800:0.4
300:0.2,900:0.4
300:0.2,900:0.72
```

Result:

* all tested late broadening schedules were much worse than constant radius 0.2.

## SITCOM-ODE noise sweep command pattern

The SITCOM-ODE runner used:

```bash
REPO=/egr/research-pac/huang248/external/SITCOM_ODE
DATA_ROOT=/egr/research-pac/huang248/external/SITCOM_ODE/dataset/ffhq_available25
GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/external_sitcom_ode_ffhq25

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python posterior_sample.py \
  +data=demo \
  +model=ffhq256ddpm \
  +task=phase_retrieval \
  +sampler=edm_daps \
  save_dir="$OUTDIR" \
  name="$RUN_NAME" \
  num_runs=4 \
  batch_size=5 \
  gpu=0 \
  data.root="$DATA_ROOT" \
  data.resolution=256 \
  data.start_id=0 \
  data.end_id=25 \
  model.model_config.model_path="$GUIDED_MODEL_PATH" \
  task.operator.sigma="$SIGMA" \
  sampler.diffusion_scheduler_config.num_steps=5 \
  sampler.annealing_scheduler_config.num_steps=200 \
  'eval_fn_list=[psnr,ssim,lpips]' \
  save_samples=True \
  save_traj=False \
  wandb=False
```

Important warning:

If using:

```bash
if [[ -d "$OUTDIR/$RUN_NAME" ]]; then skip; fi
```

make sure the run is not accidentally reusing stale output. Prefer unique timestamped names for reruns.

MD

````

---

## 6. Add a repo audit note

```bash
cat > docs/repo_audit_notes.md <<'MD'
# Repo audit notes after FFHQ tuning phase

## Scope

This audit is based on the current local repo after FFHQ guided NP tuning. GitHub search shows many historical scripts and docs, including current README/docs, historical reports, external DiffFPR/SITCOM wrappers, NeurIPS phase scripts, and backup files. The local PAC repo may contain additional untracked or modified files; always check `git status --short`.

## High-priority checks

### 1. Check that local score-mode patches are present

```bash
grep -R "score_mode" -n scripts/pr_external_difffpr_np_benchmark.py scripts/pr_external_difffpr_np_guided_benchmark.py
````

Expected:

* guided runner exposes `--score_mode`, `--score_reg_lambda`, `--score_huber_delta`;
* base runner accepts scoring modes such as `lf`, `prev_l2`, `consensus_l2`, and `huber_lf`.

If these are absent, the local repo has not incorporated the score-mode work.

### 2. Check CUDA-safe complex magnitude

```bash
grep -R "complex_abs_safe" -n scripts/pr_external_difffpr_np_benchmark.py scripts/pr_external_difffpr_np_guided_benchmark.py
grep -R "\.abs()" -n scripts/pr_external_difffpr_np_benchmark.py scripts/pr_external_difffpr_np_guided_benchmark.py
```

Complex FFT magnitude should use `complex_abs_safe` where CUDA/cuFFT had issues.

### 3. Check projection-radius schedule support

```bash
grep -R "proj_radius_schedule\|parse_radius_schedule\|radius_at_step" -n scripts/
```

Expected:

* schedule parser in base benchmark;
* `--proj_radius_schedule` in guided runner;
* scheduled projection radius used inside `noise_picking_reconstruct_oversampled`.

### 4. Remove accidental backup files before final cleanup

Search currently indicates backup files may exist under `scripts/`, such as:

```text
*.bak_score_ablation
*.bak_before_summary_fields
*_patched.py
```

Before release:

```bash
find scripts -maxdepth 1 \( -name "*.bak*" -o -name "*_patched.py" \) -print
```

Then either delete them or move to `docs/historical/code_snapshots/` if intentionally preserved.

### 5. Compile Python scripts

```bash
python -m compileall prdiffusion scripts
```

If compile errors occur in historical scripts that are no longer maintained, record them and consider moving those scripts into a historical folder or marking them as deprecated.

### 6. Search outdated CelebA-first wording

```bash
grep -R "CelebA\|celeba\|NeurIPS\|Phase 10\|Phase 11" -n README.md docs scripts | head -200
```

Not all mentions are wrong, but current README/docs should make clear that FFHQ is the main benchmark and CelebA-HQ is historical/development.

### 7. Check result-analysis scripts

The current analysis scripts created during this phase may live only in local `scripts/`. Decide which should be committed:

```text
analyze_ffhq_np_noise_sweep_nopandas.py
analyze_ffhq_np_bestof2_candidate_ablation_nopandas.py
analyze_ffhq_np_score_mode_s1_s4_nopandas.py
analyze_ffhq_np_s2_lambda_sweep_nopandas.py
analyze_ffhq_np_schedule_screen_nopandas.py
```

These are useful and should probably be kept under `scripts/analysis/` or `scripts/`.

## Conceptual correctness notes

1. Do not interpret `condition_level_summary.psnr_best` as image-level best-of-k. It is usually the single best run among all images/seeds.
2. For best-of-k, group by image first, take best over seeds, then average over images.
3. Raw-vs-aligned comparisons must be explicit.
4. If comparing to SITCOM-ODE, verify whether SITCOM's `num_runs=4` evaluation is best-of-4, averaged, or another convention.
5. Avoid claiming NP beats SITCOM overall. Current evidence supports:

   * NP is competitive/close at sigma_y=0.05;
   * NP is stronger in PSNR/SSIM at higher noise;
   * SITCOM-ODE dominates low noise and LPIPS;
   * NP has complementary failure behavior.

MD

````

---

## 7. Optional cleanup commands

Do not run these until you inspect the files:

```bash
find scripts -maxdepth 1 \( -name "*.bak*" -o -name "*_patched.py" \) -print
````

If they are accidental backups, remove them:

```bash
git rm scripts/*.bak* scripts/*_patched.py
```

If the glob fails because some files do not exist, use:

```bash
find scripts -maxdepth 1 \( -name "*.bak*" -o -name "*_patched.py" \) -print -delete
git add -A
```

---

## 8. Audit and commit

```bash
git status --short

python -m py_compile scripts/pr_external_difffpr_np_benchmark.py
python -m py_compile scripts/pr_external_difffpr_np_guided_benchmark.py

grep -R "score_mode" -n scripts/pr_external_difffpr_np_benchmark.py scripts/pr_external_difffpr_np_guided_benchmark.py
grep -R "proj_radius_schedule" -n scripts/pr_external_difffpr_np_benchmark.py scripts/pr_external_difffpr_np_guided_benchmark.py

git diff -- README.md docs/progress_report.md docs/current_experiment_plan.md docs/runbooks/tmp_runner_record.md docs/repo_audit_notes.md
```

Commit only after checking the diff:

```bash
git add README.md docs/progress_report.md docs/current_experiment_plan.md docs/runbooks/tmp_runner_record.md docs/repo_audit_notes.md docs/historical
git commit -m "Update FFHQ phase retrieval progress and next experiment plan"
```

---
