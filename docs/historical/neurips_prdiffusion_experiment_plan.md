# prdiffusion NeurIPS Experiment Plan

This document is a concrete execution plan for the current **prdiffusion** project.
It is written to be followed directly as a working checklist.

The current codebase is built around:
- a pretrained diffusion prior (`google/ddpm-celebahq-256`)
- magnitude-only Fourier measurements
- two reconstruction methods: **SITCOM** and **Noise Picking**

The intended paper story is:

> **Noise Picking uses masked measurement consistency as part of the method itself, softly before projection starts and harder after that, and should be compared primarily against unmasked SITCOM.**

This means the headline comparison for the paper is:

- **Noise Picking masked**
- **SITCOM unmasked**

while

- **SITCOM masked**

is a **secondary ablation**, not the main baseline.

---

## 1. Main goals

The goal, by the end of the month, is to have:

1. a frozen evaluation protocol
2. a canonical comparison script matching the paper story
3. a validation-based choice of low-frequency radius
4. a held-out benchmark on the current CelebA-HQ 256 setting
5. restart-aware evaluation metrics
6. two to three focused ablation studies
7. a paper draft with a finished experimental section and figure/table plan

The goal is **not** to finish everything possible.
The goal is to finish a clean, defensible first version of the paper.

---

## 2. Core paper claim

The paper should be organized around a simple, clear claim:

> **Noise Picking improves diffusion-based phase retrieval by using masked measurement consistency to select better reconstruction trajectories.**

This should be supported in three ways:

1. **It works better** than unmasked SITCOM on the current benchmark setting.
2. **Masking is essential** to the method, rather than just a small implementation detail.
3. **The best radius depends on the reporting mode**: single-run performance and best-of-R performance may prefer different radii.

---

## 3. Very important protocol decision: single-run vs best-of-R

Phase retrieval papers often report the **best reconstruction over multiple random restarts**.
That is reasonable, but it is **not the same thing** as reporting average behavior from random initialization.

For this project, all future experiments should report **both**:

### 3.1 Single-run performance
For each image, over the fixed seed set, compute:
- mean PSNR
- median PSNR
- std PSNR
- mean full measurement error
- mean low-frequency measurement error when relevant
- mean runtime

This tells us how the method behaves from a random initialization.

### 3.2 Best-of-R performance
For each image, over the same fixed seed set, compute:
- best PSNR over `R=10` restarts
- best measurement error if desired
- total runtime for those 10 restarts, or at least clearly note the restart budget

This tells us how the method behaves when the experimenter is allowed multiple random reconstruction attempts.

### 3.3 Non-negotiable rule
The paper must **never silently mix** these two reporting modes.
Every table and figure should clearly state whether it is:
- mean over seeds
- median over seeds
- best-of-10 over seeds

---

## 4. Canonical experimental protocol

### 4.1 Dataset setting
Stay on the current **CelebA-HQ 256 / face-prior** setting first.
Do not move to additional datasets until the current setting is complete.

### 4.2 Image splits
Create three fixed image lists:

- **dev set**: 10 images
- **validation set**: 25 images
- **test set**: 50 images

Rules:
- tune only on dev/validation
- never tune on the test set
- keep the split files in the repo or experiment folder
- for all images used here, in addition to the five images being used in exisiting experiemtns, use only images 00000.jpg to 05401.jpg because the dataset we have is only partial.

### 4.3 Seeds
Use the same seed list everywhere:

```text
100, 101, 102, 103, 104, 105, 106, 107, 108, 109
```

This gives `R=10` restarts for best-of-R reporting.

### 4.4 Metrics
Record these for every run:
- image id
- method
- seed
- radius if relevant
- PSNR
- full magnitude error
- low-frequency magnitude error if relevant
- runtime
- all important config parameters

### 4.5 Headline comparison policy
For the paper’s main comparison:

- **Noise Picking** always uses masking
- **SITCOM** does **not** use masking

For the secondary ablation:

- include **SITCOM masked**

---

## 5. One required code cleanup

Before large experiments, create a canonical comparison script for the paper.

### 5.1 Desired logic

#### Noise Picking
Use:
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`
- `score_radius = r`
- `proj_radius = r`

#### SITCOM main baseline
Use:
- `meas_radius = None`

#### SITCOM masking ablation
Use:
- `meas_radius = r`

### 5.2 Why this is necessary
The current repo has two mismatched experiment stories:

- the current `compare_methods_no_lowfreq.py` disables low-frequency score/projection for Noise Picking
- the low-frequency ablation script masks both methods at the same radius

Neither one exactly matches the intended paper claim.

### 5.3 Output format
The canonical script should save:

1. **run-level CSV**
2. **image-level aggregation CSV**

The image-level CSV should include, for every image and method:
- mean PSNR over 10 seeds
- median PSNR over 10 seeds
- max PSNR over 10 seeds
- mean measurement errors
- mean runtime

This will save a huge amount of later analysis time.

---

## 6. Experiment plan overview

The month is divided into seven phases:

0. sanity rerun on the original 5 images
1. radius validation
2. light SITCOM tuning
3. Noise Picking schedule tuning
4. runtime / step-budget study
5. mechanism ablation
6. main held-out benchmark
7. secondary masking ablation

If time becomes tight, the priority order is:

1. canonical script
2. radius validation
3. light SITCOM tuning
4. main held-out benchmark
5. mechanism ablation
6. runtime figure
7. secondary masking ablation

---

## 7. Phase 0: sanity rerun on the original 5 images

### Purpose
Confirm that the new canonical comparison script reproduces the earlier preliminary signal.

### Images
Use the original 5-image subset:
- `00004.jpg`
- `09375.jpg`
- `09671.jpg`
- `10277.jpg`
- `19500.jpg`

### Methods
- Noise Picking masked
- SITCOM unmasked

### Settings
- seeds: 10
- radii: `{0.1, 0.2, 0.5}`

### Reconstruction count
- Noise Picking: `5 × 10 × 3 = 150`
- SITCOM: `5 × 10 = 50`
- total = `200`

### What to check
We expect the same general pattern as the preliminary CSVs:
- `0.2` strong by mean/median performance
- `0.5` competitive or strongest by best-of-10 performance
- masking improves Noise Picking substantially relative to the no-lowfreq runs

### Deliverable
A small summary table with:
- mean PSNR by radius
- median PSNR by radius
- best-of-10 PSNR by radius

Do not move on until this sanity check is consistent with the earlier results.

---

## 8. Phase 1: radius validation

### Purpose
Choose the radius or radii to use in the paper.

### Images
Use the **25-image validation set**.

### Methods
- Noise Picking masked
- SITCOM unmasked as reference

### Settings
- seeds: 10
- radii: `{0.1, 0.2, 0.5}`

### Reconstruction count
For the 3-radius version:
- Noise Picking: `25 × 10 × 3 = 750`
- SITCOM: `25 × 10 = 250`
- total = `1000`

### Required metrics
Aggregate per radius:
- average of image-level mean PSNR
- average of image-level median PSNR
- average of image-level best-of-10 PSNR
- average full magnitude error
- average runtime

### Decision rule
Choose:
- one **single-run radius** based on mean/median behavior
- one **best-of-10 radius** based on best-of-10 behavior

It is perfectly acceptable if these differ.
That is itself an experimental finding.

### Expected likely outcome
Based on the current preliminary 5-image study:
- `r = 0.2` is likely the best robust default for single-run reporting
- `r = 0.5` is likely a strong candidate for best-of-10 reporting

### Deliverable
**Table: validation radius selection**

Columns:
- radius
- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- runtime

---

## 9. Phase 2: light SITCOM tuning

### Purpose
Prevent the baseline from being weak due to poor tuning.

### Important constraint
Do **not** spend too much time tuning SITCOM.
Tune enough to be fair, then freeze it.

### Images
Use the **10-image dev set**.

### 9.1 Learning-rate sweep
Use the existing LR sweep style.

Test:
- `lr_inner ∈ {0.02, 0.05, 0.1}`

Keep fixed:
- `K = 20`
- `lam = 0.1`
- current default step setup

Seeds:
- 5 seeds per image

Count:
- `10 × 5 × 3 = 150`

### 9.2 Noise/init sweep
Test:
- `eta_scale ∈ {0.5, 1.0}`
- `init_scale ∈ {0.75, 1.0, 1.25}`

Keep fixed:
- best LR from the previous sweep

Seeds:
- 5 seeds per image

Count:
- `10 × 5 × 6 = 300`

### Total count
- `450`

### Output
Choose one frozen unmasked SITCOM default.

### Important rule
After this, do not keep changing SITCOM settings unless something is obviously broken.

---

## 10. Phase 3: Noise Picking schedule tuning

### Purpose
Tune the staged candidate design without exploding the search space.

### Images
Use **10 validation images**.

### Fixed inputs
Use:
- the chosen validation radius from Phase 1
- 5 seeds (`100,101,102,103,104`) to match grid-runner defaults

### 10.1 Soft candidate count
Test:
- `num_candidates_soft ∈ {3, 5, 7}`

Keep fixed:
- `num_candidates_hard = 2`
- `proj_start = 400`

Count:
- `10 × 5 × 3 = 150`

### 10.2 Hard candidate count
Test:
- `num_candidates_hard ∈ {1, 2, 3}`

Keep fixed:
- best soft count from 10.1

Count:
- `10 × 5 × 3 = 150`

### 10.3 Projection start / stage-switch timing
Test:
- `proj_start ∈ {200, 400, 600}`

Keep fixed:
- best soft/hard counts

Count:
- `10 × 5 × 3 = 150`

### Total count
- `450`

### Deliverable
Choose one default candidate schedule.
Most likely the final values will stay near the current code defaults unless the data strongly says otherwise.

---

## 11. Phase 4: runtime / step-budget study

### Purpose
Determine the practical default compute budget and create the paper’s quality-runtime figure.

### Images
Use **10 validation images**.

### 11.1 Noise Picking step budget
Test:
- `num_steps ∈ {250, 500, 750, 1000}`

Use:
- tuned radius
- tuned candidate schedule
- 5 seeds (`100,101,102,103,104`)

Count:
- `10 × 5 × 4 = 200`

### 11.2 SITCOM budget curve
Test three SITCOM compute settings, for example:
- `(outer, inner) = (20, 20)`
- `(50, 10)`
- `(100, 5)`

Use:
- frozen SITCOM settings otherwise
- 5 seeds (`100,101,102,103,104`)

Count:
- `10 × 5 × 3 = 150`

### Total count
- `350`

### Deliverables
Produce:
- PSNR vs runtime figure
- best-of-10 PSNR vs total runtime figure, if possible
- a small runtime-budget comparison table

This is one of the most important practical experiments in the paper.

---

## 12. Phase 5: mechanism ablation

### Purpose
Show that the masked score / masked projection structure matters.

### Images
Use **10 validation images**.

### Seeds
- 5 (`100,101,102,103,104`) to match grid-runner defaults

### Variants
Run these four Noise Picking variants:

1. **full method**
   - masked score on
   - masked projection on

2. **score-only**
   - masked score on
   - masked projection off

3. **projection-only**
   - masked score off
   - masked projection on

4. **no masking**
   - masked score off
   - masked projection off

### Count
- `10 × 5 × 4 = 200`

### Deliverable
**Mechanism ablation table**

Columns:
- mean PSNR
- best-of-10 PSNR
- full mag error
- runtime

This is a core paper experiment because it supports the central claim that masking is methodologically important.

---

## 13. Phase 6: main held-out benchmark

### Purpose
Generate the paper’s headline result.

### Images
Use the **50-image test set**.

### Methods
- Noise Picking masked
- SITCOM unmasked

### Seeds
- 10

### Count
- `50 × 10 × 2 = 1000`

### What to report
For each method, report:
- image-level mean PSNR averaged over images
- image-level median PSNR averaged over images
- image-level best-of-10 PSNR averaged over images
- full magnitude error
- runtime
- image win rate over the other method

### Headline table
This is the most important table in the paper.

Rows:
- SITCOM unmasked
- Noise Picking masked

Columns:
- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- runtime
- image win rate

---

## 14. Phase 7: secondary masking ablation

### Purpose
Answer the reviewer question:

> What happens if SITCOM also uses masking?

### Images
Use **20 test images**.

### Methods
- SITCOM unmasked
- SITCOM masked
- Noise Picking masked

### Seeds
- 10

### Count
- `20 × 10 × 3 = 600`

### Deliverable
A secondary comparison table with:
- mean PSNR
- best-of-10 PSNR
- full mag error
- low-frequency mag error
- runtime

### Important note
This is secondary.
If time is tight, finish the main benchmark first.

---

## 15. Total reconstruction count

Approximate total if all phases are run:

- Phase 0: 200
- Phase 1: 1000
- Phase 2: 450
- Phase 3: 450
- Phase 4: 350
- Phase 5: 200
- Phase 6: 1000
- Phase 7: 600

**Total: 4150 reconstructions**

This is large but manageable for a month-scale HPC effort if the workflow is organized well.

### Minimum viable version
If the schedule gets tight, the minimum viable path is:
- Phase 0
- Phase 1
- Phase 2
- Phase 5
- Phase 6

That is enough for a real first draft.

---

## 16. How to aggregate results

All experiment summaries should be computed in two layers.

### 16.1 Run-level summary
Across all runs:
- mean PSNR
- std PSNR
- mean full mag error
- mean lowfreq mag error if relevant
- mean runtime

### 16.2 Image-level summary
For each image over the 10 seeds:
- mean PSNR
- median PSNR
- max PSNR
- mean measurement errors
- mean runtime

Then average these image-level summaries across images.

### 16.3 Additional recommended statistics
Also compute:
- fraction of images where NP mean PSNR > SITCOM mean PSNR
- fraction of images where NP max PSNR > SITCOM max PSNR
- average gain on images where NP wins
- average loss on images where NP loses

These are often more informative than only showing a pooled average.

---

## 17. Exact tables to produce

### Table 1: validation radius selection
Rows:
- `r = 0.1`
- `r = 0.2`
- `r = 0.5`

Columns:
- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- runtime

### Table 2: main held-out comparison
Rows:
- SITCOM unmasked
- Noise Picking masked

Columns:
- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- runtime
- image win rate

### Table 3: mechanism ablation
Rows:
- full NP
- score-only NP
- projection-only NP
- no-masking NP

Columns:
- mean PSNR
- best-of-10 PSNR
- full mag error
- runtime

### Table 4: secondary masking ablation
Rows:
- SITCOM unmasked
- SITCOM masked
- Noise Picking masked

Columns:
- mean PSNR
- best-of-10 PSNR
- full mag error
- lowfreq mag error
- runtime

### Table 5: runtime-budget comparison
Rows:
- selected Noise Picking budgets
- selected SITCOM budgets

Columns:
- compute setting
- mean PSNR
- best-of-10 PSNR
- runtime

---

## 18. Exact figures to produce

### Figure 1: method overview
Show:
- magnitude measurement input
- candidate generation
- masked candidate scoring
- late masked projection
- final reconstruction output

### Figure 2: radius selection curve
x-axis:
- radius

y-axis:
- mean PSNR
- median PSNR
- best-of-10 PSNR

This should show that single-run and best-of-R reporting may prefer different radii.

### Figure 3: quality vs runtime
x-axis:
- runtime

y-axis:
- PSNR

Include both Noise Picking budget points and SITCOM budget points.

### Figure 4: restarts curve
x-axis:
- number of restarts `R`

y-axis:
- best-of-R PSNR

Plot both methods.
This is especially useful for phase retrieval.

### Figure 5: qualitative examples
For 4 images, show:
- ground truth
- SITCOM reconstruction
- Noise Picking reconstruction
- error map

Include:
- easy case
- medium case
- hard case
- failure case

### Figure 6: mechanism figure
For one or two representative images, show over diffusion time:
- masked measurement error
- full measurement error
- optionally candidate score or reconstruction quality trend

---

## 19. One-month schedule

### Week 1
Finish:
- canonical comparison script
- split files for dev / validation / test
- Phase 0 sanity runs
- start Phase 1 radius validation
- start Phase 2 SITCOM light tuning

Writing:
- create the paper skeleton
- draft Introduction
- draft Problem Setup
- draft Method section rough outline

### Week 2
Finish:
- complete Phase 1 radius validation
- complete Phase 2 SITCOM tuning
- complete Phase 3 Noise Picking schedule tuning

Writing:
- lock the Experimental Setup section
- fully draft the Method section
- start writing the Results section structure

### Week 3
Finish:
- complete Phase 4 runtime study
- complete Phase 5 mechanism ablation
- start Phase 6 main held-out benchmark

Writing:
- draft the main Results section
- prepare Table 1, Table 3, and the runtime figure

### Week 4
Finish:
- complete Phase 6 main benchmark
- complete Phase 7 if possible
- finalize qualitative examples and mechanism figure
- consistency checks and cleanup

Writing:
- finalize Intro, Results, Discussion, Limitations, Conclusion
- finalize figure/table ordering
- write appendix / supplementary details

---

## 20. Suggested paper structure

### 1. Introduction
State clearly:
- phase retrieval is ill-conditioned
- diffusion priors help but reconstruction trajectories remain sensitive
- Noise Picking uses masked consistency to improve trajectory selection
- contributions: method, experiments, restart-aware evaluation, mechanism ablation

### 2. Related Work
Keep focused:
- classical phase retrieval
- diffusion-based inverse problems
- diffusion for phase retrieval / SITCOM-like methods
- candidate selection or restart-based reconstruction strategies if relevant

### 3. Problem Setup
Define:
- Fourier magnitude-only measurements
- reconstruction objective
- notation
- evaluation and restart protocol

### 4. Method
Include:
- Noise Picking overview
- masked candidate scoring
- late masked projection
- staged candidate schedule
- compute discussion

### 5. Experimental Setup
Include:
- dataset and preprocessing
- measurement model
- baselines
- seed/restart protocol
- metrics
- hyperparameter selection process
- hardware

### 6. Results
Suggested order:
- validation radius selection
- main held-out comparison
- runtime/restart tradeoff
- mechanism ablation
- qualitative examples

### 7. Discussion and Limitations
State honestly:
- current evidence is on the face-prior setting
- runtime remains significant
- single-run and best-of-R may prefer different hyperparameters
- broader-domain evaluation is future work

### 8. Conclusion
Summarize only what was actually shown.

---

## 21. Non-negotiable rules during the month

1. Do not tune on the test set.
2. Do not silently mix mean and best-of-10 reporting.
3. Do not change the headline comparison away from:
   - Noise Picking masked
   - SITCOM unmasked
4. Do not expand to additional datasets before the current test benchmark is done.
5. Do not keep sweeping very many radii after Phase 1.
6. Do not over-tune SITCOM beyond a reasonable fair baseline.

---

## 22. Practical checklist

### Code / setup
- [ ] Create canonical masked-NP vs unmasked-SITCOM script
- [ ] Save dev / validation / test image lists
- [ ] Ensure CSVs include all needed config fields
- [ ] Add image-level aggregation output

### Phase 0
- [ ] Rerun original 5 images with radii `{0.1, 0.2, 0.5}`
- [ ] Confirm preliminary signal matches earlier results

### Phase 1
- [ ] Run validation radius study
- [ ] Select single-run radius
- [ ] Select best-of-10 radius

### Phase 2
- [ ] Tune SITCOM LR
- [ ] Tune SITCOM eta/init scales
- [ ] Freeze SITCOM default

### Phase 3
- [ ] Tune `num_candidates_soft`
- [ ] Tune `num_candidates_hard`
- [ ] Tune `proj_start`
- [ ] Freeze Noise Picking schedule

### Phase 4
- [ ] Run Noise Picking step-budget study
- [ ] Run SITCOM compute-budget study
- [ ] Make quality-runtime plot

### Phase 5
- [ ] Run mechanism ablation
- [ ] Make mechanism table

### Phase 6
- [ ] Run main held-out benchmark
- [ ] Compute mean / median / best-of-10 summaries
- [ ] Compute image win rates
- [ ] Make main comparison table

### Phase 7
- [ ] Run masked-SITCOM secondary ablation if time allows

### Drafting
- [ ] Write paper skeleton in week 1
- [ ] Write Method and Experimental Setup in week 2
- [ ] Write Results in week 3
- [ ] Finish draft and figures in week 4

---

## 23. Final target statement

If the month goes well, the paper should be able to support a concise final message like this:

> **Noise Picking’s masking mechanism is essential, the best masking radius depends on whether one evaluates typical or best-of-R performance, and under a frozen CelebA-HQ 256 phase retrieval protocol the masked method outperforms unmasked SITCOM.**

That is enough for a real first experimental draft.
