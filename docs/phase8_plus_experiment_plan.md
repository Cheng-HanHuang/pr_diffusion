# prdiffusion continued experiment plan (Phase 8+)

This document continues the existing experiment roadmap in `docs/neurips_prdiffusion_experiment_plan.md`, starting from **Phase 8**.

The goal of these next phases is **not** to immediately broaden the paper claim. The goal is to run the next most informative experiments, especially those that test whether the current masked-consistency idea can improve more than one reconstruction host and whether the late masked projection idea is stronger than simpler measurement-forcing alternatives.

This continuation is written using the current best frozen settings from `docs/progress_report.md` whenever earlier docs and later findings differ.

---

## 0. Frozen defaults for Phase 8+

Unless a phase explicitly says otherwise, use the following defaults.

### 0.1 Global defaults

- model: `google/ddpm-celebahq-256`
- dataset setting: current CelebA-HQ 256 face-prior setting
- primary radius: `r = 0.5`
- secondary radius: `r = 0.2`
- standard seed list:
  - `100,101,102,103,104,105,106,107,108,109`
- report both:
  - single-run summaries (mean / median over seeds)
  - best-of-10 summaries

### 0.2 Frozen Noise Picking default

Use the current PAC main configuration as the default masked-NP reference:

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `proj_start = 400`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

Fast backup point:

- `soft = 3`
- `hard = 1`
- `proj_start = 400`
- same radius

### 0.3 Frozen SITCOM default

Use the current tuned SITCOM baseline as the default SITCOM reference:

- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `backprop_unet = True`
- `inner_optim = adam`
- unmasked baseline: `meas_radius = None`

### 0.4 Split discipline

- tune / pilot on `validation_10` or `validation_25`
- freeze choices on `validation_25`
- use `test_20` for confirmation runs once a design is selected
- use `test_50` only for final held-out large runs

---

## 1. Practical goal of the next block

Phases 0-7 already established:

- a frozen main NP setting,
- a strong main comparison against unmasked SITCOM,
- and strong evidence that masking matters.

The next block should answer these more practical questions:

1. Can the masking / late-projection idea improve **SITCOM itself**?
2. Can the same idea improve a **second diffusion reconstruction host**?
3. Is the gain specifically from the proposed **late low-frequency enforcement**, rather than from any forcing at all?
4. Do the gains survive under small perturbations such as noise, alternate radius, or runtime / restart constraints?

---

## 2. Recommended execution structure on PAC

PAC has four GPUs, so use **waves** of experiments rather than a single serial queue.

### Wave A: fast pilots on small validation slice

Use `validation_10`, 5 seeds.

Run in parallel:

- **GPU 0**: Phase 8 pilot (SITCOM insertion ladder)
- **GPU 1**: Phase 10 pilot (second-host insertion ladder)
- **GPU 2**: Phase 11 pilot (forcing-type comparison)
- **GPU 3**: Phase 13 pilot (noise robustness or schedule refinement, whichever code is ready first)

Purpose:

- verify code,
- catch obvious failures,
- identify clearly bad settings before the full `validation_25` runs.

### Wave B: full validation runs

Use `validation_25`, 10 seeds.

Run in parallel:

- **GPU 0**: Phase 8 full run
- **GPU 1**: Phase 10 full run
- **GPU 2**: Phase 11 full run
- **GPU 3**: Phase 9 or Phase 13, depending on what Wave A says

### Wave C: confirmation runs

Use `test_20`, 10 seeds.

Run the frozen best variants from Phases 8-11 in parallel.

### Wave D: final held-out expansion

Use `test_50`, 10 seeds.

Split the final method set across the four GPUs by method or by image chunks.

---

## 3. Phase 8: SITCOM insertion ladder

### Purpose

Test whether the masked-consistency idea improves **SITCOM** when inserted into SITCOM itself.

This is the most important immediate next experiment.

### Required code support

Extend SITCOM so that it can support not only:

- always-unmasked (`meas_radius = None`)
- always-masked (`meas_radius = r`)

but also a **scheduled / late-masked** version, where measurement forcing is turned on only after a chosen outer-step threshold.

A minimal implementation is enough. The cleanest first version is:

- before switch: unmasked measurement loss
- after switch: masked low-frequency loss with radius `r`

Optional second version:

- before switch: no measurement masking
- after switch: masked low-frequency loss with radius `r`
- keep the same `lr_inner`, `lam`, and optimizer

### Methods to compare

Run the following four SITCOM variants:

1. **SITCOM-unmasked**
   - canonical baseline
   - `meas_radius = None`

2. **SITCOM-masked-all**
   - masked at radius `0.5` for the whole run

3. **SITCOM-late-mask-0.5**
   - unmasked early
   - switch to masked forcing at `start = 400`
   - radius `0.5`

4. **SITCOM-late-mask-0.2**
   - unmasked early
   - switch to masked forcing at `start = 400`
   - radius `0.2`

### Images

Use `validation_25`.

### Seeds

Use 10.

### Count

- `25 × 10 × 4 = 1000`

### Metrics

Per method report:

- avg image mean PSNR
- avg image median PSNR
- avg image max PSNR
- avg image mean full mag L2
- avg image mean lowfreq mag L2
- avg image mean runtime
- image win rate against SITCOM-unmasked

### Decision rule

Select one best SITCOM+masking insertion variant.

The most likely good outcome is that **late masked forcing** is better than always-masked forcing, but this should be treated as an empirical question.

### Parallel recommendation

- **Wave A pilot** on `validation_10` with 5 seeds
- **Wave B full** on `validation_25` with 10 seeds

---

## 4. Phase 9: SITCOM schedule refinement

### Purpose

If Phase 8 shows that late masking helps SITCOM, refine only the **switch timing**, not the whole SITCOM configuration.

Do not reopen broad SITCOM tuning.

### Conditions

Only do this phase if Phase 8 shows a clear signal in favor of a scheduled / late-masked SITCOM.

### Settings to sweep

Use the Phase 8 winning SITCOM insertion form and sweep:

- `mask_start ∈ {200, 400, 600}`

and optionally one secondary radius comparison:

- radius in `{0.2, 0.5}`

### Recommended two-stage execution

#### Pilot

Use `validation_10`, 5 seeds.

Count:

- if sweeping start only: `10 × 5 × 3 = 150`
- if sweeping start and radius: `10 × 5 × 3 × 2 = 300`

#### Freeze run

Take the best 1 or 2 settings and rerun on `validation_25`, 10 seeds.

Count:

- best 2 settings: `25 × 10 × 2 = 500`

### Deliverable

A small table showing the best SITCOM insertion schedule.

### Parallel recommendation

Run this on **GPU 3** during Wave B if Phase 8 pilot already points to late masking.

---

## 5. Phase 10: second-host insertion study

### Purpose

Test whether the masked projection idea can improve a second reconstruction host that is **not SITCOM**.

To keep implementation burden reasonable, use a host derived from the current diffusion pipeline rather than importing a completely separate solver family first.

### Recommended second host

Use a **single-candidate diffusion host**:

- same diffusion prior
- same denoise / resample backbone as the NP pipeline
- but no candidate competition
- effectively `num_candidates_soft = 1` and `num_candidates_hard = 1`

This host is useful because it removes the candidate-selection advantage and lets you test the measurement-enforcement idea more directly.

### Methods to compare

1. **1cand-base**
   - one candidate throughout
   - no masked projection

2. **1cand-lateproj-0.5**
   - one candidate throughout
   - low-frequency masked projection on
   - `proj_start = 400`
   - radius `0.5`

3. **1cand-alwaysproj-0.5**
   - one candidate throughout
   - projection active from the start (`proj_start = 0` or earliest valid step)
   - radius `0.5`

4. **1cand-lateproj-0.2**
   - one candidate throughout
   - `proj_start = 400`
   - radius `0.2`

5. **NP-canonical**
   - current main NP reference
   - `soft = 5, hard = 1, proj_start = 400, r = 0.5`

### Images

Use `validation_25`.

### Seeds

Use 10.

### Count

- `25 × 10 × 5 = 1250`

### Metrics

Per method report:

- avg image mean / median / max PSNR
- full mag error
- lowfreq mag error
- runtime
- image win rate against `1cand-base`

### Decision rule

If `1cand-lateproj-0.5` or a nearby scheduled version clearly beats `1cand-base`, then the projection idea is helping a second host.

### Parallel recommendation

- **Wave A pilot** on `validation_10`, 5 seeds
- **Wave B full** on `validation_25`, 10 seeds

---

## 6. Phase 11: forcing-type comparison on a common host

### Purpose

Test whether the proposed forcing mechanism is better than simpler alternatives.

This phase should be run on **one host only first**, to keep implementation manageable. The recommended first host is the second host from Phase 10, because it isolates the forcing mechanism more cleanly.

### Minimal new code support

Add one additional projection operator that enforces **full Fourier magnitude**, not only low-frequency magnitude.

Then compare four forcing modes.

### Methods to compare

1. **no forcing**
   - no projection

2. **full-proj-all**
   - full Fourier-magnitude projection from the start

3. **lowfreq-proj-all**
   - low-frequency projection from the start
   - radius `0.5`

4. **lowfreq-proj-late**
   - low-frequency projection only after `start = 400`
   - radius `0.5`

Optional fifth method if code is easy:

5. **full-proj-late**
   - full Fourier-magnitude projection after `start = 400`

### Images

Use `validation_25`.

### Seeds

Use 10.

### Count

For 4 methods:

- `25 × 10 × 4 = 1000`

For 5 methods:

- `25 × 10 × 5 = 1250`

### Decision rule

This phase is successful if it establishes whether:

- low-frequency projection is better than full projection,
- and late projection is better than always-on projection.

### Parallel recommendation

- **Wave A pilot** on `validation_10`
- **Wave B full** on `validation_25`

---

## 7. Phase 12: cross-host confirmation on test_20

### Purpose

After Phases 8-11 are frozen on validation data, confirm the most promising variants on held-out `test_20`.

This is the first phase that directly checks whether the insertion effects generalize beyond validation.

### Method set

Run the following five methods:

1. **SITCOM-unmasked**
2. **SITCOM + best insertion from Phase 8/9**
3. **1cand-base**
4. **1cand + best insertion from Phase 10/11**
5. **NP-canonical**

### Images

Use `test_20`.

### Seeds

Use 10.

### Count

- `20 × 10 × 5 = 1000`

### Deliverable

A cross-host held-out comparison table.

Suggested columns:

- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- lowfreq mag error
- runtime
- image win rate against host baseline

### Parallel recommendation

Split the five methods across the four GPUs by launching two methods on the fastest available GPU or by splitting image chunks.

---

## 8. Phase 13: noise robustness

### Purpose

Check whether the same conclusions survive mild measurement noise.

This is useful even if the current main paper remains noise-free, because it tests whether the masking idea is brittle.

### Measurement model

Inject noise into the target magnitude before reconstruction.

Use a simple relative-noise model first, for example:

- `noise_level ∈ {0.0, 0.01, 0.03}`

where the exact implementation should be documented carefully.

### Recommended method set

Use four methods:

1. **SITCOM-unmasked**
2. **SITCOM + best insertion**
3. **1cand + best insertion**
4. **NP-canonical**

### Recommended two-stage execution

#### Pilot

Use `validation_10`, 5 seeds.

Count:

- `10 × 5 × 3 noise levels × 4 methods = 600`

#### Full validation run

Use `validation_25`, 5 seeds.

Count:

- `25 × 5 × 3 × 4 = 1500`

If this is too expensive, reduce to two levels:

- `0.0` and `0.03`

which would give:

- `25 × 5 × 2 × 4 = 1000`

### Deliverable

A robustness table or plot of PSNR and measurement error versus noise level.

### Parallel recommendation

Use **GPU 3** for the pilot while Phases 8-11 occupy the other GPUs.

---

## 9. Phase 14: restart and compute-fairness analysis

### Purpose

Show how the insertion variants behave under restart-aware and compute-aware evaluation.

This phase should avoid unnecessary reruns wherever possible.

### 14.1 Restart curves

From the existing 10-seed outputs of Phases 8, 10, and 12, compute best-of-R curves for:

- `R ∈ {1, 3, 5, 10}`

for the following frozen methods:

- SITCOM-unmasked
- SITCOM + best insertion
- 1cand-base
- 1cand + best insertion
- NP-canonical

This requires **postprocessing only**, not new reconstruction runs.

### 14.2 Compute-matched comparison

Run a small explicit compute-matched comparison on `test_20`.

Suggested methods:

1. **SITCOM-unmasked (20,20)**
2. **SITCOM + insertion (20,20)**
3. **SITCOM + insertion heavy (50,10)**
4. **1cand + insertion (1000 steps)**
5. **NP-canonical (1000 steps)**
6. **NP-fast (500 steps, same mask settings)**

### Seeds

Use 10.

### Count

- `20 × 10 × 6 = 1200`

### Deliverable

A runtime / quality fairness table and a best-of-R plot.

### Parallel recommendation

This phase can be launched after Phase 12 method choices are frozen.

---

## 10. Phase 15: final held-out cross-host run on test_50

### Purpose

Produce the final large held-out experiment once the insertion design is frozen.

This is the final phase of the continuation block.

### Frozen method set

Run:

1. **SITCOM-unmasked**
2. **SITCOM + best insertion**
3. **1cand-base**
4. **1cand + best insertion**
5. **NP-canonical**

### Images

Use `test_50`.

### Seeds

Use 10.

### Count

- `50 × 10 × 5 = 2500`

### Deliverables

#### Main table

Columns:

- mean PSNR
- median PSNR
- best-of-10 PSNR
- full mag error
- lowfreq mag error
- runtime

#### Additional host-improvement table

For the two insertions:

- improvement of `SITCOM + insertion` over `SITCOM-unmasked`
- improvement of `1cand + insertion` over `1cand-base`

#### Win-rate table

Per method pair:

- image-level mean win rate
- image-level max win rate

### Parallel recommendation

Use all four GPUs.

Two suggested execution styles:

#### Option A: split by method

- GPU 0: SITCOM-unmasked
- GPU 1: SITCOM + insertion
- GPU 2: 1cand-base + 1cand + insertion
- GPU 3: NP-canonical

#### Option B: split by image chunk

For the slower methods, divide `test_50` into chunks of 10 or 25 images and run multiple chunks in parallel.

---

## 11. Priority order if time becomes tight

If not all phases can be completed, use this priority order.

### Highest priority

1. **Phase 8** — SITCOM insertion ladder
2. **Phase 10** — second-host insertion ladder
3. **Phase 12** — held-out `test_20` cross-host confirmation
4. **Phase 15** — final `test_50` cross-host run

### Medium priority

5. **Phase 11** — forcing-type comparison
6. **Phase 14** — restart / compute-fairness analysis

### Lower priority but still useful

7. **Phase 13** — noise robustness
8. **Phase 9** — schedule refinement, but only if Phase 8 clearly needs it

---

## 12. Immediate next launch recommendation

If the goal is to move quickly and make PAC effective, the next launch block should be:

### GPU 0
**Phase 8 pilot**
- SITCOM-unmasked
- SITCOM-masked-all-0.5
- SITCOM-late-mask-0.5
- SITCOM-late-mask-0.2
- `validation_10`
- 5 seeds

### GPU 1
**Phase 10 pilot**
- 1cand-base
- 1cand-lateproj-0.5
- 1cand-alwaysproj-0.5
- 1cand-lateproj-0.2
- NP-canonical
- `validation_10`
- 5 seeds

### GPU 2
**Phase 11 pilot**
- no forcing
- full-proj-all
- lowfreq-proj-all
- lowfreq-proj-late
- `validation_10`
- 5 seeds

### GPU 3
**Phase 13 pilot**
- noise levels `0.0, 0.01, 0.03`
- four frozen methods if code is ready,
- otherwise use this GPU for Phase 9 pilot after initial Phase 8 signals appear.

This is the most efficient next wave because it tests:

- insertion into SITCOM,
- insertion into a second host,
- comparison against alternate forcing,
- and robustness,

all before committing to larger `validation_25` or `test_20` runs.

---

## 13. Final note

The main principle for this continuation is:

- keep the **existing frozen defaults** unless the new experiment is explicitly about changing them,
- do not retune on the test set,
- and use PAC parallelism to answer multiple targeted questions at once rather than making one giant serial sweep.

The continuation block is successful if, by the end, you know:

1. whether the masking idea improves SITCOM,
2. whether it improves a second host,
3. whether late low-frequency enforcement is better than simpler forcing alternatives,
4. and whether these improvements survive on held-out images.
