# prdiffusion current experiment plan

This document is the active experiment plan for the current state of the project.
It supersedes older phase plans that were written before the recent mechanism studies, SITCOM transfer runs, and the NP→SITCOM hybrid direction.

---

## 1. Current scientific picture

The project is still in the CelebA-HQ 256 / face-prior / magnitude-only Fourier setting with:

- pretrained prior: `google/ddpm-celebahq-256`
- baseline families: **SITCOM** and **Noise Picking (NP)**
- PAC as the active experiment machine

The current best understanding is:

1. **Deferred hard consistency is important in the NP family.**
   Hard projection from the beginning is harmful, and never using hard projection is also harmful.
2. **Projection is essential.**
   Pure score-guided selection without later hard enforcement performs badly.
3. **The candidate-count switch is probably not the main reason NP works.**
   Full Phase 10 results suggest that a fixed-`k` late-projection variant can outperform canonical NP in quality, though it is slower.
4. **SITCOM can benefit modestly from a delayed masked-loss proxy.**
   The current positive SITCOM transfer result is `sitcom_late_mask_proxy`.
5. **A stronger transfer hypothesis is now being tested by hybrid methods.**
   The current hybrid idea is to use NP-style soft early dynamics and switch to a SITCOM-style suffix later.

So the active mechanism story is now:

> Use measurement information softly early, defer hard consistency until later, and test whether a late optimization-style refinement helps after the diffusion trajectory has stabilized.

---

## 2. Frozen defaults

Unless a phase explicitly says otherwise, use the following defaults.

### Global defaults

- model: `google/ddpm-celebahq-256`
- primary radius: `r = 0.5`
- secondary radius: `r = 0.2`
- standard seed list:
  - `100,101,102,103,104,105,106,107,108,109`
- report both:
  - image-level mean / median over seeds
  - image-level max / best-of-10 over seeds

### Noise Picking reference defaults

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `proj_start = 400`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

### SITCOM reference defaults

- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `backprop_unet = True`
- `inner_optim = adam`

### Split discipline

- pilot / tune on `validation_10`
- freeze on `validation_25`
- confirm on `test_20`
- reserve `test_50` for final held-out results

---

## 3. Active method set

### Noise Picking side

Keep the following methods active:

1. `np_canonical`
2. `np_fixedk_lateproj`
3. `np_fixedk_alwaysproj` (negative control)

Drop from the active matrix as separate methods:

- `np_projection_only_switch` (duplicate of `np_fixedk_lateproj` in the current implementation)
- `np_candidate_switch_only`
- Phase 11 aliases that collapse to duplicates in the current implementation

### SITCOM side

Keep the following methods active:

1. `sitcom_unmasked`
2. `sitcom_late_mask_proxy`
3. `sitcom_hard_from_start_masked` (control / comparison only)

Do not prioritize further work on:

- `sitcom_weak_then_strong`

### Hybrid side

The current hybrid ladder is active and running via `scripts/pr_phase12_hybrid_ladder.py`.

Active hybrid family:

- `np_to_sitcom_400`
- `np_to_sitcom_600`
- `np_to_sitcom_masked_400`
- `np_to_sitcom_masked_600`

The purpose of this family is to test the hypothesis:

> NP-style soft early dynamics may be better than early SITCOM optimization, but a later SITCOM suffix may still improve final refinement.

---

## 4. Immediate experiment order

### Phase A: finish currently running full runs

Wait for:

- full Phase 10
- full Phase 11
- current NP→SITCOM hybrid runs

These are the highest-priority jobs already in flight.

### Phase B: summarize and prune

Once the running jobs finish:

1. write a single mechanism summary for the full Phase 10/11 results
2. write a hybrid summary
3. prune duplicate or dead branches from the active method matrix

### Phase C: decide the next expansion path

Use the completed Phase 10/11 + hybrid results to choose between these two directions.

#### Direction 1: mechanism / transfer paper direction

Use this if:

- `np_fixedk_lateproj` remains strong,
- the hybrid methods look promising,
- and at least one transfer story beyond canonical NP looks real.

Then the next step is:

- confirm the best hybrid and baseline methods on `test_20`
- then run a final held-out set on `test_50`

#### Direction 2: new-solver direction

Use this if:

- NP and its late-hard variants remain strong,
- but transfer to SITCOM and/or hybrids is weak or inconsistent.

Then the next step is:

- treat NP as the main solver family,
- compare the best NP variant(s) against strong external baselines,
- and narrow the claim away from broad plug-and-play transfer.

---

## 5. Current success criteria

The current block is successful if it answers these questions clearly:

1. Is deferred hard consistency the right explanation for NP?
2. Does a hybrid NP→SITCOM method outperform both pure NP and pure SITCOM variants?
3. Is the project better framed as:
   - a transferable mechanism story,
   - or a new solver story centered on NP?

---

## 6. What not to do right now

- Do not reopen broad radius sweeps.
- Do not reopen broad SITCOM hyperparameter tuning.
- Do not keep duplicate Phase 10/11 branches in future tables.
- Do not move to large second-host experiments before the hybrid results are understood.
- Do not tune on `test_20` or `test_50`.

---

## 7. Minimal active reporting set

For every active method, keep reporting:

- avg image mean PSNR
- avg image median PSNR
- avg image max PSNR
- avg image mean full mag L2
- avg image mean lowfreq mag L2
- avg image mean runtime
- image-level win rates versus the nearest baseline

---

## 8. One-paragraph working summary

The project has moved beyond the original “late masked projection” interpretation. The strongest current mechanism story is that **hard consistency should be deferred**: hard projection from the beginning is harmful, pure soft scoring without hard enforcement is insufficient, and fixed-`k` late projection can outperform canonical Noise Picking on validation quality, albeit at higher runtime. SITCOM now has one modest but real positive transfer result (`sitcom_late_mask_proxy`), but the stronger current test of the idea is the **NP→SITCOM hybrid ladder**, which is currently running. The immediate goal is therefore not to broaden the paper claim yet, but to use the full Phase 10/11 results and the hybrid results to decide whether the project should be framed as a **transferable soft-early / hard-late mechanism** or as a **new solver direction centered on Noise Picking**.
