# prdiffusion current experiment plan

This document is the active experiment plan for the current state of the project.

It supersedes the earlier plan that left open a choice between a broad transfer / hybrid direction and a solver-centered direction.

The current evidence now supports the **new-solver direction centered on Noise Picking (NP)**.

---

## 1. Current scientific picture

The project remains in the CelebA-HQ 256 / face-prior / magnitude-only Fourier setting with:

- pretrained prior: `google/ddpm-celebahq-256`
- baseline families: **Noise Picking (NP)** and **SITCOM**
- PAC as the active experiment machine

The current best understanding is now:

1. **Deferred hard consistency is the main NP-side mechanism.**
2. **The late dense refinement regime is critical.**
3. **Trajectory resolution matters more than candidate inflation at too-coarse schedules.**
4. **Runtime-matched plain NP is already strong against default SITCOM.**
5. **Hybrid and SITCOM-side delayed-consistency ideas are informative, but not strong enough to become the main project claim.**

So the active project story is now:

> **Noise Picking is a new solver family for diffusion-based phase retrieval, and its central design principle is deferred hard consistency with strong late refinement.**

---

## 2. Frozen defaults

Unless a phase explicitly says otherwise, use the following defaults.

### Global defaults

- model: `google/ddpm-celebahq-256`
- primary radius: `r = 0.5`
- standard seed list:
  - `100,101,102,103,104,105,106,107,108,109`
- report all of:
  - image-level mean PSNR
  - image-level median PSNR
  - image-level max / best-of-10 PSNR
  - image-level mean full mag L2
  - image-level mean lowfreq mag L2
  - image-level mean runtime

### Canonical NP reference

- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `proj_start = 400`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

### Canonical SITCOM reference

- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `backprop_unet = True`
- `inner_optim = adam`

### Split discipline

- pilot / debugging on `validation_10`
- method selection on `validation_25`
- confirmatory comparison on `test_20`
- final held-out headline table on `test_50`

---

## 3. Active method set

### Main NP family (keep active)

These are the current main methods that matter.

1. `np_canonical`
2. `np_fixedk_lateproj`
3. `NP420`
4. `NP500`
5. `REFL red400 t599`
6. `REFL red500 t599`

Interpretation of roles:

- `np_canonical` = main practical in-repo NP reference
- `np_fixedk_lateproj` = best current NP-family quality reference
- `NP420` = clean runtime-matched practical point
- `NP500` = strong modest-over-budget practical point
- `REFL red400 t599` and `REFL red500 t599` = strongest piecewise evidence for the late-dense-refinement mechanism

### Controls (keep active, but secondary)

1. `np_fixedk_alwaysproj`
2. `sitcom_unmasked`

### Secondary interpretive methods (do not center future tables on these)

1. selected reduced-hybrid best variants
2. `sitcom_late_mask_proxy`
3. zero-measurement SITCOM sanity check

### Drop from active future comparison tables

- duplicate Phase 10 / 11 aliases
- broad reduced-NP candidate-scaling variants beyond the completed evidence
- broad reduced-hybrid ladders as primary methods
- `t400` piecewise variants except as secondary mechanism evidence
- `sitcom_weak_then_strong`

---

## 4. Direction decision

The project should now follow:

## Direction 2: new-solver direction

### Why this is now the right direction

Use this direction because:

- NP and its late-hard variants remain clearly strong,
- runtime-matched plain NP is already competitive or better than default SITCOM,
- candidate-scaled reduced NP does not rescue coarse schedules,
- piecewise results reveal a clear NP-centered mechanism story,
- transfer to SITCOM and hybrids remains informative but weaker and less decisive,
- and no hybrid or SITCOM-related method currently beats the strongest NP variants on the most important quality views, especially **best-of-10**.

### What this means scientifically

The paper should now primarily claim:

- **Noise Picking is a new solver family**,
- with **deferred hard consistency** and **dense late refinement** as the main mechanism explanation.

The broader transfer story should remain:

- secondary evidence,
- supporting interpretation,
- and possible future work,

but not the main paper headline.

---

## 5. Immediate experiment order

### Phase A: tighten the main NP-centered comparison set

Prepare the final active comparison matrix on `validation_25` from the already completed evidence:

- `sitcom_unmasked`
- `np_canonical`
- `np_fixedk_lateproj`
- `NP420`
- `NP500`
- `REFL red400 t599`
- `REFL red500 t599`
- `np_fixedk_alwaysproj` (control)

Goal:

- identify a clean main paper table,
- and identify which methods belong in the final held-out block.

### Phase B: confirm on `test_20`

Run the selected finalists on `test_20`.

The default recommendation is to confirm at least:

- `sitcom_unmasked`
- `np_canonical`
- `np_fixedk_lateproj`
- `NP420`
- `NP500`
- one piecewise representative:
  - preferably `REFL red500 t599`

Goal:

- check whether the ranking and best-of-10 behavior remain stable,
- especially for runtime-aware NP points.

### Phase C: final held-out table on `test_50`

Once the `test_20` confirmation is stable, run the final held-out comparison on `test_50`.

Recommended final held-out set:

- `sitcom_unmasked`
- `np_canonical`
- `np_fixedk_lateproj`
- `NP420`
- `NP500`
- optionally one piecewise representative if it still looks clearly worthwhile after `test_20`

Goal:

- produce the final main paper table with both quality and runtime.

---

## 6. Reporting priorities

Because the field strongly values best reconstruction quality, future tables should emphasize:

### Required quality views

- image-level **best-of-10 / max PSNR**
- image-level mean PSNR
- image-level median PSNR

### Required measurement views

- image-level mean full mag L2
- image-level mean lowfreq mag L2

### Required runtime view

- image-level mean runtime

### Recommended fairness views

For final comparisons, also prepare:

- **best-of-k curves** for `k ∈ {1,2,4,8,10}`
- and, if practical, **best-within-time-budget** summaries

These are especially important because the project direction is now solver-centered rather than only mechanism-centered.

---

## 7. Specific scientific questions still worth answering

The project no longer needs broad branching, but a few focused questions remain worthwhile.

### Q1. Which NP-family variant is the best final paper representative?

This is the central active question.

Candidates:

- `np_canonical`
- `np_fixedk_lateproj`
- `NP420`
- `NP500`
- possibly `REFL red500 t599`

### Q2. Is the best runtime-aware NP point good enough to be a main practical headline?

This is likely between:

- `NP420`
- `NP500`

### Q3. Should one piecewise method be kept in the main paper or only in the mechanism appendix?

Current best candidate:

- `REFL red500 t599`

My default expectation is that one piecewise result should probably be kept for mechanism explanation, but not necessarily foregrounded as the main practical solver.

### Q4. Is a delayed-measurement SITCOM ablation still worth doing?

Possibly, but only as a **secondary mechanism study** if it helps explain why delayed-consistency ideas do not automatically transfer.

This should not delay the main NP-centered paper path.

---

## 8. What not to do right now

- Do not reopen broad radius sweeps.
- Do not reopen broad SITCOM retuning.
- Do not reopen broad reduced-NP candidate-scaling sweeps.
- Do not re-center the paper on hybrids.
- Do not tune on `test_20` or `test_50`.
- Do not expand to a broad transfer claim without stronger best-of-10 transfer evidence.

---

## 9. Minimal active reporting set

For every finalist, keep reporting:

- avg image mean PSNR
- avg image median PSNR
- avg image max / best-of-10 PSNR
- avg image mean full mag L2
- avg image mean lowfreq mag L2
- avg image mean runtime
- image-level win rates against the nearest practical baseline

For final paper-quality comparison, also prepare:

- best-of-k curve,
- and optionally best-within-time-budget comparison.

---

## 10. One-paragraph working summary

The project should now proceed under the **new-solver direction**. The current evidence shows that Noise Picking is not merely a mechanism toy: runtime-matched plain NP already performs strongly against default SITCOM, while candidate inflation at very coarse schedules does not recover the same quality. Full mechanism studies and piecewise schedule experiments further indicate that the core gain comes from **deferred hard consistency**, especially a **dense late refinement regime**, rather than from candidate switching alone. Hybrid and SITCOM-side delayed-consistency variants remain scientifically useful, but they do not currently beat the strongest NP variants on the quality views that matter most, especially best-of-10. The next experimental block should therefore narrow to the strongest NP-family methods, confirm them on `test_20`, and then produce a final held-out `test_50` table with quality, measurement, and runtime-aware reporting.