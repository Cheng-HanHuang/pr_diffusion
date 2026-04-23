# prdiffusion progress report (archived snapshot before Direction 2 pivot)

This file archives the previous active `docs/progress_report.md` before the project plan was updated toward the NP-centered solver direction.

Archived on: 2026-04-23

---

# prdiffusion progress report

This note records the current experimental status of the **prdiffusion** project and is intended to be the main up-to-date progress summary in the repo.

It supersedes older summaries that were written before the current mechanism reinterpretation, full Phase 10/11 runs, and the NP→SITCOM hybrid direction.

---

## 1. Current objective

The project is still focused on phase retrieval with:

- the face-prior setup,
- `google/ddpm-celebahq-256`,
- magnitude-only Fourier measurements, and
- the two in-repo reconstruction families:
  - **SITCOM**
  - **Noise Picking (NP)**

PAC remains the active experiment machine.

The current objective is no longer just to compare masked NP against unmasked SITCOM. The main objective now is to determine the strongest correct scientific framing among:

1. **transferable mechanism**: soft early measurement use, hard consistency later,
2. **hybrid method**: NP-style early dynamics with a later SITCOM suffix,
3. **new solver direction**: Noise Picking itself as the primary solver family.

---

## 2. Stable earlier findings that still hold

### 2.1 SITCOM tuning

Earlier SITCOM tuning established:

- `lr_inner = 0.02` is the best current default,
- `lr_inner = 0.1` is clearly too large,
- and `eta_scale = 0.5` can help PSNR but worsens measurement consistency.

The practical frozen SITCOM default remains:

- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `backprop_unet = True`
- `inner_optim = adam`

### 2.2 Radius and early NP schedule

Earlier validation and schedule studies established:

- primary radius: `r = 0.5`
- secondary radius: `r = 0.2`
- canonical NP schedule:
  - `num_candidates_soft = 5`
  - `num_candidates_hard = 1`
  - `proj_start = 400`

These are still the current reference defaults.

---

## 3. Shift in interpretation after mechanism studies

The project has moved beyond the earlier simple story of “late masked projection helps.”

The stronger current interpretation is:

> **hard consistency should be deferred**.
>
> Early in the diffusion trajectory, measurement information is better used softly; later, harder consistency becomes important.

This reinterpretation came from the combination of:

- earlier mechanism ablations,
- the full Phase 10 decoupling run,
- the full Phase 11 hard-early / hard-late run,
- and the Phase 12 SITCOM-side confirmation runs.

---

## 4. Full Phase 10 results: NP mechanism decoupling

Phase 10 tested:

- `np_canonical`
- `np_fixedk_lateproj`
- `np_fixedk_alwaysproj`
- `np_fixedk_noproj`
- `np_candidate_switch_only`
- `np_projection_only_switch`

### Main findings

#### 4.1 Deferred hard consistency is real

`np_fixedk_lateproj` is the best-quality variant on the validation run.

It outperforms canonical NP on average quality and measurement consistency, while keeping the same basic mechanism family.

#### 4.2 Hard-from-start projection is harmful

`np_fixedk_alwaysproj` is much worse than `np_fixedk_lateproj`.

This is strong evidence that **hard projection from the beginning is too aggressive**.

#### 4.3 No projection is bad

`np_fixedk_noproj` performs poorly and has very large measurement error.

So **hard projection later is not optional**.

#### 4.4 Candidate switching alone is not enough

`np_candidate_switch_only` performs very poorly.

So the current evidence does **not** support the story that the success of NP mainly comes from switching from many candidates early to one candidate late.

#### 4.5 One Phase 10 branch is a duplicate

`np_projection_only_switch` is effectively identical to `np_fixedk_lateproj` in the current implementation and should not be treated as a distinct active method in future tables.

### Phase 10 interpretation

The strongest interpretation after the full run is:

- the critical ingredient is **deferred hard consistency**,
- not the candidate-count switch by itself.

---

## 5. Full Phase 11 results: hard-early vs hard-late

Phase 11 tested:

- `hard_from_start`
- `hard_late`
- `hard_never`
- `soft_only`
- `soft_then_hard`

### Main findings

As implemented, Phase 11 mostly collapses to three distinct behaviors:

- `hard_from_start`
- `hard_late`
- `hard_never`

with:

- `soft_then_hard` behaving like `hard_late`,
- `soft_only` behaving like `hard_never`.

So Phase 11 does **not** add a truly independent new soft-only mechanism in the current implementation.

### What still matters scientifically

Even with that implementation limitation, the full run reinforces the same key message as Phase 10:

- **hard late** is best,
- **hard from start** is much worse,
- **hard never** is also bad.

This strengthens the “soft early / hard late” interpretation, even if the code-level ladder should be simplified later.

---

## 6. Current best understanding of Noise Picking

The current best-quality NP-side variant is:

- `np_fixedk_lateproj`

But there is an important practical caveat:

- it is materially slower than canonical NP.

So the project now has two distinct NP-side reference points:

### Best practical reference

- `np_canonical`

### Best current quality reference

- `np_fixedk_lateproj`

This is important for later paper framing. A future paper may need to distinguish:

- best quality,
- versus best quality/runtime tradeoff.

---

## 7. Phase 12 SITCOM-side results

Phase 12 tested four SITCOM-side variants:

- `sitcom_unmasked`
- `sitcom_hard_from_start_masked`
- `sitcom_late_mask_proxy`
- `sitcom_weak_then_strong`

### Main findings

#### 7.1 `sitcom_late_mask_proxy` is the best current SITCOM variant

On the fuller Phase 12 run, `sitcom_late_mask_proxy` showed:

- modest but real PSNR improvements over unmasked SITCOM,
- stronger improvements in full and low-frequency measurement consistency,
- and almost no runtime cost increase.

This is enough to keep SITCOM in the discussion.

#### 7.2 `sitcom_hard_from_start_masked` is too aggressive as a default

This variant can improve some best-case outcomes and consistency, but it is not the best average SITCOM variant.

#### 7.3 `sitcom_weak_then_strong` is not the main survivor

This weighted-loss surrogate is not currently the strongest SITCOM-side branch and should not be prioritized further.

### SITCOM interpretation

The current positive SITCOM transfer result is:

- `sitcom_late_mask_proxy`

This is still only a **modest** transfer result, not a dramatic one, but it is real enough to keep SITCOM in the active matrix.

---

## 8. Current hybrid direction

A new hybrid direction is now active.

The central hypothesis is:

> NP-style soft early dynamics may be preferable to early SITCOM optimization,
> but a later SITCOM suffix may still help refine the solution.

This hybrid family is currently running via:

- `scripts/pr_phase12_hybrid_ladder.py`

Active hybrid variants include:

- `np_to_sitcom_400`
- `np_to_sitcom_600`
- `np_to_sitcom_masked_400`
- `np_to_sitcom_masked_600`

So at the moment:

- **the NP→SITCOM hybrid ladder is still running**
- and its results are a key pending input for the next project decision

---

## 9. Active method matrix

### Keep active

#### Noise Picking side

- `np_canonical`
- `np_fixedk_lateproj`
- `np_fixedk_alwaysproj` (negative control)

#### SITCOM side

- `sitcom_unmasked`
- `sitcom_late_mask_proxy`
- `sitcom_hard_from_start_masked` (control only)

#### Hybrid side

- the currently running NP→SITCOM hybrids

### Drop or demote from active status

- `np_projection_only_switch` as a separate method (duplicate)
- `np_candidate_switch_only`
- `soft_only` and `soft_then_hard` as distinct Phase 11 methods in their current implementation
- `sitcom_weak_then_strong`

---

## 10. Current strategic question

The project is now at a fork, but the fork should not be decided until the hybrid runs finish.

The two serious directions are:

### Direction A: transferable mechanism / hybrid story

Use this direction if:

- the hybrid methods are promising,
- and the combination of NP + SITCOM gives a coherent story of soft-early / hard-late / optimization-late.

### Direction B: Noise Picking as the main solver direction

Use this direction if:

- NP and its late-hard variants remain clearly strong,
- but transfer beyond NP remains weak or not convincing enough.

At the moment, it is too early to choose decisively between these two directions.

---

## 11. Immediate next steps

1. finish the currently running hybrid ladder,
2. write one combined note for:
   - full Phase 10,
   - full Phase 11,
   - full Phase 12,
   - hybrid results,
3. simplify the active experiment matrix,
4. then decide whether the next external-comparison block should be framed as:
   - hybrid / transfer continuation,
   - or NP-centered solver comparison.

---

## 12. What should not be repeated now

At this point, the following should not be reopened unless a later result forces it:

- broad radius sweeps,
- broad SITCOM hyperparameter retuning,
- duplicate Phase 10/11 branches,
- large second-host expansions before the hybrid story is understood.

---

## 13. One-paragraph current summary

The project has now moved beyond the original interpretation that “late masked projection helps.” Full Phase 10/11 results indicate that the stronger mechanism story is **deferred hard consistency**: hard projection from the beginning is harmful, pure soft guidance without later hard enforcement is insufficient, and a fixed-`k` late-projection variant can outperform canonical NP in validation quality, though at a substantial runtime cost. On the SITCOM side, a late masked-loss proxy now provides a modest but real positive transfer result, while more aggressive or purely weighted alternatives are less convincing. The project is therefore no longer just about masked NP versus unmasked SITCOM. The active question is whether the work is best framed as a **transferable soft-early / hard-late mechanism**, especially through the currently running **NP→SITCOM hybrid ladder**, or as a **new solver direction centered on Noise Picking**.
