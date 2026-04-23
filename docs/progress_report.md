# prdiffusion progress report

This document is the main up-to-date project progress note for **prdiffusion**.

It replaces the earlier main progress report, which has now been moved to:

- `docs/historical/progress_report_pre_piecewise_and_runtime_matched.md`

The current version incorporates:

- full Phase 10 / 11 mechanism results,
- reduced-NP and reduced-hybrid results,
- runtime-matched NP results,
- candidate-scaled reduced-NP results,
- piecewise-schedule NP results,
- and the resulting project-direction update.

---

## 1. Current project status in one paragraph

The project has now moved past the earlier open question of whether the work should be framed mainly as a broad transferable “defer hard consistency” mechanism or as a new solver direction. The current evidence favors the **new-solver direction centered on Noise Picking (NP)**. Within the NP family, the mechanism story is now much clearer: **hard consistency should be deferred, and the late dense refinement regime is critical**. Full Phase 10/11 results show that hard-from-start projection is harmful and that late hard projection is the main driver of good performance. Runtime-matched experiments further show that plain NP at roughly SITCOM-matched runtime already outperforms default SITCOM on average quality, while candidate scaling at only 100 steps does not rescue performance. Reduced hybrids and delayed-handoff studies remain scientifically informative, but no SITCOM-side or hybrid variant currently surpasses the strongest NP variants on the quality views that matter most, especially best-of-10.

---

## 2. Frozen reference setup

Unless explicitly stated otherwise, the current working defaults remain:

### Global

- model: `google/ddpm-celebahq-256`
- primary radius: `r = 0.5`
- standard seeds: `100,101,102,103,104,105,106,107,108,109`
- primary reporting views:
  - image-level mean PSNR,
  - image-level median PSNR,
  - image-level max / best-of-10 PSNR,
  - image-level mean full magnitude L2,
  - image-level mean low-frequency magnitude L2,
  - image-level mean runtime.

### Canonical Noise Picking reference

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

---

## 3. Earlier findings that still hold

The earlier conclusions from the pre-piecewise stage remain valid:

1. **masked / projection-based NP strongly beats default unmasked SITCOM** on the canonical held-out comparison.
2. **late hard projection is the main NP-side mechanism**.
3. **hard-from-start projection is harmful**.
4. **candidate switching by itself is not enough**.
5. **SITCOM-side delayed masked variants can help modestly**, but these improvements are not strong enough to overturn the main NP-side story.

These points are now reinforced by the new runtime-aware and piecewise experiments.

---

## 4. Full Phase 10 / 11 mechanism results

### 4.1 Main NP mechanism result

The full Phase 10 / 11 block now supports a much cleaner interpretation of Noise Picking:

> The main source of gain is **deferred hard consistency**, not the candidate-count switch alone.

The best-quality mechanism-side NP variant on validation is:

- `np_fixedk_lateproj`

Main Phase 10 / 11 takeaways:

- `np_fixedk_lateproj` is stronger than canonical NP in validation quality.
- `np_fixedk_alwaysproj` is much worse, showing that **hard-from-start projection is too aggressive**.
- `np_fixedk_noproj` is bad, showing that **late hard projection is essential**.
- `np_candidate_switch_only` is bad, showing that **candidate switching alone is not the main reason NP works**.
- several Phase 10 / 11 branches collapse into duplicates in the current implementation and should not be treated as independent active methods.

### 4.2 Practical caveat

`np_fixedk_lateproj` is materially slower than canonical NP.

So there are now two distinct NP-side references:

- **best practical reference**: `np_canonical`
- **best current quality reference**: `np_fixedk_lateproj`

This distinction remains important for future paper tables.

---

## 5. Reduced-NP results

Reduced NP at very coarse step counts was run to ask whether plain NP remains competitive when heavily compressed.

### Main result

- reduced NP at `20 / 50 / 100` steps improves steadily with more steps,
- but all three variants are still well below canonical NP,
- and all three are below default SITCOM on quality, even though they are much faster.

### Important interpretation

This means the early reduced-NP failure was **not** just a matter of measurement mismatch or too few candidates. It suggested that NP needs more of the diffusion trajectory than the 20/50/100-step schedules preserved.

This conclusion later became much sharper once runtime-matched and piecewise runs were added.

---

## 6. Reduced-hybrid results

Reduced hybrids were run to test whether a SITCOM suffix could rescue an over-compressed NP prefix.

### Main result

- the best reduced-hybrid variant is typically a **masked earlier-handoff** variant,
- reduced hybrid consistently improves over reduced NP at the same compressed schedule,
- but reduced hybrid still does **not** outperform the strongest plain NP baselines,
- and it does **not** provide a convincing best-of-10 improvement over the strongest NP-side methods.

### Interpretation

Reduced hybrid is scientifically useful because it shows:

- a SITCOM suffix can partially rescue a too-coarse NP prefix,
- but this rescue is not strong enough to replace the main NP solver story,
- and the hybrid story becomes more of a secondary supporting observation than the main project direction.

---

## 7. Runtime-matched NP results

This is one of the most important new blocks.

### Main result

Plain NP at roughly SITCOM-matched runtime is already strong.

The most important points are:

- **NP420** is the cleanest runtime-matched comparison to default SITCOM.
- **NP500** is the strongest modest-over-budget NP point.

### What the results show

At about the same runtime as default SITCOM:

- NP420 is clearly better than SITCOM on **image-level mean** and **median** quality.
- NP420 is roughly tied or slightly behind SITCOM on **best-of-10**, depending on the exact slice.

At a slightly larger runtime budget:

- NP500 beats SITCOM on **mean**, **median**, and **best-of-10**.

### Interpretation

This is a major update in the project understanding:

> NP is not only good because it is given a much longer trajectory.
>
> Even at a runtime close to default SITCOM, plain NP is already a strong solver.

This strongly supports the **new solver** direction.

---

## 8. Candidate-scaled reduced-NP results

The candidate-scaled reduced-NP block asked whether a very short NP schedule could be rescued by adding many more candidates.

### Main result

It could not.

In particular:

- 100-step NP with many more soft or hard candidates did **not** recover the quality of 400–500 step NP,
- and candidate inflation at 100 steps did not provide a strong best-of-10 rescue.

### Interpretation

This is one of the clearest mechanistic findings in the project:

> Missing trajectory resolution cannot be replaced by simply exploring more candidate branches at a too-coarse set of timesteps.

This sharply separates:

- **time-grid / trajectory resolution**, and
- **candidate breadth**.

The evidence now strongly suggests that the former matters more.

---

## 9. Piecewise-schedule NP results

The piecewise experiments are the strongest new mathematical / mechanistic clarification block.

Two main families were tested:

- **full-early / reduced-late**
- **reduced-early / full-late**

with the `t599` split treated as the fair analogue of the earlier `proj_start = 400` semantics, and `t400` treated as a secondary later-deferral study.

### Main result

The decisive finding is:

> **A dense late stage matters more than a dense early stage.**

### What survives best

The best fair piecewise runs are the **reduced-early / full-late (REFL)** variants, especially:

- `REFL red500 t599`
- `REFL red400 t599`

These significantly outperform their plain reduced-NP counterparts and remain quite close to canonical NP.

### What does not survive as well

The **full-early / reduced-late (FERL)** variants are much weaker relative to their cost.

This shows that what full NP really needs is not just “many total steps,” but specifically:

- **many repeated late branch-and-project refinement opportunities on a dense grid**.

### Secondary t400 result

The `t400` piecewise runs are consistently worse than the corresponding `t599` runs.
So delaying the hard-projection regime even further does **not** help here.

### Interpretation

This is currently the strongest mechanistic refinement of the NP story:

> Deferred hard consistency is important, but the **late dense refinement regime** is the truly critical part.

---

## 10. Zero-measurement SITCOM sanity check

A small sanity check was performed for the zero-measurement SITCOM question.

### Result

With measurement weight effectively removed, SITCOM reduces to plain diffusion-like behavior rather than a meaningful measurement-guided solver.

### Why this matters

This clarifies a conceptual difference between:

- **noise selection / branch choice**, and
- **optimization with a measurement objective**.

It also motivates a future delayed-measurement SITCOM experiment as a possible mechanistic comparison, though this is not currently the top-priority project branch.

---

## 11. Current best method picture

### Best practical NP-side reference

- `np_canonical`

### Best current quality NP-side reference

- `np_fixedk_lateproj`

### Strong practical runtime-aware NP point

- `NP420`

### Strong modest-over-budget NP point

- `NP500`

### Strongest piecewise mechanism point

- `REFL red500 t599`

### SITCOM status

- default SITCOM remains the main in-repo optimization baseline,
- but no SITCOM-side or hybrid-side result currently overturns the strongest NP-side story.

### Hybrid status

- hybrids remain scientifically useful,
- some are close in runtime and reasonably close in quality,
- but they do **not** currently outperform the strongest NP variants on best-of-10.

---

## 12. Project-direction decision

The project should now be treated as following the **new-solver direction**.

### Why

This direction is justified because:

1. NP and its late-hard variants remain clearly strong.
2. Runtime-matched NP already competes well against SITCOM.
3. Candidate-scaled reduced-NP does not rescue overly short trajectories.
4. Piecewise results show a clear NP-specific mechanism story centered on the late dense refinement regime.
5. Hybrid and SITCOM-side transfer results are informative but **not** strong enough to become the main project claim.
6. Most importantly, no hybrid or SITCOM-related variant currently beats the strongest NP variants on the quality views that matter most, especially **best-of-10**.

### What this means

The paper should now be framed primarily as:

> **Noise Picking as a new solver family for diffusion-based phase retrieval**,
>
> with delayed hard consistency and dense late refinement as the core mechanism explanation.

The broader “transferable mechanism” story should remain in the paper as:

- supporting mechanism insight,
- secondary transfer evidence,
- and possible future work,

but not as the main project claim.

---

## 13. What should now be active vs inactive

### Keep active

#### Main NP family

- `np_canonical`
- `np_fixedk_lateproj`
- `NP420`
- `NP500`
- selected piecewise references, especially:
  - `REFL red400 t599`
  - `REFL red500 t599`

#### Controls

- `np_fixedk_alwaysproj`
- default `sitcom_unmasked`

### Secondary / interpretive only

- reduced hybrid best variants
- `sitcom_late_mask_proxy`
- zero-measurement SITCOM sanity findings

### Drop from active future comparison tables

- duplicate Phase 10 / 11 branches
- broad reduced-NP candidate-scaling families beyond the already completed evidence
- broad reduced-hybrid ladders as a main optimization direction
- `t400` piecewise variants as anything more than secondary mechanism evidence

---

## 14. Immediate next steps

1. Update the repo docs and freeze the new project direction.
2. Build the next experiment plan around the **new-solver direction**.
3. Focus the next major comparison block on:
   - best NP variants,
   - runtime-aware best NP variants,
   - selected external baselines,
   - and reporting that emphasizes **best-of-k / best-of-10**.
4. Keep the delayed-consistency mechanism story as a supporting explanation.
5. Optionally run a small delayed-measurement SITCOM ablation later if it helps explain the mechanism, but do not let it redirect the main paper claim.

---

## 15. One-paragraph current summary

The current project evidence supports a shift to the **new-solver direction centered on Noise Picking**. Full NP-side mechanism studies show that the key gain comes from **deferred hard consistency**, especially a strong **late dense refinement regime**, rather than candidate switching alone. Runtime-matched NP already performs strongly against default SITCOM, while candidate scaling at very short schedules fails to recover the same quality. Piecewise experiments further show that preserving the dense late stage matters more than preserving the dense early stage. Hybrid and SITCOM-side delayed-consistency ideas remain scientifically informative, but they are not currently strong enough to replace the main NP solver story, especially on the most important quality views such as best-of-10. The project should therefore now be developed primarily as a **new NP solver paper with a strong mechanism explanation**, rather than as a broad plug-and-play transfer claim.