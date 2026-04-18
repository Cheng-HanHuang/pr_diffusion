# prdiffusion docs cleanup recommendations

This note separates docs into:
- keep
- update in place
- archive / rename
- safe delete

The goal is to avoid deleting useful historical notes while still cleaning the docs folder.

---

## 1. Keep

### `docs/neurips_prdiffusion_experiment_plan.md`
Keep this.
It is still the base historical roadmap for Phases 0–7.

### `docs/progress_report.md`
Keep this.
It is a useful historical status report and also records the frozen PAC defaults that later plans should inherit.

---

## 2. Update in place

### `README.md`
Update, do not delete.
It still points to the repo structure and main scripts, but it explicitly says information is outdated and now underspecifies the Phase 8+ direction.
Recommended action:
- keep the repo overview
- update the “current status” paragraph
- add links to the current continuation plan

### `docs/phase8_9_execution_and_second_host_setup.md`
Update or archive.
It is operationally useful, but it is based on the earlier “late projection / second-host immediately” interpretation.
Recommended action:
- if you still want the shell commands, update it to match the new Phase 10/11/12 order
- otherwise move it to `docs/archive/`

---

## 3. Archive / rename instead of keeping as active planning docs

These are not wrong, but they are now conceptually stale.

### `docs/phase8_plus_experiment_plan.md`
Archive or replace.
Reason:
- it is built around “late masked projection” as the main transferable idea
- it jumps early into second-host experiments before fully isolating the mechanism
Recommended action:
- move to `docs/archive/phase8_plus_experiment_plan_initial.md`
- replace with the updated continuation plan

### `docs/phase10_second_host_literature_review.md`
Archive or revise substantially.
Reason:
- it assumes the main transferable object is “late low-frequency projection”
- that is no longer the best working interpretation
Recommended action:
- either archive it as an initial host-selection note
- or rewrite it around the new question:
  - which hosts support **soft-early / hard-late** insertion cleanly?

---

## 4. Safe delete candidates

These are the only docs I would currently consider deleting outright, and even these are optional.

### Delete only if you prefer a very clean docs folder
- obsolete one-off execution notes that have been fully superseded and are not useful as record
- duplicated local analysis notes copied into `docs/` if the same results are already summarized elsewhere more cleanly

At the moment, based on the repo docs I inspected, I would **not** aggressively delete the main planning/history docs.
I would mostly:
- keep the historical base docs,
- archive the now-stale continuation docs,
- and replace them with one updated active continuation plan.

---

## 5. Recommended immediate cleanup action

### Keep active
- `README.md`
- `docs/neurips_prdiffusion_experiment_plan.md`
- `docs/progress_report.md`

### Replace as active continuation doc
- replace active use of `docs/phase8_plus_experiment_plan.md`
- with a new updated continuation plan built around deferred hard consistency

### Move to archive
- `docs/phase8_plus_experiment_plan.md`
- `docs/phase10_second_host_literature_review.md`
- `docs/phase8_9_execution_and_second_host_setup.md` if not updated immediately

---

## 6. Suggested archive layout

If you want a tidy structure:

- `docs/archive/phase8_plus_experiment_plan_initial.md`
- `docs/archive/phase10_second_host_literature_review_initial.md`
- `docs/archive/phase8_9_execution_and_second_host_setup_initial.md`

Then keep one active document such as:

- `docs/continuation_experiment_plan.md`

This is cleaner than deleting everything and losing the planning history.
