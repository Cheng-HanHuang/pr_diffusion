# Lab-machine migration plan (keep one repo)

This note defines a clean migration strategy from the institution machine setup to the lab-machine setup **without creating a new repository**.

---

## Recommendation in one sentence

Keep a **single code repository** and migrate machine-specific differences into a small configuration layer (env files + wrappers), while preserving script names and experiment semantics.

---

## Why not create a new repo right now

Creating a second repo would immediately add avoidable overhead:

- duplicated bug fixes,
- diverging experiment scripts,
- broken provenance across runs,
- more difficult paper reproducibility.

Given that your core algorithms and workflows are unchanged, this is a **configuration migration**, not a research-fork migration.

---

## What should be machine-specific vs repo-global

### Machine-specific (should move to config)

- absolute paths (`REPO_ROOT`, `DATA_ROOT`, `OUT_ROOT`, `SPLIT_DIR`),
- environment name (`CONDA_ENV`),
- scheduler/account defaults,
- machine labels used in logs.

### Repo-global (should stay shared)

- Python methods/algorithms,
- canonical experiment scripts,
- metric definitions,
- split-generation logic,
- postprocessing logic.

---

## Minimal migration design

Create and standardize a small per-machine profile convention:

- `env/machine.institution.env`
- `env/machine.lab.env`

Each file exports:

```bash
export REPO_ROOT=...
export DATA_ROOT=...
export RUN_ROOT=...
export SPLIT_DIR=...
export CONDA_ENV=...
```

Then run scripts with:

```bash
source env/machine.lab.env
CONDA_ENV=$CONDA_ENV DATA_ROOT=$DATA_ROOT OUT_ROOT=$RUN_ROOT SPLIT_DIR=$SPLIT_DIR \
  sbatch scripts/slurm_neurips_phase6_main.sh
```

This keeps all existing scripts usable while removing machine-coupled editing.

---

## Branch/commit strategy for clean history

Use a dedicated migration branch and make small commits:

1. **docs commit**: migration policy + usage instructions.
2. **config commit**: add `env/*.env.example` templates.
3. **script hygiene commit**: remove any remaining hardcoded machine paths.
4. **validation commit**: smoke-run command examples for both machines.

This gives clear traceability and easy rollback.

---

## Decision rule for when to split into a second repo

Only create a new repo if at least one is true:

- the lab machine requires a fundamentally different pipeline,
- dependencies/toolchains cannot be reconciled,
- outputs/benchmarks are no longer comparable,
- you need an intentionally separate research direction.

If differences are mostly paths/env names/script wrappers, keep one repo.

---

## Immediate next actions (suggested)

1. Keep this repository as the single source of truth.
2. Introduce per-machine env profile files.
3. Update startup docs to be machine-profile based.
4. Run one pilot (`test_20`) from the lab profile and verify identical output schema.
5. Tag that run as the first official lab-machine baseline.

