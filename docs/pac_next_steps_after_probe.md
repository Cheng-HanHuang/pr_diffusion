# PAC next steps after the 10-image validation probe

This note records the recommended next steps after the PAC-based 10-image Phase-1 probe.

## Main conclusion from the probe

The 10-image probe is enough to move beyond very small-data testing.

Practical takeaways:
- Use **radius = 0.5** as the current primary working radius.
- Keep **radius = 0.2** as the main secondary/check radius.
- Do **not** spend more time on additional tiny sanity studies before moving forward.

The queued full cluster Phase-1 validation should remain in queue as a **confirmation run**, not a blocker.

## Recommended strategy

Do a **staged PAC migration**, not a full all-at-once migration.

That means:
1. Keep using PAC as the primary active experiment machine.
2. Copy only the data subsets required for the next phase.
3. Scale up in the following order.

## Stage 1: PAC as active development machine

Already done:
- repo present on PAC
- environment working on PAC
- Hugging Face model loading works on PAC
- 10-image probe dataset present on PAC
- canonical comparison script runs on PAC

## Stage 2: immediate next PAC experiments

Run these first on PAC at **radius = 0.5**:

1. **NP schedule tuning**
   - start with the validation-10 subset
   - this is the most useful next tuning study

2. **Mechanism ablation**
   - also on validation-10
   - compare full / score-only / projection-only / no-masking

Recommended secondary check:
- rerun the most important setting at **radius = 0.2** afterward if needed

## Stage 3: moderate scale-up on PAC

Before running bigger experiments, copy only the next needed splits to PAC:

- `validation_25.txt` image set
- `test_20.txt` image set

Do **not** move the full 5400-image folder yet unless there is a concrete need.

Rationale:
- smaller staged copies are easier to manage
- they avoid unnecessary storage clutter
- they are enough for the next useful experiments

## Stage 4: early main-comparison pilot on PAC

After Stage 2 looks stable:

- run a **test-20** main comparison on PAC
- use the currently frozen radius = 0.5
- keep the cluster full Phase-1 validation queued in the background

This gives an early stronger comparison without waiting for the cluster.

## Stage 5: broader migration only if needed

Only after the above runs are successful should you consider:

- moving the full validation/test subsets more broadly
- or copying the larger image pool to PAC
- or making PAC the full primary machine for all remaining experiments

## What not to do now

Do not:
- keep doing more 5-image or 10-image radius studies
- wait idly for the cluster Phase-1 job before progressing
- copy the entire 5400-image dataset immediately unless needed

## Priority order

1. Continue PAC work.
2. Freeze current main radius at `0.5`.
3. Run PAC NP schedule tuning.
4. Run PAC mechanism ablation.
5. Copy `validation_25` and `test_20` image subsets to PAC.
6. Run a PAC test-20 main comparison.
7. Use the full cluster validation later as confirmation.

## Practical note on naming

Use neutral names on PAC, for example:
- output root: `phase_retrieval_YYYYMMDD`
- subfolders: `validation_probe_lab`, `schedule_tuning_lab`, `mechanism_lab`, `main_compare_lab`

Avoid workflow names that look like venue plans.
