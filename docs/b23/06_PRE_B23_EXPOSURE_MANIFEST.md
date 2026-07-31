# Pre-B23 exposure and future split protocol

## Conservative exclusion rule

`manifests/b23/PRE_B23_EXPOSURE.csv` covers B19, B20, early NP/SITCOM Branch A/B, B21 development
and validation, the B21.11 final benchmark, B22 fixed-baseline evaluation, failure/atlas inspection,
and planning evidence. If image or measurement exposure is uncertain, it is exposed.

Required columns are exactly:

```text
image_id, measurement_id, dataset_split, first_project_stage, roles_seen,
ground_truth_inspected, artifacts, exclusion_reason, source_evidence
```

`UNKNOWN_ALL_MEASUREMENTS` means every measurement for that image is excluded. This is deliberately
stronger than pretending an unresolved measurement tag is a seed.

## Two-step construction

The pre-run repository manifest is a conservative seed containing the known early 25-image panel,
documented B19/B20 examples, B22 replay image, and manually classified B22 failures. The zero-GPU
PAC collector then merges:

- B21.11 panel, execution, locked-measurement, and checked-panel manifests;
- B21 summary and B22 failure taxonomy;
- bounded, Git-indexed text/config/manifest paths in the historical checkout;
- any exact measurement seed attached to the same manifest record.

It scans no dataset, model, environment, cache, or output tree recursively. Target evidence files
are individually capped at 2 MiB; the historical Git-index text pass is bounded by file count,
individual size, and total bytes.

The post-run manifest, coverage counts, missing-source list, and SHA-256 are committed from the
capsule. A required source missing is a B23.0 stop.

## Allowed and prohibited reuse

Exposed items may support historical evidence, reader debugging, or a retrospective sanity check
after a choice is frozen. They may not select schedule/boundary, donor threshold, controller feature,
stopping rule, promotion, or final claim.

## Future registry

`manifests/b23/future_split_registry.csv` is intentionally header-only. Its schema is
`schemas/b23/future_split_registry.schema.json`. The one- and four-image B23.1 smoke templates are
also empty. B23.0 assigns no new image, measurement, seed, GPU, or method schedule.

After separate authorization, each row must be assigned once before execution, prove image and
measurement disjointness from `PRE_B23_EXPOSURE.csv`, carry a source-manifest hash, and receive
independent measurement and solver base seeds. The full final-plan split names and minimum sizes are
encoded in the schema, but no rows are populated here.

Any unresolved pre-B23 item blocks a proposed overlapping split; it does not permit a guess.
