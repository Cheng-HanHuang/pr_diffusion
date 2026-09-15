# B26 checkpoint report

## Current gate

`B26.0 IMPLEMENTATION / PAC MANIFEST FREEZE PENDING`

Signed base `c906d36e0e396a6abbf761e3d65c91433428170f` and ancestry were verified before branch creation. Branch `codex/b26-np-conditional-correction` and draft PR #40 were created from that exact base. No B26 scientific execution has occurred at this checkpoint.

## Required pre-run evidence

Before B26.1/B26.2 launch, the executor must publish a pushed B26 pre-run SHA containing:

- `configs/b26/b26_spec.json`;
- PAC-derived `configs/b26/b26_dev80_manifest.json` with all 80 DEVELOPMENT identities, exact measurement/GT hashes, four NP root seeds, accepted NP4 terminal replay records, DEV16 hash ranking, and smoke subset;
- B26-owned native H/R runner and launch/status/resume/stop scripts;
- B26.2 finite-estimator implementation;
- zero-GPU integrity tests;
- analysis/bootstrap and packager;
- exact model/DiffFPR/source identities and resource limits.

## Scientific gates after pre-run

1. B26.2 CPU may launch independently after zero-GPU tests and pre-run identity pass.
2. B26.1 smoke launches only on explicit physical GPUs passing UUID/free-memory/process gates.
3. H smoke must replay the accepted B24 NP4 root terminal identities; R must pass target-separation/lineage/count/finite-output gates.
4. DEV16 completes only after smoke PASS; DEV80 completes only after DEV16 integrity/resource and projected 24-GPU-hour reservation gate PASS. PSNR direction is never a continuation gate.

## Locked boundaries

B24 closed; confirmation locked; no new FFHQ measurements; no weighted FFHQ execution; no external baseline rerun; no historical mutation.
