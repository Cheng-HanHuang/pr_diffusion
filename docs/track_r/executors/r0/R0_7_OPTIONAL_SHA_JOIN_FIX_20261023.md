# R0.7 CPU join stopped at optional measurement hash — diagnostic and repair (2026-10-23)

## PAC evidence

Attempt root:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_7_cpu_20261023T060703Z`

- Git HEAD: `f7f73dc73b1c65fcc842115d0c7f5ef6b3c21f88`.
- CPU suite `74 passed`, `R07_CPU_TEST_RC=0`; launcher started.
- Final state `STOP_R07_JOIN` with `cpu_join RC=8`.
- The failure is `STOP_R07_ORIGINAL_TASK_GT_JOIN S1_b22|DAPS|61978|fresh1`.
- Stage stopped **before an offline GT catalog was written or a P1–P4
  result was computed**. R0.4's 2520 scores and R0.6's 720 scores remain
  unchanged. No GPU was used.

## Root cause established from committed manifest producer code

`scripts/track_r/r0_2_build_manifest.py::value` accepts an
optional `measurement_sha=None`; the historical S1 DAPS manifest
is produced without specifying it. The original R0.2 CSV therefore
records blank `measurement_sha256`, even though candidate identity,
signed-y path, and source GT source are fully specified.

`scripts/track_r/r0_4_prepare_cpu.py::tasks` opens each
signed-y file, **computes its actual SHA256**, and freezes that
mandatory hash into the 2520-item GT-free scoring tasks. R0.4
validated this same value in every candidate score receipt.

The first R0.7 join erroneously required the optional R0.2 field
to exactly equal the verified R0.4 digest. Missing-vs-computed was
treated as a mismatch. This was a join-schema error, not evidence
of a wrong observation, source, scored reconstruction, or GT.

## Narrow repair

`scripts/track_r/r0_7_prepare_cpu.py` now:
- Enforces exact UID, source, family, image ID, run index,
  candidate path, candidate format and signed-measurement path.
- Enforces `measurement_sha256` exact equality whenever the R0.2
  row supplies a nonblank digest. Any wrong populated hash STOPS.
- Accepts an explicitly missing R0.2 SHA only against a
  nonempty, well-formed 64-hex-character R0.4 verified digest.
- **Recomputes the SHA256 of each distinct, explicitly listed
  exposed signed-measurement file**, caches it, and requires
  equality to the frozen R0.4 digest. No candidate-or-GT path
  mismatch is bypassed.
- Records how many original rows lacked the optional SHA in the
  offline join receipt and catalog for audit.
- Includes synthetic tests for allowed omission, mismatch,
  changed candidate/measurement path, malformed frozen digest
  and absent UID.

No R0 threshold, scientific definition, source-cohort restriction,
model output, or original frozen result is modified.
The earlier failed output root is preserved for provenance;
the repaired R0.7 run must use a new output root and a new pinned
Git HEAD. CPU tests must pass before the new job is launched.

R0.7 scientific P1–P4 outcome remains unknown and is left for
the Track R planner to adjudicate only after the CPU evaluator
successfully produces an immutable report.
