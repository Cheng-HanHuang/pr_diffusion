# Track R R0.2 CPU autopilot stop and repair — 2026-10-22

## Status for Cheng

The PAC-owned unattended CPU run
`r0_autopilot_20261022T081105Z` completed
the initial R0.1 suite **15/15 PASS** and stopped at the
R0.2 manifest build (RC 1) before G0 and before any GPU work.
The run receipt confirms zero GPU forward calls and a clean pinned
Track R SHA `d60a0958bd3ef84e707d641ba8568ee734638759`.
All historical outputs remain read-only. The protected B24 cohort
has not been accessed.

The first manifest blocker was missing
`B24_3_dev80_overnight_.../workers/gpu1/task001_A_24322/
methods/NP4_INDEPENDENT/terminal_0.pt`.
This is **not** evidence that the reconstruction failed.
The accepted B26 exposed development manifest explicitly records
`accepted_np4_result_path` for image `24322` under
`B24_3_pilot16_20260913T045410Z/.../NP4_INDEPENDENT/result.json`,
because the later DEV80 task **reused** the completed Pilot16 NP4
runs rather than regenerating them.

This provenance is corroborated by
`scripts/b24/run_b24_3_dev80_image.py`:
`source_np_results` is read and validated when `run_np`
is false, and the completed DEV80 task's `np_results` carries
the adopted result paths. Fresh DAPS4/SITCOM4 baselines still reside
under each new DEV80 task's `baseline/` tree.

## Committed source adapter fix

`s2_accepted_np_source` resolves the native
four `terminal_{i}.pt` files beside the precise accepted NP4
`result.json`, not in the DEV80 worker's directory. It checks:

- The task completion's `np_results.NP4_INDEPENDENT`
  equals the pinned accepted result path.
- The task completion's `np_pilot_result_reused`
  and `np_execution_performed` agree with the
  recorded `pilot16` flag.
- Reused candidates originate from the exposed
  `B24_3_pilot16_` namespace; fresh candidates from
  `B24_3_dev80_overnight_`.
- Exact accepted result-file SHA-256 equals the source index;
  `status=PASS`, `arm=NP4_INDEPENDENT`, and four terminals.
- Each accepted native terminal's `terminal_index`,
  `reconstruction_path`, tensor-content SHA-256 and
  recorded raw PSNR agree with the corresponding frozen
  `np_roots` entry.

A discrepancy is a stop, **not** permission to search arbitrary
directories, fabricate run metrics, substitute GT oracle winners,
or regenerate historical NP4 candidates.

Synthetic tests for Pilot16 and newly generated DEV80 terminal
paths, SHA mismatch, source provenance disagreement and
missing-terminal failure were added to `tests/track_r`.

## Next

Restart the unchanged CPU-only autopilot on the new pushed
Track R SHA with a **fresh** output directory. Expected path:
R0.1 tests → full S1–S4 manifest → G0 CPU validation.
A new source-schema mismatch must stop and be audited;
no G1 or GPU computation is authorized yet.
