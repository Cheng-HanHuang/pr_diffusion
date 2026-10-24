# R0-N N2 CLIP zero-shot orientation: final supplemental return

R0N original N0/N1/N3 published at 5af47dbb4f6dba5ba5aa140b9b30677039b31e39.
This addendum supplies only optional N2; original reports unchanged.

## Provenance
- Frozen N2 source candidate manifest SHA256: 7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b
- Frozen original N2 preparation HEAD: b1637f7c27a3caaa03a8b204171fca0d3fccfcd5
- Corrected evaluator HEAD: 3f42c62825a135982a82aaec73e7c7f665a5e3d7
- Pinned UnivFD CLIP source changed only in text-encoding wrapper;
  the external repository and checkpoint were never changed.
- First failed stage wall: 6.83 seconds (no candidates scored).
- Recovery GPU stage wall (including imports): 29.758 seconds.
- Recovery instrumented GPU score wall: 27.588 seconds.
- Combined upper bound (stage wall): 36.588 seconds / 900.

## Frozen K2 comparison
GLOBAL errors only; raw partial-orbit failures are recorded
separately; all successes and harms use the precomputed R0.7 PSNR.

| Selector | GLOBAL errors | Repaired | GOOD harm |
|---|---:|---:|---:|
| CLIP | 84 | 84 | 0/2040 |
| PRIOR_GLOBAL | 84 | 83 | 0/2040 |
| PRIOR_MID_GLOBAL | 84 | 84 | 0/2040 |
| UNIVFD_GLOBAL | 84 | 71 | 108/2040 |

The frozen planner, not this executor, adjudicates K2:
CLIP versus prior-global success, GOOD harm and relative compute cost.
No claim of a new solver, R1 authorization or protected validation.

## Artifacts
- R0N_N2_CLIP_COMPARISON.json: all source/family cluster CIs
- R0N_N2_COMPLETE_RECEIPT.json: frozen offline-evaluation receipt
- R0N_N2_RECOVERY_README.md: this return
