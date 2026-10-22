# Track R · R0.3a canonical G1 return and R0.3b execution plan

**Date:** 2026-10-22. **Source:** Cheng's uploaded PAC receipt
`R0_3A_G1_RECEIPT.json`, generated from
`02751a48b570f40bc0647074608895be55e2726c`
on physical GPU **1**. The complete unabridged receipt remains
on PAC under its R0.3a run directory.

## Validated gate result — G1 PASS

- Case: `S3_a12|DAPS|68539|rep2`.
- Original RGB8 raw PSNR `6.363577842712402` dB.
- GT-best exact orbit was `111`, PSNR
  `31.87063217163086` dB, a **+25.5071 dB** recovery ceiling.
- Pre-registered GT-free prior chose `s_star=111`,
  achieving `31.87063217163086` dB. **G1_PASS**.
- Frozen score `S(000)=0.010259374409542943`;
  `S(111)=0.00987358222482726`.
  Paired margin `0.0003857921847156831`; `z=6.747459383066972`.
- Secondary `S_mid(000)=0.002792146900901571`;
  `S_mid(111)=0.0023136319759942126`.
- Frozen checkpoint/schedule match:
  `max_diff_backend_schedule=0.0`.
- 256 single-image denoiser forwards, 32 batch calls of size 8.
- Prior-only wall 9.302591344807297 sec, model load 0.86605 sec,
  peak torch-reserved 9288 MiB.
- Manifest SHA
  `8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890`.
- The receipt explicitly states `not_full_g0: true`
  and `s5_fast_runs_generated: 0`.

This is an **already known, preselected** exact-orbit case;
it demonstrates that the prior can repair it in the frozen setup,
not that the repair works on the general candidate distribution.
No confirmatory/held-out claim is warranted.

## R0.3b — next required before full R0

The exact one-case G1 outcome is favorable, but G0's last gate,
independent physical GPU / batch-size determinism, is still missing.

The next committed stage is a **bounded, eight-candidate smoke**:
- CPU freezes 8 UIDs from the validated 2520-candidate manifest.
  It includes the canonical G1 candidate, one S1 DAPS GOOD and
  NONORBIT_FAILURE selected by separate offline GT-stratum
  verification, one S1 SITCOM, one S2 NP, one S4 FAST and two
  fixed-SHA-ranked additional candidates.
- A stripped worker manifest contains **no GT and no PSNR**.
  Candidate file hashes and exact selection SHA are checked.
- On two distinct available Blackwell physical GPUs, score the
  same eight RGB8 candidates once at batch 8 and once at batch 64.
  Each worker performs 8 × 256 = 2048 single-image UNet
  forwards. The batch 8 worker also applies the pinned UnivFD
  CLIP/FC head to all eight orbit members plus the repaired
  member (72 images), checking the repeated repair score.
- The CPU adjudicator requires maximum relative difference
  ≤ 1e-5 for all orbit S, S_mid, and per-time averages, as
  well as identical selected orbit indices and candidate SHAs.
  It reports exact GPU hours and projects remaining prior
  scoring cost. The projective FAST generation cost remains
  explicitly **unmeasured**.

The autopilot stops with
`G0_FULL_PASS_PENDING_FAST_SMOKE_AND_BUDGET` if the
cross-GPU gate passes. It does **not** authorize R0.4/R0.5/R0.6,
nor attempt FAST generation. The remaining native FAST one-root
smoke, near-exact A1.5 replay and aggregate 16-GPU-hour
projection are required before those stages. Any determinism
failure escalates to the planner under the binding handoff.

Historical sources remain read-only; no path under the protected
B24 cohort may be accessed, listed, hashed or guessed.
