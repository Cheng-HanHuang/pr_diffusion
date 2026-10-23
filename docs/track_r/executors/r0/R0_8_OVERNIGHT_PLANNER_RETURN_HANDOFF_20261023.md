# Track R R0.8 — safe overnight planner-return autopilot
Date: 2026-10-23 EDT, following successful CPU R0.7 retrospective evaluation.

## Confirmed PAC receipt

R0.7:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_7_cpu_20261023T080434Z`

Operator-reported:
- `R0_7_ANALYSIS_COMPLETE`, 3,240/3,240 historical + FAST scored reconstructions.
- Both `cpu_join` and `cpu_analysis` RC=0, zero active workers.
- R0 compute used: **3.4309563972961366 aggregate GPU-hours** (ceiling 16).
- R0.7 missing tau groups = 0.
- R0.7 original PSNR anomalies = 0.
- Historical optional SHA fields reconciled: **1400/2520** R0.2 manifest rows; every source signed observation SHA reverified against frozen R0.4 tasks.
- R0.7 GT catalog SHA:
  `139dc7bd3fc826d512149a4b9509757baefa1243ed35e8cb707e4bde1dba13b9`.
- R0.7 executor HEAD `c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b`.
- The actual P1/P2/P3/P4 findings **have not been shown in chat**;
  the overnight R0.8 consumes them directly from immutable PAC files.

## R0.8 work authorized unattended

The operator requested to run through remaining tasks as far as
possible overnight and to diagnose problems afterward. This is an
instruction to finish **scientifically legitimate, frozen R0 work**;
it is *not* a license to access confirmation data, redefine outcomes,
or invent a new R1 protocol.

New script: `scripts/track_r/r0_8_overnight_cpu.py`.
New tests: `tests/track_r/test_r0_8_overnight.py`.

The worker:
1. Confirms exact Git HEAD, frozen R0 v1 config, CPU-only
   `CUDA_VISIBLE_DEVICES=""`, completed R0.7 3,240-row receipts,
   output hashes, no protected access and no PSNR/tau anomalies.
2. Reads **only** R0.7's previously evaluated JSONL + aggregate
   JSON and report. Does not open GT paths, candidate paths,
   source measurements or any protected cohort.
3. Independently replays all 3,240 raw labels and eight-orbit
   PSNR/repair identity, P1 categories and P2 numerators, and all
   E0/E1/E2/E3/O_raw/O_orbit P4 outcomes and primary summary
   statistics for every source/family/K.
4. Independently recomputes the pooled ORBIT_ERROR success/harm
   denominators and the five pooled FAST/DAPS P3 AUROCs
   (prior S, image-centred S, UnivFD, signed residual, clipped residual).
   A discrepancy causes `STOP_R08_...`, with no silent adjustment.
5. Mechanically evaluates only pre-registered D-material
   (>=0.20), D-repair (success >=0.90 and harm <=0.01 PASS;
   success <0.75 or harm >0.03 FAIL), and D-detect
   (S or S_img minus UnivFD >=+0.03 or >=-0.02
   on BOTH FAST and DAPS). Mark outputs **provisional**.
6. Creates a clearly EXPLORATORY casebook of greatest PSNR
   gains, harms, attainable-orbit and unrecovered-oracle
   gaps from the already frozen rows, deterministic by UID.
7. Creates a sanitized planner ZIP including complete
   R0.7 analysis/report, R0.8 scientific audit, decision
   inputs, P4 CSV and casebook. **Never include R0.7 offline
   GT path catalog or detailed GT JSONL in ZIP.**
8. Finishes with `R0_8_PLANNER_HANDOFF_READY` and an
   immutable ZIP SHA-256 and receipt.

All tasks are CPU-only; there are zero additional GPU requests.
A source identity mismatch or scientific replay failure returns
nonzero immediately, retaining the original R0.7 output unchanged.

## Research handoff and stop boundary

The frozen `docs/planning/track_r/R0_EXPERIMENT_SPEC.md`
explicitly reserves D-material/D-repair/D-detect and any R1
direction to the **Track R scientific planner**. R1 cannot be
autolaunched from a provisional threshold check.

At the end of R0.8, the operator can attach:
`.../r0_8_overnight_<UTCSTAMP>/R0_8_PLANNER_RETURN.zip`.

The planner must read results and conditional branch:
- D-material and D-repair PASS → R1 repair-and-detect on FAST.
- D-material fail, repair PASS, detect superior → R1 prior detection/restart.
- D-repair FAIL, detect superior → R1 detection-only.
- D-repair FAIL, detect not superior → stop Track R and assess note/drop.
- Grey zone → scientific planner weighs complete tables and CIs.

Before any R1 GPU execution, the planner must freeze a separate
R1 protocol and cite the research/narrative novelty relative to
existing FAST-DIPS and FA-FPR work. B24 confirmation305 remains
strictly locked through R1 and may only be released by an R2
planner decision.

This document is a *planned automated handoff*, not evidence of
R0.8 execution. Do not assert completion until the operator reports
a verified `R0_8_PLANNER_HANDOFF_READY` receipt.
