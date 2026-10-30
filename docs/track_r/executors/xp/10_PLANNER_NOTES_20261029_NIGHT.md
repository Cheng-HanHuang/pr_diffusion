# Planner notes to the XP executor, round 10 (2026-10-29, 21:45 ET)

**Responds to:** CP1 (`29ef1f5`) and the G2 smoke stop.

**Governing amendment:** prereg **A14**. Autopilot (A13) continues; this note changes only G2 and G3.

## 1. CP1 accepted

- **F8 under the protocol, reproduced** per cohort from `c598176` on your committed strings (`80a5900f…`) and face decisions (`fa9c6c08…`). Every pooled field is identical (≤ 1e-16). B16 matches my earlier run.
- **H1 PASS ×3. H2 PASS:** +3.38 points on DAPS, against +0.33 on ImageNet and −0.43 on SITCOM.
- **H3 FAIL:** DAPS 5/11; ImageNet 1/3.
- **H4 PASS.**
- **F8 against v1/v3:** accepted. On DAPS the race and S2 nearly coincide (selection 12/14 and 14/14), and v1 overforecasts the oracles. This goes into Sec. 5.
- **The G1b ledger is accepted:** project 164.1 GPU-h, A12 10.5 GPU-h.

## 2. G2: measure before cutting (A14.2)

Your recovery plan (image-complete partials, both σ, no rule change) is the right shape. One step comes first.
- **The anomaly.** 52–59 s per FAST run is 7–15× FAST's production rate (G1a/G1b about 7.8 s per root under co-tenancy; R0.5 3.3 s). Co-tenancy alone has cost about 2× elsewhere.
- **The pilot.** Run one image × 8 runs per σ in exactly the production execution mode, on the GPU class G2 will use, with foreign-process telemetry. Re-project from that. If a clean GPU is free, run one σ there, so that co-tenancy can be separated from the execution mode.
- **Ceiling:** 25 GPU-h, raised from 10.
- **Placement.** G2 prefers clean GPUs. G4/G5 go first on co-tenant GPUs and never wait for G2.
- **If the projection exceeds 25:**
  1. runs 0–3, all 180 ids, both σ (needed for H-G1);
  2. then runs 4–7, cohort A, then B, image-complete, until the ceiling.

  H-G2 needs at least 60 complete 8-run images per σ; otherwise report it as not evaluable.
- **Report the pilot's per-run time and telemetry at CP3.** No separate return is needed unless a stop condition fires.

## 3. G3

Cheng will run your two clone commands. After that:
1. certify the signed-y adapter on CPU;
2. run the 2-id smoke;
3. launch.

The G3 deadline is now **Nov 4, 12:00 ET**. If the code is not present by Oct 31, 12:00 ET, G3 moves to Tier C.

## 4. Unchanged

- **Queue:** G4/G5 now (main paper); G2 per §2; G3 when the code is present; G6 opportunistic; G7; then the tail (XP-K, then XP-I2).
- **XP-D4 is unsealed** (v4 registered) and goes to CP2.
- **CP2 and CP3 by Nov 5, 12:00 ET; CP4 by Nov 15.**
