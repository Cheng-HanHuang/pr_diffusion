# A13 autonomous Track R executor — current campaign contract (2026-10-29)

**Governing:** prereg A13, planner branch `988aaed`, `docs/track_r/executors/xp/09_PLANNER_NOTES_20261029_EVE.md`. This supersedes the earlier A12-only version of this file. The planner has independently accepted D1-P2, B16 v1/v3, and G1a. Native **G1a** 1,600/1,600 generation is complete. Later stages have **implementations, not yet PAC-certified GPU results**. No claimed results without run receipts.

## Campaign caps, scientific release, order

- Project cap **400 occupied GPU-h**. Conservative project total after G1a **157.154512636 GPU-h**, including a **31.314365 GPU-h upper charge** for an older unresolved B16 process (not measured actual wall time). Historical source-backed ledger SHA256: `3eabf7b978e1341549f72690061d65d2aee1c29046c0749df0e2b821d74a2f8c`.
- **A12 cap = 150 GPU-h exclusively for G1a–G7.** A10's 55.545785 GPU-h counts against the project cap **but NOT A12**. G1a charged 3.572931 GPU-h; remaining A12 authority **146.427069 GPU-h** as of G1a.
- Registered theory **v4** at `149331c319bb8697c1a0f40c04bf0e4016ec3420`, JSON SHA256 `b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3`, recorded **2026-10-29 20:02:30 UTC**. XP-D4 run ≥16 is **unsealed**. The GPU workers remain GT-free; only the authorized CPU analysis may score it.
- Mandatory no-gap order: **G1b now → G2 → G3 → G4 → G5 → G6 → G7.** G1a is already complete. G7's Nov 15 is a **deadline**, not a delayed start.
- Stage caps: G1b 40, G2 10, G3 45, G4 30, G5 40, G6 1, G7 4 GPU-h. If one stage hits its cap, **retain complete-image partial panels and continue**; never silently increase the budget or alter a frozen method. Corrected A12 and 400 project caps must be checked before every admission and include occupied partial attempts.
- After A12 is complete or running and a GPU is available: Tier-C tail with separate **60 GPU-h cap**, in order **T1 XP-K** (A5.4 exactly; 30 stage cap), then **T2 XP-I2** (40 stage cap, only if G3 smoke passed). **Cancelled:** B32, XP-H, XP-G2, XP-J.

## Implemented entrypoints and remaining validation

- `scripts/track_r/xp_a13_launch_now.sh`: creates a **separate isolated pinned** A13 worktree, launches G1b, B16 protocol, F8 CP1 CPU, and G2–G5 input preparation/queue handoff. This script MUST run against the **same immutable commit** with a clean worktree, no prior A13 worktree, and PAC-visible source models/measurements. It has not been run remotely by this chat.
- `xp_a12_g1b_prepare_cpu.py`, `xp_a13_g1b_autopilot.py`, `xp_a12_campaign_fast_gpu.py`: 100 images × reps 32–63, exact existing FAST semantics, pinned v4 registry, zero GT in generation, separate shared campaign lock and dual GPU ledger.
- `xp_a13_after_g1b_prepare.sh`, `xp_a12_g2_prepare_cpu.py`, `xp_a12_campaign_autopilot.py`, `xp_a12_g45_prepare_cpu.py`, `xp_a12_g45_daps_gpu.py`: after G1b terminal receipt, paired-noise G2 (360 frozen y, 2,880 FAST roots, two sigma smoke gates), conditional four-particle PC-DAPS G3, two distinct native DAPS historical replays each for G4 and G5, 80% VRAM and charged per-attempt accounting. If G3 code is missing, request the exact clone from Cheng once in issue #42; do not use bundled FFHQ.
- `xp_a12_g3_prepare_cpu.py`, `xp_a12_g3_pcdaps_gpu.py`: exact PC-DAPS `59b2f046` 180 four-particle bags; four native particles are retained and reported as a **mean of particle metrics**, never a made-up fifth reconstruction. Conditional on code being present and 2-id smoke passing.
- `xp_a12_g6_prepare_cpu.py`, `xp_a12_g6_clean_autopilot.py`, `xp_a12_g6_daps_gpu.py`: G6 batch-1 **native F8 DAPS 400×5** FFHQ and FAST ImageNet clean timing, only missing clean pairs, 1 GPU-h stage cap, shared A12/project ledger. Do not substitute DAPS-4K for F8 400×5.
- `xp_a12_g7_c_clip_cpu.py`, `xp_a12_g7_c2_gpu.py`, `xp_a12_g7_c2_autopilot.py`: frozen diagnostic GT-only CLIP orientations and C2 denoiser equivariance profiling. Other frozen prior/UnivFD orientation arms remain distinct dependencies; G7 is not complete merely because C2 finishes.
- Tail **T1/T2 automatic controllers** are still implementation/admission work. Their authorization is A13, but implementation must not be presented as a completed smoke or a running GPU process.

## CPU protocol and checkpoints

For every panel: **score → A9 schema strings and per-candidate orientation choices (copy to `theory_inputs/`) → all registered forecast comparisons → frozen protocol v1 from `c598176`, per-cohort tau and image-weighted pooling → prereg hypothesis table → commit `xp/returns/`.**

- **CP1:** B16 protocol plus F8 v1/v3 comparisons and protocol, and project GPU ledger; B16 reference provisional pooled success **0.8186**, mean runs **3.290**, gain **−0.0043**; reproduce within **0.002** before reporting.
- **CP2:** ImageNet XP-D4 against v2/v4 and protocol at n = 32/64.
- **CP3:** XP-G paired-noise, E′8, D2-8, PC-DAPS XP-I.
- **CP4:** Tier C including XP-C/C2, XP-K, XP-I2.
- Return to planner **only** at CP1–CP4 (each ≤20 lines). CP1–CP3 due **Nov 5, 12:00 ET**; CP4 due Nov 15.

## Five A13.4 stop conditions

Ask planner/Cheng only if (1) a frozen rule cannot be applied, (2) data-governance conflict, (3) a cap would be exceeded, (4) frozen bytes or receipt mismatch, or (5) Cheng must act (clone, permissions, disk). For an infrastructure-only fault: one documented retry may be authorized without asking, but not after changing the science implementation or frozen settings.

**No code in this branch constitutes PAC runtime verification.** Gate, smoke and terminal receipts govern admissibility and claims.
