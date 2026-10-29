# A12 complete-campaign autopilot — execution specification (2026-10-29)

**Status: design / unlaunched.** The G1a controller is implemented and **its production is complete**. G2–G7 are *not* yet implemented as independently smoke-validated workers and must not be represented as runnable. This file specifies the single-launch control plane and required fail-closed worker adapters. Governing authority is planner A12 (2026-10-29, 12:35 ET) at `65ba723`, `docs/planning/track_r/XP_PREREG_v1_1.md` and `docs/track_r/executors/xp/08_PLANNER_NOTES_20261029_PM.md`.

## Status and caps

- Project lifetime: 400 occupied GPU-hours. Conservative source-backed ledger before G1a: 153.581581288 GPU-h, SHA256 `3eabf7b978e1341549f72690061d65d2aee1c29046c0749df0e2b821d74a2f8c`. This total contains 31.314365 upper-billed GPU-h for a historical uncertain worker, **not measured actual runtime**.
- A12 campaign: 150 GPU-h inclusive of A10's 55.54578541457511 and G1a's 3.572931348127313, leaving **90.88128323729758 GPU-h** of campaign authority; overall project conservative remaining **242.84548736419086 GPU-h**.
- G1a: `xp_a12_g1a_prod_20261029T180624Z`, 1,600/1,600 complete and sealed, replay 2/2 pixel exact, zero failed attempts, controller head `c15651b152198d0bbd014c279499b2c5a25814b3`. Stage cap 25; actual stage use 3.572931348127313.
- The sum of *future* individual stage ceilings exceeds 90.881 A12 remaining. A scheduler **must enforce BOTH campaign cap and per-stage caps**; the stages are maximum allowable consumption, not guaranteed allocation.

## The only authorized queue

| Order | Stage | New GPU outputs | Local cap, GPU-h | Required admission | Target deadline (ET) |
| --- | --- | ---: | ---: | --- | --- |
| 1 | G1a XP-D4a | 1,600 FAST IN runs 16–31 | 25 | original D1 2-root replay | **DONE**, outcomes sealed |
| 2 | G2 XP-G | 2,880 FAST FFHQ, paired-noise σ .02/.10, n8 each | 10 | pre-specified paired y / SHA, 2-root smoke **per σ** | Nov 3 noon |
| 3 | G3 XP-I | 180 FFHQ bags ×4 PC-DAPS particles | 45 | exact source `59b2f046`, own native PR command, 2-id smoke; code absent ⇒ publish exact clone command, **skip**; projection >45 ⇒ cohort A only | Nov 3 noon |
| 4 | G4 XP-E′8 | 720 DAPS-4K FFHQ runs 4–7 | 30 | original E′ 2-root replay and same y/seed/checkpoint | Nov 4 noon |
| 5 | G5 XP-D2-8 | 400 DAPS-4K ImageNet runs 4–7 | 40 | original D2 2-root replay and same y/seed/checkpoint | Nov 4 noon |
| 6 | G1b XP-D4b | 3,200 FAST IN runs 32–63 | 40 | only after G1a–G5 launched and conservative campaign projection fits; otherwise remain n32 | Nov 4 noon |
| 7 | G6 XP-T | conditional clean batch-1 timing (18 runs) | 1 | only missing pairs, **clean GPUs only** | opportunistic |
| 8 | G7 XP-C/C2 | orientation/defect supplemental runs | 4 | exact frozen GT cohort, evaluation-only | Nov 15 |

**NEVER authorize automatically:** B32, XP-K, XP-H, XP-G2, XP-J, XP-I2.

## Controller contract — target single PAC launch (not yet runnable)

1. Read-only CPU preflight: verify code/manifest commits, A12 prereg and all source SHAs, immutable generation-only G1a final status, full historical ledger, four GPU visibility and allowed source ids. Reject project or A12 cap violations, overlapping worker processes, failed checksum, or missing stage adapters.
2. Each stage adapter must publish independently pinned `PREPARED`, `SMOKE_PASS` / `REPLAY_PASS`, `ADMITTED`, per-attempt `COST`, `CHUNK_PASS` and `FINAL` receipts. Failed replay never promotes to production. Replays may use bit-exact comparison, or 0.1 dB raw PSNR AND identical frozen labels under A12/A10.1. Replays and smoke are charged.
3. Use a single controller-owned PID and lock, atomic durable controller state, no foreign-worker signalling, concurrency ≤4 GPU workers, source-specific memory reservations, GPU memory ≤80%, stop file and SIGTERM drain. Count each attempt from occupied GPU allocation start to child exit, including failures and retries; stop and report partial on deadline or any cap.
4. Queue next eligible stage whenever a GPU is free but do not bypass required order/dependency. G2 per-σ smoke, G3 conditional code presence and cohort-A reduction, G4/G5 individual replays, and G1b projection eligibility are **different gates** and cannot be inferred from G1a PASS.
5. Before each admission, compute conservative cumulative **campaign** use including A10, all completed A12 stages, and currently running intervals, plus projected remaining work; check project 400 cap too. No separate per-stage controller may spend the same budget independently.
6. Distinguish **generation** from **scoring**. XP-D4 run ≥16 outputs and labels/GT stay sealed until theory v4 registered/declined OR November 2, 12:00 ET. Other-stage CPU scoring proceeds only under A12's frozen hypotheses, on a separate CPU lane. Never expose a protected DAPS/FAST/PC-DAPS bundled FFHQ dataset.
7. Progress, completion and failure receipts go to the shared `xp/returns/` handoff path (via independent publishing worktree), with exact SHA256, stage cost, source/environment pins, cohort coverage, and one ≤20-line planner return per genuinely informative gate. Routine chunk completions should not require conversation turns.
8. If a required stage adapter is absent or untested, **stop that stage and publish a blocked receipt**, not a fake success. A central dispatcher alone does not make the unimplemented stages safe to launch.

## Immediate scientific priorities alongside GPU work

- D1-P2 first held-out protocol and exact G1/G2 theory-v4 inputs are **already committed**: `5f6b213`.
- B16 v1/v3 prospective 2880/2880 and theory inputs are **already committed**: `8798462`.
- Next CPU outputs: B16 two-cohort frozen protocol with N2 face-set orientation and `τ_{B22}=0.0448503`, `τ_{DEV80}=0.0449706`; then F8 v1/v3 and two-cohort protocol with `τ_{B22}=0.0474649`, `τ_{DEV80}=0.0475986`; no retuning.
- Theory v4 must be given the **unsealed 100×16 D1-P2** files immediately, without the sealed G1a run16–31 outcomes. Register forecast before lifting D4 embargo whenever possible.
