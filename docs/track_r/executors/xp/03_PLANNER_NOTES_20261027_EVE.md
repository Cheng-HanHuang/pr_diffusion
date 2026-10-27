# Planner notes to the XP executor, round 3 (2026-10-27, 17:15 ET)

**Responds to:** issue #42 (comments 16:17–20:47Z) and the issue #43 handoff.

**Governing amendment:** XP prereg **A7** (procedural). Merge the planner branch at your next return point. Do **not** change the checkout of a running campaign.

## Acknowledged

**Published:**
- **A8b** at `c638b54`. The planner checked it:
  - its A8 columns are identical to A8;
  - the ρ order agrees with `e0_rank_order` in all 760 rows;
  - every C and I label agrees with its frozen τ, across all 3,240 runs.
- **D1 phase-1 strings** at `41d4121`, in `theory_inputs/`. The v2 clock runs until **Oct 29, 12:01 ET**.
- **XP-C/C2** CPU input freeze.

**A6 GPU run** (root `xp_a6_20261027T165112Z`, HEAD `b7af2f7`):
- smoke PASS;
- D1 phase 2: 1,200/1,200 (sealed);
- D3: 400/400;
- E′: 45/720.

It then stopped with `STOP_NO_CLEAN_GPU_FOR_30_MIN` because other users' jobs took the GPUs. That is a scheduling stop, not a failure. This run used 13.1 GPU-h; the cumulative total is about 52 of 400.

**Audits:**
- All 200 S1 DAPS candidates come from `B21_11_fresh2_final_val100_meas5401`. This matches A6.1, and the replay is still to do.
- The B16 RNG audit shows that B22 SITCOM used one sequential stream per image, starting from the post-model state.
- D3 and E′ seeds overlap the XP-T pilots (9 roots each). Record these as the same draws, not extra replicates.

## Decisions (prereg A7)

### 1. Resume E′ as an audited same-run resume

- Run the existing supervisor in `--mode main` on the original root, at the pinned HEAD `b7af2f7`. Do not use `xp_a6_launch.sh`.
- First verify the 1,645 completed receipts. Write an append-only resume log.
- Admission or scheduling code may change between resume attempts, provided the worker code, manifests, seeds and environment stay byte-identical. Record the controller version per job.

### 2. Co-tenancy for generation stages (A7.1; on by default, Cheng may revoke it)

A generation-only job may share a GPU with other users' jobs if:
- free VRAM is at least its request (about 1.5 × peak + 2 GiB);
- device memory after admission stays at or below 80%;
- it is the only Track R worker on that GPU;
- Track R holds at most **2 GPUs** while foreign jobs are present.

Further rules:
- Never touch foreign processes. Record co-tenancy per job.
- Replace the 30-min exit with wait-and-retry, under a hard wall-clock limit (for example Oct 29, 08:00 ET) and a stop file.
- XP-T timing and replays that must be bitwise stay clean-only.

### 3. CRITICAL lane, after or alongside the E′ resume

1. **XP-B16, with the A7.2 lineage gate.**
   - For every B22 image: regenerate runs 0–3 from the post-model state; check them against the historical candidates (hash, or else 0.1 dB PSNR with an identical label, scored offline); then continue the same stream through runs 4–15.
   - DEV80 runs 4–15 use independent native seeds.
   - Prefer clean GPUs for B22.
   - Cost: about 23 GPU-h.
2. **The 8-root DAPS replay at 400 × 5 on a clean GPU, then XP-F8** with the A7.3 seeds, about 12 GPU-h.
3. **XP-C and XP-C2 GPU workers:** smoke, then run.
   - The XP-C CLIP worker also scores the 400 D1 phase-1 candidates with the G1 and G2 prompts.
   - Commit those decisions as a **new** file in `theory_inputs/` before Oct 29, 12:01 ET if possible. Do not modify the committed strings.

### 4. CPU work now

1. **Score XP-D2**: the published DAPS-4K protocol, with the per-metric oracle, against first run, E0, E1_CLIP (G1) and O_orbit.
2. **Score XP-D3**: anatomy and policies.

   Both are allowed before any freeze. Commit the aggregates to `xp/returns/`.
3. **A7 ZIP and the A5.5 historical summary.**
   - A5.5 is now cleared for the theory executor, because mech v1.1 is registered. Commit it to `xp/returns/` and a copy to `theory_inputs/`.
   - The ZIP goes to Cheng.
4. **The post-freeze analysis script** must be ready and validated on K ≤ 4 by **Oct 28, 18:00 ET** (the v1 target):
   - use the A6.4 panels;
   - compare against the registered variants with predictive intervals;
   - accept per-image forecast files (the theory executor may supply one), so that an image failing the B16 gate can be dropped.

### 5. BACKGROUND lane

XP-K (the SITCOM and FAST parts) → XP-B32 (continues the B16 stream through runs 16–31) → XP-G → XP-I and XP-I2 (wrapper first) → XP-H → XP-G2 → XP-J → XP-D4.

## Reporting

- **Return when:**
  - E′ has resumed or completed;
  - the B16 lineage gate has a result;
  - the DAPS replay has a result;
  - XP-C/C2 are done;
  - D2 and D3 are scored;
  - A5.5 is ready;
  - any stop condition fires.
- **Every return:** a digest of at most 20 lines, in issue #42 or a new issue (link #43).
- **Until the matching freeze is registered**, post no A16, A64, B16 or F8 outcome anywhere, and no D1 phase-2 outcome before v2 (or 48 h).
- **E′** may be scored once complete (it is allowed), but commit it to `xp/returns/` only. The planner will not relay it to the theory executor before v1.
