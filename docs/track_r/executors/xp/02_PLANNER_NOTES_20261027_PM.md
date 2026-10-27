# Planner notes to the XP executor, round 2 (2026-10-27, 12:30 ET)

**Responds to:** issue #42, and commits `d5e83a9` (overnight evidence) and `69e1830` (CPU tools).

**Governing amendment:** XP prereg **A6**. Merge the planner branch before your next launch.

## Acknowledged

**Generation is closed and certified for:**

| Stage | Runs | Evidence |
|---|---|---|
| XP-A16 | 2,160 | `d73388d` |
| XP-A64 | 8,640 | `d5e83a9` |
| XP-D1 phase 1 | 400 | `d5e83a9` |
| XP-D2 | 400 | `d5e83a9` |

- Integrity passed and the embargo was respected: no GT was read and no outcome was exposed. Well done.
- **The four unverified job timings do not matter.** Timing claims in the paper use XP-T batch-1 measurements only (§1).
- **Budget:** cumulative use is about 39 GPU-h of the 400 cap.

## Decisions

1. **Freeze order.** v1 (FFHQ) and v2 (ImageNet FAST) are independent; neither waits for the other.
2. **Generate XP-D1 phase 2 now** (1,200 runs) and keep it sealed. Generation is never embargoed (A6.3).
3. **A4.4.** The repository records the historical FFHQ DAPS configuration as 400 annealing × 5 ODE steps, with the pixel Langevin task group (A6.1). The sources:
   - `configs/b23/fresh1_frozen.yaml`:
     - the source and patch identity;
     - the historical reference `/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401` (`manifest.tsv`, `analysis_theta0.7/fresh2_final_summary.json`);
     - the seed rule `seed1 = 22000 + frozen panel row_id`.
   - `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_AUTHORIZATION.md` l. 51 (DEV80 DAPS, 400 × 5).
   - `configs/b24/b24_3_dev80_overnight.json` (DAPS4 "frozen DAPS-1 semantics"; seed domain `B24_DEV80_DAPS_SOLVER_V1`).

   **Check** whether the R0.7 catalog's S1 DAPS candidates point to that B21.11 root. ("Fresh2", two runs per image, matches B22 DAPS n = 2.) Then run the **8-root replay under this configuration**. If it is exact, that is the A4.4 record and the XP-F8 configuration. If it is not, stop and report.
4. **XP-E′ is now eligible**, because the historical 400 × 5 differs from DAPS-4K's 400 × 10. It is CRITICAL-if-ready (A6.2):
   - DAPS-4K on FFHQ, 180 ids × 4 runs = 720 runs, about 13 GPU-h;
   - the pristine DAPS-4K worker and the frozen FFHQ y;
   - new seeds under the XP rule (dataset `ffhq_daps4k`);
   - no replay gate, because it is a separate population.

## GPU queue

**CRITICAL lane, now:**
1. XP-D1 phase 2: FAST, 1,200 runs, sealed.
2. XP-B16 (SITCOM FFHQ runs 4–15, 2,160 runs, sealed), in parallel with XP-D3 (SITCOM ImageNet, 400 runs). Use your replay-verified SITCOM worker and the historical RNG lineages:
   - B22: the post-model sequential stream continues at indices 4–15;
   - DEV80: independent native seeds, with the collision check.
3. The XP-C and XP-C2 GPU workers, as soon as each passes smoke. Both are light.
4. The DAPS 8-root replay (A6.1), then XP-F8 (sealed).
5. XP-E′.

**BACKGROUND lane** (idle slots only, units of about 20 min or less):
1. XP-K, SITCOM and FAST parts. The DAPS part comes after the replay.
2. XP-B32 (sealed).
3. XP-G.
4. XP-I and XP-I2: CRITICAL priority once the wrapper passes smoke.
5. XP-H.
6. XP-G2.
7. XP-J.
8. XP-D4.

At measured costs, the CRITICAL list is about 55 GPU-h, roughly 14 h of wall time on 4 GPUs:

| Item | Runs × time | GPU-h |
|---|---|---|
| XP-D1 phase 2 | 1,200 × 7.2 s | 2.4 |
| XP-B16 | 2,160 × 32 s | 19 |
| XP-D3 | 400 × 58 s | 6.5 |
| XP-F8 | 920 runs at 400 × 5 (estimated) | 9–15 |
| XP-E′ | 720 × 66 s | 13 |
| XP-C, XP-C2 and the replay | — | about 1.5 |

The cumulative total stays near 95 GPU-h, well under the cap.

## CPU work now

1. **Publish A8b** to the r0n return tree (theory v1 is waiting for it).
2. **Score XP-D1 phase 1** (runs 0–3 only).
   - Commit the per-image strings to **`docs/track_r/executors/xp/theory_inputs/`** (new; A6.6): the A8 schema, Good20, E0 rank order and ρ strings, plus the CLIP G1 decisions once the XP-C CLIP worker exists.
   - The v2 clock starts when they land.
   - Never place anything else there.
3. **Score XP-D2** (allowed before any freeze; A6.7). Report the published protocol (DAPS-4K, per-metric oracle over 4 runs) next to first run, E0, E1_CLIP (G1) and O_orbit, with PSNR, SSIM and LPIPS, against the published 25.78 ± 6.92 dB / 0.743 / 0.254. Commit the aggregates to `xp/returns/`.
4. **A7 ZIP and the A5.5 historical summary.**
   - Commit A5.5 to `xp/returns/` only, never to `theory_inputs/`: the theory executor may pre-register relations for it until Oct 28, 18:00 ET.
   - The ZIP goes to Cheng.

## Post-freeze analysis script

Evaluate on the A6.4 panels:

| Family | Primary panel | Secondary panel |
|---|---|---|
| FAST FFHQ | n = 64 | n = 16 |
| SITCOM FFHQ | n = 16 | n = 32, if XP-B32 is done |
| DAPS FFHQ | n = 8 | — |
| FAST ImageNet | n = 16 | — |

Ties go to the lowest run index. Compare every registered forecast variant (S2, race, tail) per A6.5: a predictive interval passes when the observation lies in [lo95, hi95]. Report the anatomy curves.

**Validate the script now** on historical K ≤ 4, reproducing A4.3 and A8, so that scoring is one command once v1 is registered.

## Reporting

- **Return when:**
  - the D1-P1 strings are in `theory_inputs/`;
  - XP-D2 is scored;
  - A8b is published;
  - the replay has a result;
  - XP-B16 and XP-D3 are done;
  - any stop condition fires.
- **Until v1 is registered**, post no XP-A16 or XP-A64 outcome anywhere, including issues.
