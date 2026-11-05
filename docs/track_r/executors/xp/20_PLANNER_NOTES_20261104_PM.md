# Planner notes to the XP executor, round 20 (2026-11-04, evening)

**Responds to:** Issue #46 reports of Nov 4; XP `da130b5`, `967e73c`, `6d2afbf`, `9d379d2`; closeout `4d1bd60`.

**Governing amendment:** prereg **A24**.

## 1. G7 is complete and accepted

**My check** (`docs/track_r/planner_analysis/A24_g7_gt/`):
- All four files hold the same 280 images, with the same source hashes as the CLIP file.
- The S_mid threshold rule holds in every row.
- The aggregates equal the per-image counts.
- S_mid all8 chooses exactly as S_mid global2 on all 280 images.

**Upright kept, FFHQ / ImageNet:**

| Test | FFHQ | ImageNet |
|---|---|---|
| S_mid global2 | 180 | 95 |
| S_mid all8 | 180 | 95 |
| 2-forward | 180 | 93 |
| UnivFD | 172 | 89 (18 global and 1 channel-wise error, both datasets together) |
| CLIP G1 / G2 (A23) | 180 | 92 |

**New observation:** the ImageNet errors overlap little. No image is turned by all of G1, S_mid global2, 2-forward and UnivFD, and 20 are turned by at least one. This is reported in S2 as a fact only. It is not a new selector (A19.3).

**Your summary message said UnivFD "failed".** Your Issue #46 closeout says it was recovered and published (`9d379d2`). I used the closeout. The recovery and the publication-only retry are accepted.

**Ledger:** G7 5.97 of 10 GPU-h; project 247.80 of 400.

**Placement:** supplement S2 has a new table with every orientation test on the ground-truth images.

## 2. What remains

1. **XP-K,** under the three A23.5b gates:
   - a committed diff showing exactly one changed parameter;
   - an identity replay of two B22 roots per family;
   - DAPS only after the F8 replay gate.

   **Stop and ask if a gate fails.**
2. **CP4 by Nov 15, 12:00 ET:**
   - XP-K against mech v1/v1.1;
   - the final project ledger;
   - the list of everything not run.

   G7 is done and need not be repeated in CP4; cite `4d1bd60`.

**GPU policy (Cheng, Nov 4):** co-tenant-safe work may run; clean-GPU work waits. For the XP-K identity replays, a quiet GPU is preferable, but the tolerance must not be weakened.

**Unchanged rules:** no protocol v2, no new selector, no matched-compute claim. The main-paper cut is closed.
