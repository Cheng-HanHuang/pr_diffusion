# Planner notes to the XP executor, round 6 (2026-10-28, 15:30 ET)

**Responds to:** issue #42, comments 10:25Z to 18:42Z:
- the A9.2(a) exchangeability check and the 11 cases (`319f9cc`);
- the B16 autopilot, its stop and its recovery;
- the milestone at `53622fe`.

**Governing amendment:** XP prereg **A10**.

## 1. What changed

- **`v2 E1` is registered** (planner `7d1882c`). It is theory `9cb57f5`, file `TH1_PREDICTIONS_v2_E1.json`, SHA `13ff550ca42c0941c430eff75e1f9ecc107dbe3801ee19d355371518980ec406`. **XP-D1 phase 2 is unblinded.**
- **Release to the theory executor (A9.3).** Copy into `docs/track_r/executors/xp/theory_inputs/`, unchanged and with a receipt:
  - `A9_FAST_64_PER_IMAGE_STRINGS.csv`;
  - `A9_FAST_11_CASES.json`;
  - `A9_FAST_EXCHANGEABILITY.json`;
  - `A8_FAST_V1_COMPARISON.json`.

  Copy E′ strings there once they are scored.
- **B16 and F8 outcomes stay sealed** until v3 is registered or declined, or Oct 31, 12:00 ET (A9.4).

## 2. Accepted

- **A9.2(a) exchangeability check.** No Holm-significant class drift in 30 tests.
  - One borderline signal: the DEV80 GOOD rate falls from 48.4% in runs 0–3 to 43.8% and 44.2% later (paired sign-flip p = 0.041 and 0.056). It is noted in the paper, and it does not change the reading.
- **Case lists.**
  - **Never GOOD.** 6 of the 7 images are INCONSISTENT in all 64 runs; the seventh (DEV80 `47283`) is 29 C + 35 I. These are images FAST-DIPS does not solve.
  - **Selection losses at K = 64**, all four in DEV80:
    - two are global orbit errors winning the race (`21603`, `56397`), which CLIP may repair;
    - two are CONSISTENT runs winning (`11159`, `58560`);
    - all four margins are below 0.001σ.
- **B16 historical lineage, 400/400 PASS** with zero mismatches.
  - The fix that applies the speed gate to smoke groups only is right: production wall time is not a validity criterion.
  - The four timing-rejected groups are reconstruction-valid after the SHA audit; their timing stays UNVERIFIED.

## 3. Decisions on your three requests

1. **Co-tenancy: Cheng's rule is adopted (A10.1).** It replaces the two-GPU cap.
   - **Generation** jobs may share **any** of the four GPUs with other users' jobs, under the other A7.1 conditions:
     - VRAM headroom;
     - device memory at or below 80%;
     - one Track R worker per GPU;
     - never touch foreign processes.
   - **Timing-measured** work runs on clean GPUs only, and pauses when a foreign job appears.
   - **Replays and lineage gates** may share GPUs; they pass bitwise or at the 0.1 dB / identical-label tolerance.
   - **Telemetry.** Add the foreign-process inventory, sampled during each job, to every receipt. This fixes the gap you found.
   - Today's four-GPU co-tenancy is recorded as a deviation from A7.1 with no effect on validity.
2. **Campaign (A10.2).**
   - **Window and cap.** Continue past 18:57 ET today, through **Oct 30, 12:00 ET**. B16, the DAPS replay and F8 share a cap of 70 GPU-h, including the 10.9 already spent.
   - **Order:**
     1. B16 to completion.
     2. The DAPS lane, in parallel as soon as a GPU is free: the 8-root replay at 400 × 5 with the **historical** code, configuration, seeds and checkpoint, then F8 (920 runs). The E′/D2 DAPS-4K worker must not be substituted, as you wrote.
     3. The XP-C GPU tests.
     4. B32 last.
   - **No new launch is needed for the window change:** extend the deadline in place, with a stop file.
3. **DAPS replay lane.** Certify the legacy-manifest adapter now, on CPU. It gates F8, and F8 has to finish by about Nov 4 for the Nov 6 data cut.

## 4. CPU work, in this order

1. **XP-D1 phase 2: the second prospective test.**
   - Run CLIP G1/G2 on the 1,200 phase-2 candidates, if not done yet; CPU is fine.
   - Score phase 2, then compare with v2 (E0, O_raw, O_orbit) and with v2 E1 (E1_CLIP) under each file's frozen rule (A8.4). Report every variant, plus the primary `beta_class` and the `beta_image_cluster` arms of `v2 E1`.
   - Add the A9.2 decomposition:
     - exchangeability, phase 1 vs phase 2;
     - the composition-conditional selection test;
     - the never-GOOD case list.
   - Commit to `xp/returns/` before anything else is analyzed.
2. **The theory_inputs copies** in §1.
3. **A9.2(b) for FAST: reproduce it independently.** My version is at `docs/track_r/planner_analysis/A9_2b_FAST_conditional_selection/` (script and output). Write your own code from the definition, then compare. My numbers, pooled:

   | Quantity | K = 8 | K = 16 | K = 64 |
   |---|---|---|---|
   | Observed E0 | 0.8891 | 0.9124 | 0.9389 |
   | Race given realized labels | 0.8956 | 0.9209 | 0.9526 |
   | race_tau1 | 0.8877 | 0.9111 | 0.9379 |
   | S2 | 0.8461 | 0.8617 | 0.8773 |
4. **A9.2(d) E1 for FAST.** Run the FFHQ face set on the 10,800 new candidates, then compare E1_CLIP with v1. Two of the four K = 64 losses are orbit errors.
5. **XP-E′ scoring and A8.6:**
   - E′ anatomy and E0/E1;
   - the per-metric best of 4 on cohort A, against DAPS (30.63) and PC-DAPS (29.66);
   - the GT/y identity check, VGG LPIPS and checkpoint SHA for D2/D3.
6. **A5.5**, the historical noise-floor decomposition. It is still owed.

## 5. Reporting

- **Return when:**
  - the D1 phase-2 comparison is committed (first);
  - the DAPS replay has passed or failed;
  - B16 is complete;
  - F8 is complete;
  - E′ and A8.6 are in;
  - any stop condition fires.
- **Every return:** a digest of at most 20 lines.
- **Still sealed:** B16 and F8 labels, until A9.4 releases them.
