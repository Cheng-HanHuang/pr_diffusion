# Planner notes to the XP executor, round 18 (2026-11-03, evening)

**Responds to:** Issue #46 reports (XP-C3 complete, recovery, XP-I2), XP `91fe7a3`, `a6c6a64`, `591a28f`, `424399f`, `1eae9bf`.

**Governing amendment:** prereg **A22**.

## 1. XP-C3: accepted, reproduced, in the paper

**The recovery was handled correctly.**
- The read-only audit came first.
- The publications ran on CPU only, with no GPU rescore.
- The stop receipt is preserved.

A22 records the stop as an engineering incident, and all three panels count as complete. The A21.10 replicate was not run; it is not claimed anywhere.

**My checks** (`docs/track_r/planner_analysis/A22_xpc3_xpi2/`):
- All 13,600 rows read; labels equal the frozen strings; the 1e-7 threshold rule holds in every row.
- Wherever the prior and CLIP chose the same orientation, the ground-truth outcomes agree.
- `*_chosen_good` is evaluated against ground truth at the chosen orientation. So a G run flipped to 111 can still count as GOOD on a nearly symmetric image (e.g. `00049094`). That is correct, but say so in the receipts next time.
- The D4 CLIP file still uses the old labels `O`/`P`; I mapped them to `O_g`/`O_p`.

**Outcome** (exploratory, E1-prior against E1-CLIP; exact estimator):

| Panel | Prior − CLIP |
|---|---|
| XP-N | **+3.8** at K = 4 [+0.9, +7.0]; **+5.3** at K = 16 [+1.1, +9.8]; +5.0 at K = 64 [−1.0, +11.0]. Orientation loss at K = 64: 7.0 → 2.0 |
| XP-D4 | −0.4 / +0.1 / +1.0 at K = 4 / 16 / 64 (intervals contain 0) |
| G5 | +1.4 / +1.0 at K = 4 / 8 (intervals contain 0) |

**Where it appears:** Sec. 6 (one sentence) and S4 (new table). Protocol v1 is unchanged; the prior test is reported, not adopted.

**Your candidate-level rates are right.** For example, XP-N's 2,311 prior-oriented GOOD candidates are 2,309 G or O_g candidates oriented correctly, plus 2 O_p candidates that the global flip happens to make GOOD.

**Controller fix, for future controllers only.** Signal only workers that are still alive. Treat a worker whose own log shows a normal exit as rc 0, whatever signal arrives afterwards.

## 2. XP-I2: accepted, reproduced, in the paper

**Reproduction:** your policy table matches mine to 3.6e-15, and the four DAPS-4K comparators match image by image.

**Paired bootstrap of E1 − native** (20,000 replicates, mine):
- GOOD **+22.25 points [+14.8, +30.0]**;
- PSNR +4.11 dB [+2.61, +5.64];
- SSIM +0.110;
- LPIPS −0.091.

**Where it appears:**
- the abstract and intro: "79.0% → 84.4% on FFHQ and 31% → 53% on ImageNet";
- Sec. 6: one sentence;
- S4: a new ImageNet table with the DAPS-4K rows, compute-labeled, no wall-clock times.

## 3. Requests

1. **Tab. 3 E1 columns (A7 ZIP), due Nov 6, 12:00 ET.** DAPS-4K and SITCOM on ImageNet:
   - E1 with G1 at K = 4 (and 8 for DAPS-4K);
   - PSNR, SSIM and VGG-LPIPS at the chosen orientation.
2. **The frozen N2 face prompts, verbatim.** Copy the strings and their source commit into `theory_inputs/`. The supplement must quote them (S2).
3. **Optional, information only:** the A21.10 cross-GPU replicate (32 candidates per panel with the smallest SHA-256 of the uid), on an otherwise idle GPU, ≤ 0.2 GPU-h.
4. **Then, unchanged:** XP-K, the rest of G7 (XP-C2 equivariance profile; UnivFD), CP4 (Nov 15).

**Unchanged rules:** no protocol v2, no new selector, no matched-compute claim. Stop and ask only under A13.4.
