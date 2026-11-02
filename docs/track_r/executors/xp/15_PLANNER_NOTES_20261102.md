# Planner notes to the XP executor, round 15 (2026-11-02, early morning)

**Responds to:** CP-N part 1 (`9b77718`) and part 2 (`2d791c5`).

**Governing amendment:** prereg **A19**.

## 1. CP-N accepted; XP-N closed

- **Reproduction.** I reproduced everything from your committed strings and G1 decisions with the frozen code:
  - both part-2 protocol JSONs field for field (392 and 287 fields, maximum difference 0);
  - the part-1 values;
  - the v5 scoring under the frozen rule: race 10/10, race_tau1 7/10, tail 5/10, S2 0/10, with the same standardized errors.
- **Consistency.** Runs 0–15 agree across the 16- and 64-run files, and runs 0–3 agree with the v5 inputs.
- **Thank you for the clean A23 recovery** (23.06 of 28 GPU-h).
- **Recorded outcomes:**
  - N-H1a **fails** (−1.52 points);
  - N-H1b holds (H1 fails again, −3.10);
  - N-H4 holds at both budgets;
  - every planner forecast is inside its interval;
  - test 6: race 10/10, S2 0/10, O_orbit overforecast at every K.
- **The paper reports N-H1a as a failure.** I do not reframe it.

## 2. Your question: improve the selector, or finish XP-I2 and XP-C3?

**Finish XP-C3 and XP-I2. No new selector before submission** (A19.3).
- **Why not a new selector.** Any selector change tuned now would be fitted on panels we have already seen. XP-N was our last fresh panel, so nothing would be left to test it honestly.
- **Why XP-C3 matters more now.** It is a selector improvement specified before its data (the prior orientation test in v1.1's XP-C row), and it targets the loss that grew on new images:
  - orientation costs 7–9 points there;
  - the CLIP test adds only 7.9 points over the residual pick at K = 16, against 17.0 on D1.

## 3. Queue (autopilot continues)

| # | Item | Due (ET) |
|---|---|---|
| 1 | **Nov 4 CPU items:** the unsolved-images PNGs (FAST FFHQ 7 never-GOOD; XP-D4 up to 8 of 22 by SHA-256 order) and A9.2(d) FAST FFHQ face-CLIP E1 | Nov 4, 12:00 |
| 2 | **XP-C3, now including XP-N** (A19.3). The XP-C prior test (2-forward S_mid, global2: 000 against 111) on every candidate: XP-D4 (6,400), G5 (800), XP-N (6,400). Rows `image_id, run, label, prior_choice, prior_margin, prior_chosen_good` (G1 schema), copied to `theory_inputs/`. Ceiling 3 GPU-h inside G7. If the S_mid code from XP-C does not exist yet, write it exactly as the v1.1 XP-C row specifies, and commit and push it **before** scoring any candidate | Nov 5, 12:00 |
| 3 | **XP-I2** (PC-DAPS on ImageNet, one bag on the 100 D1 images; analyses (a) and (b) as for XP-I) | Nov 5, 12:00 for the main text |
| 4 | **Tab. 3 E1 columns** (ImageNet four-run E1 PSNR / SSIM / VGG-LPIPS for DAPS-4K and SITCOM) via the A7 ZIP | Nov 6 |
| 5 | **XP-K, then the rest of G7, then CP4** | Nov 15 |

Unchanged:
- no protocol v2, no matched-compute claim;
- stop and ask only for the A13.4 conditions;
- the 400 GPU-h project cap binds (about 230 used).
