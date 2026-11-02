# A19 XP-C3: A13.4 prospective prior-definition clarification gate

**Status:** STOP_XPC3_A13_4_FROZEN_PRIOR_DEFINITION_AMBIGUOUS. Engineering can continue on other independent stages. Do not launch XP-C3, compute any prior margins, inspect any held candidates for this test, or publish a code-freeze receipt until this gate is resolved in writing.

**Planner basis:** \`docs/track_r/executors/xp/15_PLANNER_NOTES_20261102.md\` and frozen \`docs/planning/track_r/XP_PREREG_v1_1.md\` A19/A16/§1.

## Exact contradiction in governing text

- The original XP-C prior specification defines **two distinct tests**:
  1. **2-forward:** single timestep **t=300, J=1**, shared/paired \`R epsilon\`, compare global rotations \`000\` vs \`111\`.
  2. **S_mid:** **t in {300, 400, 500, 600}, J=4**, compare global2 and all8; for global2 alone this means **4 x 4 x 2 = 32 denoiser-evaluation samples per candidate**, not 2.
- Later A16.4 says "2-forward S_mid, global2", and A19/round-15 repeat that composite wording, without a tie-break rule. \`prdiffusion/track_r/prior_score.py\` independently defines \`S_mid\` as the average of the four middle timesteps, but cannot reduce it to the single-timestep 2-forward test without changing the estimand.
- The intended rule is therefore not uniquely determined from the frozen text. It is **not** valid to select whichever score works best on any previously seen dataset, even though XP-C3 is exploratory.

## Single requested planner clarification

**Which one frozen XP-C policy should provide the sole \`prior_choice\` and \`prior_margin\` columns for all 13,600 candidates (XP-D4 6,400, G5 800, XP-N 6,400)?**

- **A:** Original **2-forward**, t=300, J=1, paired R-noise, two orientations. One score sample per orientation, exactly two denoiser evaluations per candidate.
- **B:** Original **S_mid global2**, t={300,400,500,600}, J=4, paired noise, two orientations. Mean of 16 samples per orientation, 32 denoiser evaluations per candidate.
- **C:** A specific *previously frozen* third rule — cite the exact governing definition (including t-set, J, seed, margin sign and ties) before any scoring.

This is a **definition-only** A13.4 gate. The planner must not access the observed prior margins/GOOD outcomes to decide. The response should be registered/committed before code SHA and certainly before scoring, without moving the 3 GPU-h ceiling.

## Unchanged obligations once resolved

- The A19 cohort expansion is binding: 6,400 XP-D4 + 800 G5 + 6,400 XP-N = **13,600**.
- Original GT-free selection and original offline labels are separate; \`prior_chosen_good\` is computed *after* frozen choice.
- Required exact row columns: \`image_id,run,label,prior_choice,prior_margin,prior_chosen_good\`; source-identifying cohort must be in filename, receipt or metadata.
- The native denoiser is the ImageNet checkpoint, with its SHA, the validated DDPM-1000/native time mapping, and strict checkpoint weight trace; freeze an explicit seed rule, paired rotation rule and 000-tie rule.
- The code hash/commit and source manifest hashes must be pushed before any XP-C3 candidate scoring. First pass source-only CPU integrity and a bounded GPU smoke, then main work on admissible dirty GPUs (≤80% VRAM, one Track R worker/GPU, no timing claim), with a hard 3 GPU-h cap **inside G7 and inside the project 400 GPU-h cap**.
- No protocol v2, matched-compute equivalence or new selector; any missed Nov 5 noon ET main-paper cut becomes supplement.

## Independent unaffected progress

Nov4 FAST and D4 CPU workers have detached from an independent A25 worktree, and their PAC logs report 7/7 panels, 8/8 panels and FAST face-CLIP 180/180 images (11,520 decisions). These are *completed CPU process logs*, not an independent file-SHA publication audit.

XP-I2 detached from A24 at execution head \`81ae297cc444c5573909ecdc1f72b3416be5517b\`; production/replay status is **not known** from those Nov4 logs. Do not change its checked-out HEAD or launch a duplicate supervisor.

**Do not interpret this XP-C3 A13.4 gate as a stop to independent Table 3/XP-I2 readouts or manuscript packaging.**
