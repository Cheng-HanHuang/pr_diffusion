# XP-F8 historical DAPS native lineage: 8/8 pixel-exact PASS

**Date:** 2026-10-29 06:56:28Z; **gate:** A4.4/A5.2, prereg A10.

## Result (no new-production outcomes opened)

- Historical native DAPS `e7a77d094167084faed19b599b96673b7bb11447`, with original default-off B20/B21 source patches, FFHQ checkpoint SHA `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`; 400 annealing ×5 ODE ×100 pixel Langevin steps.
- **B22:** first two sorted FFHQ IDs × frozen runs 0–1, Fresh2 B21 seeds `22020, 23020, 22035, 23035`: 4/4 output PNG pixel-identical, 0 dB raw-PSNR difference, labels identical.
- **DEV80:** first two sorted FFHQ IDs × frozen runs 0–1, original **`B24_DEV80_DAPS_SOLVER_V1`** seeds `1269390804, 1756260723, 3183756402, 377042404`: 4/4 PNG pixel-identical, 0 dB raw-PSNR difference, labels identical.
- **Overall 8/8 EXACT** at new replay output `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_f8_legacy_replay_20261029T065122Z`. Native source and checkpoint checks, per-root receipt SHA, replay GT-free generation and offline comparison were all completed.
- Frozen corrected replay-input manifest `0929a94537c30faa893346a000c6cca5b51d5f45b48fbc2dfb0f6e64fa2537b9`; corrected CPU preflight receipt `81a0fd43d75521988b4a02d1c148ed99a48c6976cccb1c64899f5c215f97b4f1`.
- Offline wave-1 receipt SHA `45e69880b04adb3a8762a940b4cb3630a4f055ec7784368ef9d891476320b1da`; wave-2 `ab34db50115ee7daef5b55ac81b321b47a318b67028ab21cda8a92b9e52d0385`.
- Historical DAPS 400×5 config is therefore corroborated (not DAPS-4K, which is 400×10); this releases the **generation** of XP-F8 920 new roots, subject to the shared A10 budget and hard deadline.

## Prior failures preserved and accounted

1. Initial adapter CPU-Hydra and dummy-dataset errors were implementation problems, not reconstruction mismatches.
2. The first eight-root 400×5 replay used the wrong **screening** seed domain `B24_SOLVER_SEED_V1` for DEV80, resulting in 5/8 tolerance PASS and large deviations. The frozen original DEV80 launcher instead used `B24_DEV80_DAPS_SOLVER_V1`; the corrected original task/group seed match produced **8/8 pixel identity**.
3. Replay worker GPU-hour accounting, including failures: `0.12478472114294871 + 0.11290400060113623 + 0.10422916961200018 + 0.10994419192403762 + ≈0.1357` = ≈0.5876 GPU-h. B16 spent `38.021455` GPU-h. Total ≈`38.609` / `70`, remaining ≈`31.391` GPU-h **before F8 production**.
4. F8 prereg projects **40–50 GPU-hours**, which exceeds the A10 remaining balance. The production controller must enforce the common 70-hour ceiling and `2026-10-30 12:00 EDT` deadline. Partial completion is not a full scientific panel; any partial records must be retained and reported without post hoc score-based decisions.

**A9.4 embargo remains active:** do not score or disclose any B16/F8 new-root PSNR, class labels, residuals or selection outcomes until v3 is registered/declined, or 2026-10-31 noon ET. Mechanical generation receipts may be reported.

Planner notification: https://github.com/Cheng-HanHuang/pr_diffusion/issues/42#issuecomment-6490203618.
