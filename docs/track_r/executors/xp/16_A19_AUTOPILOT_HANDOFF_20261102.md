# A19 Round-15 executor handoff — safeguarded autopilot, 2026-11-02 ET

## Authority / scope

Governing notes: `docs/track_r/executors/xp/15_PLANNER_NOTES_20261102.md` at planner prereg A19, planner commit 07530e0, frozen `docs/planning/track_r/XP_PREREG_v1_1.md` A19. A23 XP-N is CLOSED: native 6,400/6,400 in 23.06176 of 28 GPU-hours; CP-N Part 1 `9b77718578b8bc1e4a8ccffc94c4b305868d846c`, Part 2 `2d791c5d97fd7a09c216f27ab9b087ad26777882`, registered v5 `55ba86b71cafbd8eb066c777ac141823d9cd9a1d`. Planner has reproduced Part 2 protocols field-for-field. N-H1a FAILS; N-H1b holds; N-H4 holds; race 10/10 vs S2 0/10. Do NOT propose a new selector/protocol v2 or fabricate matched compute.

## Ordered queue and deadline (ET)

1. By Nov 4 noon: CPU unsolved-image panels: 7 FFHQ never-GOOD FAST images and up to 8 XP-D4 never-orbit-GOOD images (choose by SHA256 ordering); A9.2(d) FFHQ face-CLIP E1; exact sources, GT, lowest-residual, orbit-oracle thumbnails, anonymization and under-2-MB bound per planner.
2. By Nov 5 noon: XP-C3, A19.3 extension. All candidate orientations: XP-D4 6,400 + G5 800 + XP-N 6,400 = **13,600** rows. Predeclared XP-C prior 2-forward / S_mid global2 identity vs 180-degree rotation, native dataset checkpoint, paired rotated noise; output `image_id,run,label,prior_choice,prior_margin,prior_chosen_good` for theory. Strict ceiling **3 total GPU-hours inside G7**, project ≤400 GPU-hours. Stage is exploratory, NOT a newly tuned selector.
3. By Nov 5 noon for main text: XP-I2 PC-DAPS ImageNet on the original D1 100 images (not XP-N disjoint images). One native four-particle bag per image, frozen D1 signed measurements, native ImageNet checkpoint, first-particle, mean-of-four metrics, E0/E1/GT oracles, DAPS-K4 comparator. Original FFHQ G3 native worker is NOT yet an ImageNet worker.
4. By Nov 6: supplemental Table 3 E1 per-metric columns for original DAPS-4K and SITCOM four-run ImageNet from the historical A7 ZIP. Do not invent an A7 ZIP path or infer that the ZIP is available from a Git filename.
5. By Nov 15: XP-K, remainder of G7, then CP4 as separately authorized by freeze and budget.

Independent CPU tasks may run alongside native GPU workloads if they don't overload the shared machine or access unregistered held data. Stop/ask only at A13.4; otherwise record missing code, unavailable artifacts or budget constraints as explicit stage receipts and continue with runnable independent tasks. An autopilot MAY NOT treat missing implementations as completed.

## GPU classification (A10.1 controls)

| Stage | Clean requirement | Dirty GPUs permitted? | Policy |
|---|---|---|---|
| Nov 4 image panels / frozen CPU CLIP | no GPU | n/a | CPU only |
| XP-C3 2-forward / S_mid prior | only controlled time benchmark | YES for quality evaluations | GPU inference with checkpoint SHA, deterministic seed, numerical identity / output verification; memory ≤80%; stage 3h including unsuccessful attempts; G7/shared ledger |
| XP-I2 native PC-DAPS | clean only for controlled timing | YES for native 100-bag generation, if replay accepted | one worker/GPU, 80% VRAM safety, source/model/measurement hashes, no GT in sampler, charges all attempts |
| XP-I2 piq / CLIP scoring | clean only for controlled timing | YES if GPU needed for evaluation; otherwise CPU | GT offline labels only; no solver change |
| Tab. 3 E1 / piq VGG-LPIPS | no clean requirement for metric values | YES when GPU needed | frozen GT/data with exact piq version and A7 ZIP verification |
| XP-K / G7 remaining / CP4 | clean for timing or explicit native determinism gate that requires it | YES for ordinary sampling/evaluation | obey original frozen phase-specific caps, A10.1, replay, and global 400h |

A dirty GPU is NOT automatically admissible: admission must check observed VRAM, 1.5x native peak + 2 GiB safety reservation, total ≤80%, valid telemetry, one Track R worker per GPU; record foreign PID/VRAM pressure. Never kill foreign PIDs. Clean GPU means no foreign compute and controlled consistent timing; clean-only is not a blanket constraint for candidate-quality experiments.

## Implementation status / required freeze

**Existing reusable source**: `prdiffusion/track_r/prior_score.py` supplies per-orbit DDPM paired-noise scores and a 4-timepoint S_mid; `scripts/track_r/xp_a12_g7_c2_gpu.py` shows native FFHQ/ImageNet denoiser load and timestep mapping. These are NOT an authorized completed XP-C3 13,600-candidate scorer. Must pin exact 2-forward/S_mid definitions and source hashes and push scoring code *before reading a single XP-C3 candidate outcome*. The G7-C2 controller is not an XP-C3 controller.

`scripts/track_r/xp_a12_g3_pcdaps_gpu.py` is pinned to **FFHQ** checkpoint/configuration and 180 bags. Do not run it on D1 ImageNet or present it as XP-I2. A distinct native 100-bag ImageNet adapter and GT-free measurement/seed replay gate are needed before GPU launch.

`scripts/track_r/xp_a13_f8_faceclip_cpu.py` is existing FFHQ face-CLIP infrastructure, not automatically the A9.2(d) 11,520-decision return. A7 ZIP needs explicit file detection and hash before Tab. 3 scoring.

The A19 readiness script only audits local files, code support and resource classes; it is safe to run CPU-only while existing GPU jobs are active. No user intervention should be needed until a true A13.4 stop condition.

## Operational rules

- Do not modify committed results in prior run directories; separate stage worktrees / outputs and immutable Git HEADs.
- No GPU launch before the corresponding native worker passes source, model, signed-y, generator and replay checks.
- No extra GT-based method selection. GT only for offline quality scoring and labels, never for inverse reconstruction or prior orientation choice.
- Each real stage must use its own checkpointed controller, **not** a single shared mutable cost counter. Coordinate GPU lock and cumulative project ledger across parallel stages. Do not double-charge prior history or silently reset costs.
- Time cutoff does not authorize changing the method. If a deadline is missed, record main-text downgrade and preserve supplemental value.
