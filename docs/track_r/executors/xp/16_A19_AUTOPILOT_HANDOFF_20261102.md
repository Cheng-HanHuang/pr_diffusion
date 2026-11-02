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

## A24 implementation checkpoint (same branch, CPU/bootstrap only)

- `scripts/track_r/xp_a24_round15_preflight.sh` provides the first safe CPU kickoff with A24 unit tests, readiness receipt and original D1 ImageNet source freeze for XP-I2. It **does not start any GPU process**.
- `scripts/track_r/xp_a24_xpi2_prepare_cpu.py` freezes the D1 `XP_IMAGENET100_Y_MANIFEST.json` 100-image, 4-particle-bag manifest, native PC-DAPS source SHAs and ImageNet checkpoint SHA; GT-free inputs.
- `scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py` is an ImageNet adapter with strict weights, one bag/four particles, and **mandatory two-bag exact replay receipt** before ordinary generation. It is not yet attached to a replay-checking / 40 GPU-h stage / 60 GPU-h shared-tail supervisor, so **no production launch is authorized** by this checkpoint.
- `scripts/track_r/xp_a24_a19_readiness_cpu.py` reports status and explicitly blocks XP-C3 until a candidate worker and pre-scoring code-freeze artifact both exist. It does not inspect candidate images or GT metrics.
- XP-C3 exact 2-forward/S_mid global2 worker, Nov4 image panel/face-CLIP production, A7 ZIP Table3 E1 evaluator, and XP-K/G7/CP4 tail controllers still require implementations or verified existing artifacts. Do not silently equate A12 G7-C2 or generic prior_score.py with the approved XP-C3 test.
- This is **partial work-in-progress code**, not a claim that A19 autopilot is currently executing. Complete, CPU-test, replay/smoke, checkpoint and push each stage before its GPU work.

## Operational rules

- Do not modify committed results in prior run directories; separate stage worktrees / outputs and immutable Git HEADs.
- No GPU launch before the corresponding native worker passes source, model, signed-y, generator and replay checks.
- No extra GT-based method selection. GT only for offline quality scoring and labels, never for inverse reconstruction or prior orientation choice.
- Each real stage must use its own checkpointed controller, **not** a single shared mutable cost counter. Coordinate GPU lock and cumulative project ledger across parallel stages. Do not double-charge prior history or silently reset costs.
- Time cutoff does not authorize changing the method. If a deadline is missed, record main-text downgrade and preserve supplemental value.


## A24 XP-I2 supervisor implementation checkpoint (post-green 08:39Z PAC preflight)

- PAC confirmed the initial seven tests PASS, 100 D1 ImageNet signed-y frozen under
  \`xp_a24_xpi2_prepare_20261102T083920Z\` at producer HEAD
  \`a8c379fb6ca49448a71480cab65b49c6936b9473\`.
  Original Y manifest SHA256 \`26619959e3a17aeb2c0ad413f87b1764ecd8bc7aabfc0415c862cbaf27873b78\`.
  Original prep manifest SHA256 \`fd89323b0e7bcf3de2a0acfcf8c99857c500509bcae3d94c81268a8efb03f7ed\`.
- Follow-on code adds \`scripts/track_r/xp_a24_xpi2_autopilot.py\`,
  \`scripts/track_r/xp_a24_xpi2_launch.sh\`, and
  \`tests/test_track_r_xp_a24_i2_autopilot.py\`. It is designed to carry
  two replay bags through exact repeatability and only then generate
  100 ImageNet four-particle native PC-DAPS bags. Supervisor locks the
  Track R shared campaign ledger, admits co-tenant GPUs below 80%,
  excludes other Track R workers, accounts every child process attempt,
  cleans up owned children on failures and forbids duplicate production.
- **This new supervisor has NOT been PAC-tested or GPU-launched.**
  No native replay, 100-bag output, or XP-I2 main-text score is claimed.
  Its source unit test and CPU preflight must pass *on PAC* before launch.
- Project baseline is read from the unique accepted terminal A22 XP-N
  \`XP_N_CONTINUATION_FINAL_STATUS.json\`. A19's approximately 230h
  must not replace receipt-derived charged values. XP-I2 uses 40 GPU-h
  stage and 60 GPU-h tail ceilings, the global 400 GPU-h cap, and a
  0.2-hour worker admission reservation; all retries are charged.
- After successful XP-I2 generation, **offline scoring, G1 orientation
  decisions, descriptive (a)/(b) analyses and the theory_inputs export
  are not yet implemented** by this A24 checkpoint.
- XP-C3 exact native test and mandatory prospective SHA freeze,
  Nov4 panels and face-CLIP 11,520, A7 ZIP Table 3, and XP-K/G7/CP4
  remain staged engineering work; do not report them as running.
  The current launch is an XP-I2 component launch, not a complete
  five-stage A19 campaign launcher.
- Stop/ask at A13.4 only for frozen-rule ambiguity, protected data,
  irreconcilable SHA/receipt, cap conflict, or operator-required
  resources. Other blocked stages may be parked with precise receipts,
  but no frozen science may be changed to make a stage run.


## Nov4 CPU implementation checkpoint (subsequent to XP-I2 PAC detach)

- Confirmed by Cheng's PAC output at 2026-11-02 08:56Z: A24 XP-I2
  previous HEAD \`81ae297\`, all 9 CPU QA tests PASS, project baseline
  **229.02681142068252 GPU-h**, eligible GPU indices 0–3, supervisor
  detached PID 3899225, run
  \`xp_a24_xpi2_prod_20261102T085625Z\`. This is **LAUNCHED, NOT
  NATIVE REPLAY VERIFIED OR FINISHED**. Do not start another XP-I2.
- Follow-on A19 CPU-only scripts (GitHub, not PAC-tested):
  \`xp_a24_nov4_fast_cpu.py\` renders the seven frozen A9 FAST FFHQ
  never-GOOD panels, each anonymized under 2 MB with reference, lowest
  signed-rho and orbit oracle. It also evaluates all existing 180x64 FAST
  candidates against the *frozen* face CLIP prompt set, writes resuming
  64-row image chunks, 11,520 G1-style decision rows, and exact E0/E1
  subset U-statistics with fixed-first-K summaries. Candidate choice is
  GT-free; GOOD25 is evaluated offline only.
- \`xp_a24_nov4_d4_panels_cpu.py\` collects the frozen 22 never-orbit-GOOD
  XP-D4 ImageNet IDs, chooses eight via ascending
  SHA256(image_id UTF-8), and renders the matching anonymized panels
  using original D1/G1a/G1b candidate receipts. Offline oracle labels
  are diagnostic, never fed into any selector. See its local figure
  selection receipt for the exact ordering.
- \`xp_a24_nov4_launch.sh\` is a one-shot detached CPU launcher after
  two static/behavioral QA sets and full frozen-input preflight; it
  refuses any existing incomplete Nov4 run rather than overwriting.
  No GPU is requested or charged. The XP-I2 controller is unaffected.
- These scripts must PASS on PAC before claiming completion. They
  should **not** be used as evidence of scientific results merely
  because GitHub source code is present. Neither XP-C3's required
  prospective prior-code freeze nor XP-I2 native replay have been
  validated by this checkpoint.
