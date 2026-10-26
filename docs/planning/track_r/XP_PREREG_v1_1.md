# XP pre-registration v1.1: evaluation-only extensions, aggressive tiered scope

**Status:** FROZEN v1.1 (planner, 2026-10-26).
- **GO given.** Cheng, 2026-10-26 00:57: *"GO aggressively, the lab gpus are not actively been used and so even if some runs failed it's no hurts. Try to get as much as possible out."*
- **Supersedes `XP_PREREG_v1.md`.** No XP data existed when v1.1 was frozen: there are no executor commits after `33a34bd`, and no XP or theory branch exists. v1 is kept for the record.
- **This file is self-contained.** The executor does not need v1.

**Purpose.** Strengthen the audit for the CVPR 2027 attempt (`2026-10-25_TOP_VENUE_PLAN.md`) with:
- a second dataset (ImageNet), including a replication of the field's published DAPS protocol;
- K-scaling that **tests theory predictions frozen in advance**, in both regimes the theory distinguishes (FAST: few consistent-wrong runs; SITCOM: many), plus DAPS;
- same-GPU timing for one compute axis;
- orientation tests across domains;
- supplement-grade robustness (noise level, DPS backbone, concurrent PC-DAPS on the same images).

**Constraints.**
- **No new method.** Every result is reported whatever its outcome.
- **Cap: 400 GPU-hours**, counted as wall-hours per occupied GPU, all stages and scoring included. PAC has 4 × RTX PRO 6000 Blackwell (R0.5 record); XP-0 re-inventories them.
- **The real limits are the data cuts:** Tier A by **Nov 2**, Tier B by **Nov 6** (main paper), Tier C by **Nov 20** (supplement, due Nov 23).

## 0. Changes from v1

| # | Change |
|---|---|
| 1 | GO recorded. Cap 45 → 400 GPU-h. Work is organized in tiers A (required), B (main paper) and C (supplement), run as a work-conserving queue (§3) |
| 2 | FAST FFHQ: n = 16 (A), then **n = 64** (B). FAST ImageNet: n = 16 (A, was 8), then n = 64 (C) |
| 3 | SITCOM FFHQ K-scaling on **all 180 ids** (was 60), n = 16 (B), then n = 32 (C) |
| 4 | **New:** DAPS FFHQ K-scaling to n = 8 on all 180 ids (B) |
| 5 | DAPS ImageNet on **all 100 images** (was 25), using the **published DAPS-1K phase-retrieval command** (B) |
| 6 | **New (C):** XP-G noise robustness (FAST, σ ∈ {0.02, 0.10}); XP-H DPS backbone; XP-I PC-DAPS on the same 180 ids |
| 7 | **Two-stage prediction freeze:** v1 for FFHQ (by Oct 31), v2 for ImageNet FAST (after its runs 0–3). Plus an **embargo** on K > 4 outcomes until each freeze (§5) |
| 8 | **Replay and exchangeability gates** before adding runs to historical families (§4) |
| 9 | **Exact U-statistics** by order-statistic identities instead of subset enumeration (§6) |
| 10 | Per-image label and decision strings are committed (Cheng approved the A8 strings, 2026-10-26) |
| 11 | **New (A):** XP-C2, the equivariance-defect profile of each checkpoint on GT images. It tests the theory's mechanism for orbit errors (TH1.1 Cor. 4) |

## 1. Fixed objects

| Object | Frozen choice |
|---|---|
| Operator | R0 pinned operator A(x) = \|F_c(pad₆₄((x+1)/2))\|, 384² canvas, channel-wise, σ = 0.05, signed y. This is the DPS/DAPS standard (`oversample: 2.0`, `sigma: 0.05`) |
| FFHQ ids | The 180 ids of B22 (100) and DEV80 (80), exactly as in the R0 manifests. **No other FFHQ id** (XP-E is the gated exception) |
| FFHQ measurements | **Reuse the R0 measurement file per id.** Before any re-run of a historical family, verify per id that this y has the same SHA-256 as the y used by that family's historical runs. **On mismatch, stop and report.** Never substitute silently |
| XP-G measurements | The same 180 ids, **paired noise**: n_R0 = (y_R0 − A(x★))/0.05, then y_σ = A(x★) + σ·n_R0 for σ ∈ {0.02, 0.10}. Only σ changes. Save each y_σ and its SHA |
| ImageNet images | DAPS's 100-image `test-imagenet` set (also used by FAST-DIPS and PC-DAPS), in the DAPS/FAST-DIPS loader order `sorted(Path(root).rglob(ext))`, indices 0–99. Record each file's SHA-256 in the XP-0 manifest |
| ImageNet measurements | New. y = A(x) + σ·n with n ~ N(0, I) and σ = 0.05. Seed = first 8 bytes of SHA-256(`XPMEAS\|imagenet\|<basename>`) as an unsigned integer mod 2⁶³. Save y and its SHA. **The same y serves every family** |
| Checkpoints | FFHQ: `ffhq_10m.pt` (SHA `81d53574…7faa`), also DPS's FFHQ model. ImageNet: `/egr/research-pac/huang248/models/imagenet256.pt` (2,211,383,297 bytes; record SHA-256 in XP-0) |
| FAST-DIPS | R0.5 frozen settings (T = 150, K = 2, S = 5, ρ = 200, ε = 0.05), `FAST_DIPS_A15` @ `a9758b38`, env `fast_dips_a15_blackwell`. These equal the repository's published phase-retrieval command for both FFHQ and ImageNet (`configs_fast_dips_cli.txt`). ImageNet uses the `imagenet256ddpm` preset. **XP-G keeps every setting**, including ε (a fixed constraint tolerance in the code, not a noise parameter) |
| SITCOM | FFHQ: the B22 native settings (`configs/b22/b22_2_overnight.json`, `sitcom` block). ImageNet: SITCOM's own ImageNet phase-retrieval settings if its code or paper defines them; otherwise the FFHQ settings with the ImageNet model. Record which |
| DAPS, FFHQ | The historical configuration of the B22/DEV80 DAPS runs (confirm against the R0-N.2 A4.4 record) |
| DAPS, ImageNet and XP-E | The published DAPS-1K command at `e7a77d0` (`commands/pixel_imagenet.sh` / `pixel_ffhq.sh`): annealing 200 × diffusion 5 steps; pixel task group with 100 Langevin steps, lr 5e-5, τ = 0.01, lr_min_ratio 0.01. Our frozen y and per-(image, run) seeds are fed in through the in-repository wrapper. Record any deviation |
| DPS (XP-H) | The official DPS repository (`DPS2022/diffusion-posterior-sampling`) at a commit pinned in XP-0, using its phase-retrieval config (record the guidance scale) with `ffhq_10m.pt`. Our y is fed in. If the code is absent, **Cheng** clones it into a new external path |
| PC-DAPS (XP-I) | PC-DAPS @ `59b2f046` with its own FFHQ phase-retrieval command (4 particles). Our y is fed in. Save every particle, plus the method's own returned estimate. If the code is absent, **Cheng** clones it |
| Seeds | **Continue each family's historical rule** for new run indices. Document the rule, and show that no new seed equals an existing one for the same (family, image). For families or datasets with no history (all ImageNet, XP-G, XP-H, XP-I, XP-E): the first 8 bytes of SHA-256(`XP\|<family>\|<dataset>\|<uid>\|<r>`) mod 2³¹, where XP-G uses dataset `ffhq_s<σ>` |
| Batching | Batching across (image, run) pairs is allowed **only if** each sample draws from its own generator seeded by the rule above, and the replay gate (§4) passes for batched vs single execution. **Timing claims use XP-T batch-1 measurements only** |
| Labels, FFHQ (σ = 0.05) | R0 frozen labels: GOOD = PSNR_raw ≥ 25; ORBIT = raw < 25 and some non-identity member ≥ 25; CONSISTENT/INCONSISTENT by the **frozen R0.7 τ for that (source, family)**. τ is not re-estimated |
| Labels, ImageNet | **Primary threshold 20 dB**: GOOD_IN = PSNR_raw ≥ 20; ORBIT_IN = raw < 20 and orbit-best ≥ 20. τ_IN = the 0.95 quantile of ρ_signed over GOOD_IN runs per family, **computed on run indices 0–3 only and then frozen** for all later runs. **Secondary:** Good@{15, 18, 22, 25}, and the PC-DAPS orientation labels (amendment 2, A4.1 rule) |
| Labels, XP-G | As FFHQ with GOOD at 25 dB. τ_σ is computed per σ on run indices 0–3 and then frozen |
| Labels, new families (DPS, PC-DAPS) | As FFHQ. τ is computed on that family's runs 0–3 (all its runs, for PC-DAPS) and reported as such |
| Orientation tests | **FFHQ:** the frozen N2 CLIP face prompts. **Generic set G1:** "a photo." vs "an upside-down photo.". **Ensemble G2:** upright {"a photo.", "a photo of a scene.", "a photo of an object.", "a good photo.", "a natural photo.", "a photo of something."} vs inverted {"an upside-down photo.", "an upside-down photo of a scene.", "an upside-down photo of an object.", "a photo that is upside down.", "a photo rotated by 180 degrees.", "an inverted photo."}. Embedding, normalization and the rule "choose 111 iff u(R₁₁₁x) > u(x)" exactly as N2. **Prior:** the 2-forward rule (t = 300, J = 1, paired R ε, global2) and S_mid (t ∈ {300, 400, 500, 600}, J = 4; global2 and all8), each with the dataset's own checkpoint. **UnivFD:** as N2 |
| Evaluation metrics | PSNR raw and for all 8 orbit members; SSIM and LPIPS (piq, as DAPS) for raw and member 111 (LPIPS may run on GPU: it is evaluation only); ρ_signed, ρ_clipped |

## 2. Stages and tiers

GPU-hour figures are planner estimates from historical timings (FAST 3.3 s/run was measured on this hardware in R0.5; the others come from older campaigns). **XP-T replaces them.** Each estimate assumes one worker per GPU.

**Tier A: required, data cut Nov 2.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-0** | Bounded inventory (§4 of the handoff): GPUs; `test-imagenet` present (`test -d`; if absent, **Cheng** downloads via the DAPS README gdown link into a new data path); checkpoint SHAs; environment smoke imports; presence of the DPS and PC-DAPS code; the per-id y-SHA equality check of §1 | — | CPU |
| **XP-T** | Timing and throughput on **one GPU model**. Batch-1 timing: 3 runs per (family, dataset), covering DAPS, SITCOM, NP and FAST on the first 3 B22 ids (sorted), and FAST, SITCOM and DAPS on the first 3 ImageNet basenames; plus DPS and PC-DAPS if their code is present. CUDA-event sampling time and process wall. Per-candidate time for CLIP (2 encodings), the 2-forward prior and S_mid all8. **Throughput:** W ∈ {1, 2, 4} concurrent workers on one GPU, for FAST and SITCOM on FFHQ; choose W\* per family (largest aggregate throughput with memory ≤ 80% and outputs matching W = 1 within 0.05 dB PSNR) | ≈ 60 | ≈ 1 |
| **XP-A16** | FAST on FFHQ, 180 ids × run indices 4–15, giving **n = 16 per id** together with S5's runs 0–3 | 2,160 | ≈ 2 |
| **XP-C** | Orientation tests on clean ground truth: 100 ImageNet and 180 FFHQ GT images, with CLIP (face set for FFHQ only; G1; G2), the prior (2-forward; S_mid global2 and all8) and UnivFD. Metric: identity-win rate (000 preferred over 111; for all8, over all 7 others) | — | < 0.5 |
| **XP-C2** | **Equivariance-defect profile** of each checkpoint (tests THEORY_PROGRAM TH1.1 Cor. 4), on the same GT images. For t ∈ {100, 200, …, 900} (DDPM-1000 index, mapped to each code's own noise parameterization) and R ∈ {R₁₁₁, R₁₀₀}, draw x_t from x★ with 2 noise draws per (image, t), seed SHA-256(`XPC2\|<dataset>\|<basename>\|<t>\|<j>`). Report δ(t) = mean ‖D_t(R x_t) − R D_t(x_t)‖² / (3·256²), and δ̃(t) = δ(t) / mean ‖D_t(x_t) − x★‖² / (3·256²), per dataset with bootstrap CIs | — | < 0.3 |
| **XP-D1** | ImageNet measurements (§1), then FAST n = 16 on all 100. **Phase 1** = runs 0–3; **phase 2** = runs 4–15 | 1,600 | ≈ 3–6 |

**Tier B: main paper, data cut Nov 6.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-A64** | FAST on FFHQ, run indices 16–63 (n = 64) | 8,640 | ≈ 8 |
| **XP-B16** | SITCOM on FFHQ, **all 180 ids**: B22 runs 4–15 and DEV80 runs 4–15 (n = 16) | 2,160 | ≈ 29–53 |
| **XP-D2** | DAPS on ImageNet, n = 4 on all 100 (published DAPS-1K command) | 400 | ≈ 17–37 |
| **XP-D3** | SITCOM on ImageNet, n = 4 on all 100 | 400 | ≈ 11–28 |
| **XP-F8** | DAPS on FFHQ, all 180 ids: B22 runs 2–7 and DEV80 runs 4–7 (n = 8) | 920 | ≈ 40–50 |

**Tier C: supplement, data cut Nov 20.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-B32** | SITCOM on FFHQ, run indices 16–31 on all 180 ids (n = 32) | 2,880 | ≈ 39–71 |
| **XP-H** | DPS on FFHQ, n = 4 on all 180 ids | 720 | ≈ 12–30 |
| **XP-I** | PC-DAPS on FFHQ, 1 run (4 particles) on all 180 ids. Compare at matched NFE with DAPS + E1_CLIP | 180 | ≈ 16–32 |
| **XP-G** | FAST on FFHQ, σ ∈ {0.02, 0.10}, n = 8 per σ on all 180 ids | 2,880 | ≈ 3 |
| **XP-D4** | FAST on ImageNet, run indices 16–63 (n = 64) | 4,800 | ≈ 9–17 |
| **XP-E** | DAPS `test-ffhq` (100): FAST n = 4 and DAPS-1K n = 4 (bridge to published FFHQ numbers). **Only after Cheng confirms in writing that these ids are disjoint from confirmation305.** Agents never check membership | 800 | ≈ 10–18 |

**Scoring** (CLIP, 2-forward, S_mid, LPIPS) adds about 5–15 GPU-h in total. Estimated grand total: ≈ 210–370 GPU-h.

## 3. Queue rules

1. **Order of value** (launch priority): XP-T → XP-A16, XP-C, XP-C2, XP-D1 phase 1 → XP-D1 phase 2, XP-A64 → XP-B16 ∥ XP-D2 → XP-D3 ∥ XP-F8 → XP-B32 → XP-I → XP-H → XP-G → XP-D4 → XP-E.
2. **Work-conserving.** No GPU is left idle while any authorized, smoke-passed job is ready. A lower-priority job that is ready may fill a GPU while a higher one is still in engineering. Example: XP-B32 needs no new code, so it may run while XP-H/XP-I are being built.
3. **Concurrency:** W\* workers per GPU, per family, from XP-T. Never exceed 80% GPU memory.
4. **Smoke test per (family, dataset):** 2 images × 2 runs. Runtime must be within ±50% of the XP-T projection, and the configuration must replay the frozen settings. Otherwise stop that family and report.
5. **Drop order** if over budget or late (the reverse of value): XP-E → XP-D4 → XP-G → XP-H → XP-I → XP-B32 → XP-F8 → XP-D3 → XP-B16 reduced to the 60 ids of v1 (SHA-256(`XPB|<id>`), first 30 B22 + first 30 DEV80) → XP-D2 reduced to 25 images (SHA-256(`XPD|<basename>`)) → XP-A64. **Tier A is never dropped.**
6. **Budget check.** Before launching each stage, project its GPU-h from XP-T. Stop if the cumulative projection exceeds 400.

## 4. Gates for adding runs to historical families (XP-B, XP-F)

The K-scaling estimators pool new runs with historical runs, so the two must be **exchangeable**: the same distribution of outcomes given y.

1. **Replay gate.** On the current code, environment and GPU, re-run historical run indices 0 and 1 for the first 2 ids (sorted) of each (cohort, family). Pass if, for every replayed run, the raw PSNR is within 0.1 dB of the stored candidate and the R0 label is identical. Bitwise identity is recorded if it holds.
2. **If replay fails:** stop that family and report the differences. The planner then decides between (a) proceeding with new runs only (K-curves from new runs, with historical runs analyzed separately) and (b) a distribution check on 20 ids × 4 new runs.
3. **Seeds.** New seeds come from the historical rule (§1). Report the rule and the no-collision check.
4. FAST S5 (XP-A) was produced in R0.5 on this hardware. Its gate is the same replay of 2 ids × 2 runs.

## 5. Theory predictions, freeze order and embargo

1. **Freeze v1 (FFHQ).** Before any FFHQ outcome with run index ≥ 4 is analyzed or shared, the theory executor commits `docs/track_r/theory/TH1_PREDICTIONS_v1.json` on its theory branch. The fit uses only data with K ≤ 4 (R0 runs; A8 counts). It contains predicted values with 95% intervals:
   - **FAST**, S5 180 ids: K ∈ {8, 12, 16, 24, 32, 48, 64};
   - **SITCOM**, 180 ids, per cohort and pooled: K ∈ {8, 12, 16, 24, 32};
   - **DAPS**, 180 ids, per cohort and pooled: K ∈ {4 (B22), 6, 8};
   - policies E0, E1_CLIP, O_raw and O_orbit; metric Good25;
   - plus the model-implied **shape prediction**: the gaps O_orbit − E1_CLIP and O_raw − E0 are non-decreasing in K (THEORY_PROGRAM v1.1, TH1.2).
2. **Freeze v2 (ImageNet FAST, optional but preferred).** After the XP-D1 phase-1 per-image counts (runs 0–3) are committed, the theory executor may commit `TH1_PREDICTIONS_v2.json` for FAST ImageNet at K ∈ {8, 16} (and 32, 64 if XP-D4 runs), Good20, using runs 0–3 only. If v2 is not committed within 48 h of the phase-1 counts, ImageNet K-scaling is reported as descriptive only.
3. **Registry.** The planner records each freeze in `docs/track_r/theory/PREDICTIONS_REGISTRY.md` on the planner branch: theory branch, commit SHA, file SHA-256, commit time, and the time the planner saw it on GitHub.
4. **Order check.** The XP analysis script verifies, for each prediction file used, that its registry entry exists, that the SHA-256 of `git show <sha>:<path>` matches, and that its commit time precedes the XP commit that first contains the matching outcomes.
5. **Embargo.** Until the matching freeze is registered, the executor commits and reports **no outcome data** (labels, counts, metrics or curves) for:
   - FFHQ runs with index ≥ 4 in XP-A, XP-B or XP-F (and index ≥ 2 for B22 DAPS);
   - ImageNet FAST runs with index ≥ 4.

   **Allowed before the freeze:** completion receipts (run counts and failures only), timing, XP-C and XP-C2 results, XP-D1 phase-1 counts, XP-D2/XP-D3 outcomes (no predictions concern them), and the Tier C stages other than XP-B32 and XP-D4.
6. **Agreement criterion,** per prediction: SE_comb = √(SE_obs² + ((hi95 − lo95)/3.92)²). "Consistent" if |obs − pred| ≤ 2·SE_comb, otherwise "rejected". The shape prediction is "consistent" if no bootstrap-significant decrease (one-sided 95%) occurs between consecutive K in the grid. **Both outcomes are reported.**

## 6. Policies and estimators (analysis on CPU)

**Policies:**
- first run;
- **mean-of-K** (mean metric over the K runs);
- E0 (minimum ρ_signed; ties → lowest run index);
- **E1_CLIP**: E0 followed by the dataset's primary CLIP global decision (FFHQ: face set; ImageNet: G1);
- E1_CLIP_G2; E1_2F (2-forward prior); E1_S8 (the R0-frozen S_mid all8 repair);
- O_raw;
- O_orbit;
- **per-metric oracle**: best PSNR, best SSIM and best LPIPS chosen independently over the K runs (the DAPS/FAST-DIPS code convention);
- **single-pick oracle**: the best-PSNR run, with all of its metrics reported;
- for PC-DAPS only, its own returned estimate (as published) in addition to the above on its particles.

**Estimators.**
- **Primary: exact U-statistics** over all K-subsets of an image's n runs, by order-statistic identities (no enumeration, no random subsampling). For a selector that picks the argmin of a score, with the runs ranked 1…n by ascending score (ties by run index), run j of rank r_j is picked with probability C(n − r_j, K − 1) / C(n, K). For a max-type oracle, the run of ascending rank j is the maximum with probability C(j − 1, K − 1) / C(n, K). Good-type oracles use 1 − C(n − g, K) / C(n, K) (pass@k form).
- **Continuity:** also report the first-K version (runs 0 … K−1).
- **CIs:** image-cluster bootstrap, 10,000 replicates, seed rule `XPBOOT|<stage>`.

**Outputs:**
- K-curves for every policy (Good25 or Good20, mean, q05), with predictions overlaid where frozen;
- a compute frontier: success vs seconds per image, from XP-T batch-1 timings only;
- failure anatomy per dataset and family, and the post-selection decomposition (generation failure vs selection miss vs uncorrected orientation) as a function of K;
- orientation-test accuracy per domain (XP-C on GT; on reconstructions: fix rate and harm rate);
- the inflation decomposition per metric: per-metric oracle vs single-pick vs E1_CLIP vs mean-of-K vs first run;
- XP-D2: the published DAPS protocol (per-metric oracle over 4 runs) next to GT-free E1_CLIP and first run;
- XP-I: PC-DAPS vs DAPS + E1_CLIP at matched NFE on the same ids;
- XP-G: anatomy and E1_CLIP vs O_orbit at σ ∈ {0.02, 0.05, 0.10}.

## 7. Outputs and data rules

- **PAC roots:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTCSTAMP>/`.
- **Committed** (`docs/track_r/executors/xp/returns/XP_<UTCSTAMP>/`): aggregate CSV/JSON, figures, SHA256SUMS, and **per-image strings** (one row per (dataset, cohort, family, stage, id)):
  - counts: `n_runs`, `n_GOOD`, `n_ORBIT_global`, `n_ORBIT_partial`, `n_CONSISTENT`, `n_INCONSISTENT`;
  - `run_labels` in run order, alphabet G (GOOD), O (ORBIT global: member 111 is the best member), P (ORBIT partial), C, I;
  - `e0_rank_order`: run indices sorted by ascending ρ_signed (ties by index), comma-separated;
  - GT-free decisions per run in run order: `dec_clip_face` (FFHQ), `dec_clip_g1`, `dec_clip_g2`, `dec_prior2f` as 0/1 (1 = choose 111), and `dec_smid_all8` as the chosen member index 0–7.
  - **No paths, metric values or images.** Labels are GT-derived; Cheng approved committing such strings on 2026-10-26.
- **Delivered as a planner ZIP, not committed:** per-candidate tables (with metric values) and contact sheets (as in amendment 2 §A7).
- **GT** is used only for evaluation and labels. Selection policies never read GT.
- **Forbidden:** access to confirmation305; FFHQ ids beyond §1; any change to a historical environment, output or checkout.
