# XP pre-registration v1 — evaluation-only extensions for the audit paper

**Status:** FROZEN v1 (planner, 2026-10-25). **Execution requires Cheng's explicit "GO XP".** Until then, no GPU work.

**Purpose.** Strengthen the audit (top-venue attempt; plan in `2026-10-25_TOP_VENUE_PLAN.md`) with:
- one more dataset;
- K-scaling that **tests pre-registered theory predictions** (TH1.5);
- same-GPU timing;
- orientation tests across domains.

**Constraints.**
- **No new method.** Every result is reported whatever its outcome.
- **Cap:** 45 GPU-hours total, all stages included. **Drop order** if over budget or late: XP-B → DAPS on ImageNet → SITCOM on ImageNet.

## 1. Fixed objects

| Object | Frozen choice |
|---|---|
| Operator | R0 pinned operator A(x) = \|F_c(pad₆₄((x+1)/2))\|, 384² canvas, channel-wise, σ = 0.05, signed y |
| FFHQ ids | The 180 ids of B22 (100) and DEV80 (80), exactly as in the R0 manifests. **No other FFHQ id** (XP-E is the conditional exception) |
| FFHQ measurements | **Reuse the R0 measurement file per id**, the same y used by S5. No new FFHQ measurements |
| ImageNet images | DAPS's 100-image `test-imagenet` set, the same set FAST-DIPS and PC-DAPS evaluate on, in the DAPS/FAST-DIPS loader order: `sorted(Path(root).rglob(ext))`, index 0–99. Record each file's SHA-256 in the XP-0 manifest |
| ImageNet measurements | New. y = A(x) + σ·n with n ~ N(0, I). Seed = first 8 bytes of SHA-256(`XPMEAS\|imagenet\|<basename>`) as an unsigned integer mod 2⁶³. Save y and its SHA |
| Checkpoints | FFHQ: `ffhq_10m.pt` (SHA `81d53574…7faa`). ImageNet: `/egr/research-pac/huang248/models/imagenet256.pt` (2,211,383,297 bytes; record the SHA-256 in XP-0) |
| FAST-DIPS | R0.5 frozen settings (T=150, K=2, S=5, ρ=200, ε=0.05), `FAST_DIPS_A15` @ `a9758b38`, env `fast_dips_a15_blackwell`. ImageNet uses the repository's `imagenet256ddpm` preset with the same sampler settings |
| SITCOM | FFHQ: the B22 native settings (`configs/b22/b22_2_overnight.json`, `sitcom` block). ImageNet: SITCOM's own ImageNet phase-retrieval settings if its code or paper defines them; otherwise the FFHQ settings with the ImageNet model. Record which |
| DAPS | Historical config: 400 annealing × 5 ODE steps (confirm against the R0-N.2 A4.4 record). ImageNet uses the `imagenet256ddpm` model |
| Seeds | **Continue each family's historical rule** for new run indices. Document the rule, and show that no new seed equals an existing one for the same (family, image). If no rule is recoverable, use the first 8 bytes of SHA-256(`XP\|<family>\|<dataset>\|<uid>\|<r>`) mod 2³¹ |
| Labels, FFHQ | R0 frozen labels: GOOD = PSNR_raw ≥ 25; ORBIT; CONSISTENT/INCONSISTENT by the **frozen R0.7 τ for that (source, family)**. τ is not re-estimated |
| Labels, ImageNet | **Primary threshold 20 dB**: GOOD_IN = PSNR_raw ≥ 20; ORBIT_IN = raw < 20 and orbit-best ≥ 20; residual τ_IN = the 0.95 quantile of ρ_signed over GOOD_IN runs per family (computed once on XP-D). **Secondary:** Good@τ curves for τ ∈ {15, 18, 22, 25}, and PC-DAPS orientation labels (amendment 2, A4.1 rule) |
| Orientation tests | **FFHQ:** the frozen N2 CLIP face prompts. **Generic set G1:** "a photo." vs "an upside-down photo.". **Ensemble G2:** upright {"a photo.", "a photo of a scene.", "a photo of an object.", "a good photo.", "a natural photo.", "a photo of something."} vs inverted {"an upside-down photo.", "an upside-down photo of a scene.", "an upside-down photo of an object.", "a photo that is upside down.", "a photo rotated by 180 degrees.", "an inverted photo."}. Embedding, normalization and the "choose 111 iff u(R₁₁₁x) > u(x)" rule exactly as N2. **Prior:** 2-forward rule (t = 300, J = 1, paired R ε, global2) and S_mid (t ∈ {300, 400, 500, 600}, J = 4; global2 and all8), each with the dataset's own checkpoint. **UnivFD:** as N2 |
| Evaluation metrics | PSNR raw and for all 8 orbit members; SSIM and LPIPS (piq, as DAPS) for raw and member 111; ρ_signed, ρ_clipped |

## 2. Stages

| Stage | Content | Compute | Required? |
|---|---|---|---|
| **XP-0** | Bounded inventory: GPU count, model and free memory; `test-imagenet` present (`test -d` under the DAPS/FAST checkouts; if absent, **Cheng** downloads it via the DAPS README gdown link into a new data path); `imagenet256.pt` SHA; environment checks (DAPS, SITCOM, FAST, CLIP/UnivFD, prdiff_track_r) | CPU | Yes |
| **XP-T** | Timing calibration on **one GPU model**: 3 runs each of DAPS, SITCOM, NP and FAST on the first 3 B22 ids (sorted). CUDA-event sampling-only and process wall, batch 1. Also measure CLIP (2 encodings) and 2-forward-prior per-candidate time on 100 FFHQ candidates | ≈0.3 GPU-h | Yes |
| **XP-A** | FAST on FFHQ, 180 ids × run indices 4–15 (12 new runs per id). Together with S5's indices 0–3 this gives **n = 16 per id** | ≈2 GPU-h | Yes |
| **XP-C** | Orientation tests on clean ground truth: 100 ImageNet GT images and the 180 FFHQ GT images. CLIP (face set for FFHQ only; G1, G2), prior (2-forward; S_mid global2 and all8), UnivFD. Metric: identity-win rate (000 preferred over 111; for all8, over all 7 others) | Minutes | Yes |
| **XP-D** | ImageNet audit: FAST n = 8 on all 100; SITCOM n = 4 on all 100; DAPS n = 4 on 25 images (the first 25 when sorting basenames by SHA-256(`XPD\|<basename>`)). Score exactly as R0 plus the §1 metrics and labels | FAST ≈4 GPU-h; SITCOM ≈11; DAPS ≈8–12 (parallel over GPUs) | FAST yes; SITCOM and DAPS best effort |
| **XP-B** | SITCOM on FFHQ, 60 ids × run indices 4–15 (n = 16). Ids: the first 30 B22 and first 30 DEV80 ids when sorted by SHA-256(`XPB\|<id>`) | ≈14 GPU-h | Optional (tests the TH1 SITCOM ceiling) |
| **XP-E** | FAST n = 4 on DAPS `test-ffhq` (100). **Only after Cheng confirms in writing that these ids are disjoint from confirmation305.** Agents never check membership | ≈0.4 GPU-h | Optional |

**Smoke tests.** Each family: 2 images × 2 runs per dataset. Runtime must be within ±50% of the XP-T projection, and the replayed configuration must match the frozen settings. Otherwise stop and report.

## 3. Policies and estimators (analysis on CPU)

**Policies:**
- first run;
- **mean-of-K**;
- E0 (minimum ρ_signed);
- **E1_CLIP**: E0 followed by the dataset's primary CLIP global decision (FFHQ: face set; ImageNet: G1);
- E1_2F (2-forward prior);
- O_raw;
- O_orbit;
- **per-metric oracle**: best PSNR, best SSIM and best LPIPS chosen independently over the K runs;
- **single-pick oracle**: the best-PSNR run, with all of its metrics reported.

**Estimators.**
- **Primary:** for every K ≤ n, average each policy's outcome over **all K-subsets** of an image's n runs (a U-statistic). Then average over images.
- **For continuity:** also report the first-K version.
- **CIs:** image-cluster bootstrap, 10,000 replicates, seed rule `XPBOOT|<stage>`.

**Outputs:**
- K-curves for every policy (Good25 or Good20, mean, q05);
- a compute frontier: success vs seconds per image, using **XP-T timings** only;
- anatomy per dataset;
- the post-selection decomposition;
- orientation-test accuracy per domain;
- the inflation decomposition: per-metric oracle vs single-pick vs E1_CLIP vs mean-of-K vs first run, for each metric.

## 4. Pre-registered theory predictions (TH1.5)

1. **Freeze first.** Before any XP-A or XP-B run with index ≥ 4 is **analyzed**, the theory executor commits `docs/track_r/theory/TH1_PREDICTIONS_v1.json`. It holds the predicted curves, with intervals, for E1_CLIP, O_raw and O_orbit at K ∈ {8, 12, 16}: FAST on FFHQ (and SITCOM on the XP-B ids, if XP-B runs). The fit uses only data with K ≤ 4 (R0 runs).
2. **Order check.** The analysis script verifies that this commit precedes the XP return commit.
3. **Agreement criterion**, per prediction: |observed − predicted| ≤ 2 bootstrap SE (combining model and data uncertainty) → "consistent", otherwise "rejected".
4. Both outcomes are reported.

## 5. Outputs and data rules

- **PAC roots:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTCSTAMP>/`.
- **Committed** (`docs/track_r/executors/xp/returns/XP_<UTCSTAMP>/`): aggregate CSV/JSON, **per-image label counts per (dataset, family)**, figures, SHA256SUMS.
- **Delivered as a planner ZIP, not committed:** per-candidate tables, contact sheets (as in amendment 2 §A7).
- **GT** is used only for evaluation. Selection policies never read GT.
- **Forbidden:** access to confirmation305; new FFHQ ids beyond §1.
