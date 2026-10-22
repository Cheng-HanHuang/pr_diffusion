# R0 experiment specification

**Topic:** failure anatomy, prior-ratio orbit repair and prior-consistent detection.

**Version:** FROZEN v1, 2026-10-21, planner Claude. Machine-readable twin: `configs/track_r/r0_frozen_v1.json`. If this document and the JSON disagree, stop and ask the planner.

**Change control:**
- Any change to a definition, threshold, data source or primary analysis needs a planner amendment (v1.1, v2, …) committed to the planner branch before results are viewed.
- The executor may add analyses only under an `EXPLORATORY` label in a separate output section.

**Scope:** exposed data only. R0 is diagnosis and development. Its numbers may appear in a paper only as clearly labeled retrospective analysis.

---

## 1. Questions

| ID | Question |
|---|---|
| Q1 — ceiling | Among single-run failures (raw RGB8 PSNR < 25 dB) of each solver family, what fraction are **exact-orbit errors**? An exact-orbit error is a run where some other member of its per-channel 180° orbit reaches PSNR ≥ 25 dB. |
| Q2 — repair | Does the GT-free prior score (paired-noise diffusion denoising loss) choose an orbit member with PSNR ≥ 25 on exact-orbit errors, without flipping good outputs? Compared with UnivFD-scored orbit choice. |
| Q3 — detection | After repair, does the prior score separate remaining failures from successes (AUROC) as well as UnivFD and the signed/clipped measurement residuals? |
| Q4 — net effect | For GT-free K-run policies (K ∈ {1, 2, 4}; 8 for B22 NP), do repair and prior-based selection close the gap to the GT best-of-K ceiling in Good25, q05, Bad20, mean and min? |

## 2. Data sources (all exposed; read-only)

Paths are listed in `configs/track_r/pac_paths_track_r_v1.json`. They are copied from repo records and **must be verified in R0.0**.

| Source | Images | Families and runs per image | Notes |
|---|---|---|---|
| S1 — B22 final100 | 100, locked `meas5401` | DAPS: Fresh1 + Fresh2 runs (from B21.11) · SITCOM: 4 (sequential RNG, not independent) · NP: 8 (LF and S2 configs × seeds 100–103) | GT and measurement as used in B21/B22 |
| S2 — B24 DEV80 | 80, locked dev measurements | DAPS4, SITCOM4, NP4_INDEPENDENT (4 each) | Per-image GT and measurement paths in `configs/b26/b26_dev80_manifest.json` |
| S3 — A1.2 exposed eight | 8 | DAPS, SITCOM, NP_LF, NP_S2 × 4 reps | Paths from `docs/track_a/executors/a1-oracle-budget/A1_PAC_ARTIFACT_INDEX_20261020.json` |
| S4 — A1.5 | same 8 | native FAST-DIPS × 4 reps | A1.5 output root |
| S5 — new (R0.5) | S1's 100 + S2's 80 | native FAST-DIPS × 4 independent roots | Pinned A1.5 FAST config and source; engineering generation on exposed images |
| Optional exploratory | — | B26 H/R NP roots on DEV80; A1.6 arms | Score only if time allows; label EXPLORATORY |

**Never access:**
- B24 confirmation305: its registry, membership lists and payloads.
- Any path containing `confirmation` or `confirm305` under `/egr/research-pac/huang248/outputs/pr_diffusion/b24/`.
- The manifest builder asserts this for every path it emits.

**Not used:** B19.20, because its sample paths are unrecoverable (B21.2 record).

## 3. Frozen definitions

### 3.1 Image representation

- Model range is [-1, 1], tensor [1, 3, 256, 256].
- **Canonical scoring representation: RGB8.**
  - Tensors: x8 = round((clamp(x, −1, 1) + 1) · 127.5).
  - PNGs: PIL `convert("RGB")`.
  - Model input: x_c = x8 / 127.5 − 1.
- All primary metrics, residuals and scores use x_c for every family. Reason: the documented float-versus-PNG residual differences in A1.3, finding F.
- Float-tensor variants are EXPLORATORY.

### 3.2 Forward operator

- A(x) = |F_c(pad₆₄((x + 1)/2))| on a 384 × 384 canvas, channelwise.
- F_c = fftshift ∘ fftn(norm = "ortho") ∘ ifftshift over the last two dimensions.
- This is the pinned DAPS `PhaseRetrieval` operator, matches FAST-DIPS, and is implemented in `scripts/track_a/a13_physics.py::magnitude`, which already has a centered-FFT witness test.
- m = 3 · 384² = 442,368.

### 3.3 Measurement

- y is the saved signed noisy magnitude. It is never clipped in R0 metrics.
- y₊ = max(y, 0) is used only for the clipped-residual baseline.

### 3.4 Orbit

- For a pattern s = (s_R, s_G, s_B) ∈ {0,1}³, R_s rotates channel c by 180° (`torch.rot90(·, 2, dims=(-2,-1))`) iff s_c = 1.
- Eight members. `000` is the identity; `111` is the global 180° rotation.
- **Exact invariance** A(R_s x) = A(x) holds because, within the symmetric padding, a 180° rotation is a conjugate reflection composed with a one-pixel circular shift. It must be unit-tested.

### 3.5 PSNR

- PSNR(x, g) = −10 log₁₀ mean((x_c01 − g_c01)²), with both x and g in canonical RGB8 scaled to [0, 1].
- g is the source's **saved** GT tensor, quantized the same way. Do not re-derive GT from the FFHQ PNG when a saved GT exists.
- If a source has no saved GT tensor, the executor documents the exact derivation used by that source, and the G0 replay gate decides.

### 3.6 Labels (GT used offline only)

**Primary labels:**

| Label | Rule |
|---|---|
| GOOD | PSNR_raw ≥ 25 |
| ORBIT_ERROR | PSNR_raw < 25 and max_{s ≠ 000} PSNR(R_s x) ≥ 25. Record s_gt = argmax_s PSNR(R_s x) and the subtype: GLOBAL (s_gt = 111) or PARTIAL (any other non-identity pattern). |
| NONORBIT_CONSISTENT | Neither of the above, and ρ_signed(x) ≤ τ_{source,family}. τ is the 0.95 quantile of ρ_signed over GOOD candidates of the same source and family. |
| NONORBIT_INCONSISTENT | All remaining failures. |

**PSNR bands (separate fields):**
- CATASTROPHIC < 15 dB
- BAD < 20 dB
- MARGINAL in [20, 25) dB

### 3.7 Prior score (GT-free)

**Model and schedule:**
- Model: `ffhq_10m.pt` (SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`).
- Architecture: guided-diffusion preset `difffpr_ffhq_10m` (`prdiffusion/guided_backend.py`).
- Schedule: linear β from 1e-4 to 0.02 over 1000 steps; ε = first 3 output channels.

**Sampling grid:**
- Timesteps: T_set = {50, 100, 200, 300, 400, 500, 600, 700} (0-indexed DDPM steps).
- J = 4 noise draws per t.
- ε_{t,j} ~ N(0, I) from a CPU `torch.Generator` seeded with the first 8 bytes of SHA-256(`R0PRIOR|{candidate_uid}|{t}|{j}`), little-endian, masked to 63 bits. The noise is moved to the device after generation, so it is identical across GPUs.

**Paired coupling:**
- Orbit member s uses noise R_s ε_{t,j}, so its noisy input is exactly R_s applied to the identity member's noisy input.
- With a rotation-equivariant model all members would score identically. Any difference is the prior's orientation and colour structure.

**Loss and score:**
- Per-draw loss: ℓ_s(t, j) = mean over C, H, W of (ε_θ(√ᾱ_t R_s x_c + √(1 − ᾱ_t) R_s ε_{t,j}, t) − R_s ε_{t,j})².
- **Score:** S(R_s x) = equal-weight mean of ℓ_s over all (t, j). Lower means more plausible.
- Also store the per-t means.
- Pre-registered secondary score: S_mid, using only t ∈ {300, 400, 500, 600}.

**Repair rule:**
- s* = argmin_s S(R_s x). Ties within 1e-7 go to `000`.
- Repaired image: x' = R_{s*} x.

**Confidence:**
- Paired differences d_{t,j} = ℓ_000 − ℓ_{s*}.
- margin = mean(d); z = mean(d) / (sd(d) / √(|T_set| · J)).

**Cost and precision:**
- Cost: 8 × 8 × 4 = 256 UNet forwards per candidate. Record the actual count.
- Precision: float32. Batch size is free, but results must agree within 1e-5 relative across batch sizes and GPUs (determinism gate).

### 3.8 UnivFD score (baseline: FA-FPR's detector)

**Components:**
- Repository `Yuheng-Li/UniversalFakeDetect` @ `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` (MIT license).
- `pretrained_weights/fc_weights.pth`, SHA-256 `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847`.
- CLIP ViT-L/14 weights `ViT-L-14.pt`, SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.

**Scoring:**
- Input: canonical RGB8 image → `ToTensor` → `CenterCrop(224)`, with no resize, as in `validate.py` → normalize with CLIP mean/std.
- Output: p_fake = sigmoid(logit).
- Score all 8 orbit members (for the "UnivFD-scored orbit" baseline, s_ufd = argmin p_fake) and the prior-repaired image.

### 3.9 Residual scores (baselines)

- ρ_signed(x) = ‖A(x) − y‖₂ / √m.
- ρ_clipped(x) = ‖A(x) − y₊‖₂ / √m.
- Native selector values (SITCOM correction norm, NP selector statistic, DAPS operator-loss margin) are carried from source records where they exist. Otherwise they are recorded as N/A — never invented.

## 4. Outcomes

**P1 — Q1, anatomy:**
- For each (source, family), count GOOD, ORBIT_ERROR (GLOBAL/PARTIAL), NONORBIT_CONSISTENT and NONORBIT_INCONSISTENT over all single runs.
- Report the ORBIT share among failures with an image-cluster bootstrap 95% CI: 10,000 replicates, generator seeded from SHA-256(`R0BOOT|{analysis_id}`).
- Also report pooled-by-family results across sources.

**P2 — Q2, repair:**
- On ORBIT_ERROR candidates: success = P(PSNR(x') ≥ 25); exact = P(s* = s_gt).
- On GOOD candidates: harm = P(PSNR(x') < 25).
- Same quantities for s_ufd.
- Report pooled and per family; S_mid as secondary.

**P3 — Q3, detection.** After repair, the failure label is PSNR(x') < 25. Compute AUROC for each score:
- S(x');
- S_img(x') = S(x') minus the median S of the same image's repaired candidates of the same source and family;
- p_fake(x');
- ρ_signed and ρ_clipped.

Report per family, and per failure type (NONORBIT_CONSISTENT versus NONORBIT_INCONSISTENT).

**P4 — Q4, executable policies.** Use the first K runs in recorded order; for B22 NP use the 8-run set with K ∈ {1, 2, 4, 8}.

| Policy | Rule |
|---|---|
| E0 | argmin ρ_signed, no repair |
| E1 | E0's choice, then repair |
| E2 | Repair all; feasible set F = {ρ_signed(x') ≤ τ_{source,family}}; choose argmin S(x') over F (argmin ρ_signed if F is empty) |
| E3 | As E2 but argmin p_fake over F (FA-FPR-style detector selection, no restart) |
| O_raw (ceiling) | max_k PSNR_raw (GT) |
| O_orbit (ceiling) | max over k and over s of PSNR(R_s x_k) (GT; "DiffFPR-style GT alignment") |

- Report mean, median, Good25, Bad20, Bad15, q05 and min.
- Paired image-bootstrap CIs for E1−E0, E2−E1 and E2−E3.
- McNemar tests on Good25 for the same pairs.

**Secondary:**
- Per-t AUROC and per-t repair accuracy.
- Complexity bias: Spearman correlation of S with high-frequency energy (fraction of spectral energy outside radius 0.25 in cycles/pixel) among GOOD candidates. Also record how often argmin S picks the lowest-high-frequency candidate within an image.
- Cost per candidate: UNet forwards and seconds.
- FAST timing: CUDA-event sampling-only time and per-root wall time.

## 5. Gates and decisions (pre-registered)

### G0 — validity (all must pass before any result is interpreted)

1. **Operator and pairing.** ρ_signed(GT) ∈ [0.0495, 0.0505] for every image of every source: PASS. Values in (0.0505, 0.055] trigger WARN, listed per image (suspected GT preprocessing mismatch). Anything else is FAIL → STOP.
2. **Orbit invariance.** max |A(R_s x) − A(x)| ≤ 1e-4 · max |A(x)| for 16 random x and for 3 real GT images.
3. **Replay.** Identity-orientation PSNR recomputed with each source's own recorded convention matches the recorded value within 0.01 dB for ≥ 99% of candidates. B22 recorded float PSNR; A1.x and B24 recorded RGB8. List every mismatch.
4. **Determinism.** Prior scores for 8 candidates recomputed with a different batch size and on a different physical GPU agree within 1e-5 relative.
5. **Forbidden-path assertion** passes for the whole manifest.

### G1 — canonical sanity (before full scoring)

- A1.2 image 68539, DAPS rep2 (A1.3: 6.364 dB raw, 31.871 dB after rotation) must be repaired to PSNR ≥ 25.
- Report the result regardless. If it fails, stop and return to the planner before R0.4.

### Decisions (the executor reports numbers; the planner decides)

| Decision | Rule |
|---|---|
| D-material | ORBIT share of single-run failures ≥ 20% for FAST (S4 + S5) **or** for DAPS (S1 + S2 + S3). |
| D-repair | PASS: success ≥ 90% on pooled ORBIT_ERROR and harm ≤ 1% on pooled GOOD. FAIL: success < 75% or harm > 3%. |
| D-detect | "Prior superior": AUROC(S or S_img) − AUROC(p_fake) ≥ +0.03 for both FAST and DAPS. "Non-inferior": ≥ −0.02. |

| Outcome | Next step |
|---|---|
| D-material and D-repair PASS | R1 = repair-and-detect pipeline on FAST |
| D-material FAIL, D-repair PASS, D-detect superior | R1 centered on prior-consistent detection/restart, with repair as a component |
| D-repair FAIL | Stop orbit repair. If D-detect superior → detection-only R1; otherwise stop Track R and decide between a short evaluation note and dropping it |
| Grey zone | Planner decides with the full tables; the reasoning is recorded |

## 6. Compute budget

| Item | Estimate |
|---|---|
| Prior scoring | 256 UNet forwards × ~2,500 existing + 720 FAST candidates ≈ 0.8 M forwards; estimated 1–4 GPU-hours (measure in smoke) |
| UnivFD | 8 orbit members + 1 repaired image per candidate ≈ 30 k CLIP-L forwards; under 0.5 GPU-hours |
| FAST generation | 720 roots; under 2 GPU-hours |
| **Hard ceiling** | **16 aggregate GPU-hours** for all of R0. Stop and report if a projection exceeds it. |

## 7. What R0 cannot claim

- No prospective or generalization claims: images are exposed and tuning-adjacent.
- No claim about confirmation305 or any untouched cohort.
- No claim that orbit enumeration, failure detection, or "prior first" scheduling is new (see the [literature basis](2026-10-21_LITERATURE_BASIS.md)).
