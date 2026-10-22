# R0_EXECUTOR_RETURN — template (copy to `R0_EXECUTOR_RETURN.md` and fill)

## Status for Cheng (≤ 8 lines, plain language)
What ran, what passed or failed, headline numbers, what is next.

## 1. Identity

- **Executor branch:**
- **Executor HEADs used on PAC, per stage:**
  - R0.0
  - R0.2
  - R0.3
  - R0.4
  - R0.5
  - R0.6
  - R0.7
- **Planner branch SHA this work started from:**
- **Spec version:** v1, frozen 2026-10-21.
  - `configs/track_r/r0_frozen_v1.json` SHA-256:
- **Verified paths file:** `configs/track_r/pac_paths_track_r_verified.json` at commit:
- **Interpreters used:**

  | Interpreter | Path | torch version | CUDA arch list |
  |---|---|---|---|
  | prior scoring | | | |
  | UnivFD | | | |
  | FAST | | | |

- **Model and asset SHA-256 (as verified on PAC):**
  - `ffhq_10m.pt`:
  - `ViT-L-14.pt`:
  - `fc_weights.pth`:
  - FAST source HEAD:
  - UnivFD HEAD:

## 2. Execution ledger

| Stage | Command block (link to log) | Start / end UTC | GPUs (physical index, UUID) | GPU-hours | rc | Notes |
|---|---|---|---|---|---|---|

**Totals:**
- Aggregate GPU-hours (ceiling 16):
- UNet forwards:
- CLIP forwards:
- FAST roots:
- Denoiser forwards per FAST root (measured):

## 3. Gates

| Gate | Result | Evidence (file and key numbers) |
|---|---|---|
| G0.1 ρ_signed(GT) range, per source | PASS / WARN / FAIL | list WARN/FAIL images |
| G0.2 orbit invariance | | max relative deviation |
| G0.3 recorded-PSNR replay | | fraction within 0.01 dB; mismatches |
| G0.4 determinism | | max relative difference |
| G0.5 forbidden-path assertion | | |
| G1 canonical 68539 DAPS rep2 | | raw / repaired PSNR, chosen pattern, margin, z |
| FAST worker replay vs A1.5 | | bit-exact? PSNR difference |

## 4. Candidate counts

| Source | Family | Images | Runs per image | Candidates | Expected | Explanation if different |
|---|---|---|---|---|---|---|

## 5. Primary results (numbers only; CI = image-cluster bootstrap 95%)

**P1 anatomy.** Per source × family, plus pooled per family:

| Source | Family | GOOD | ORBIT_ERROR (GLOBAL / PARTIAL) | NONORBIT_CONSISTENT | NONORBIT_INCONSISTENT | Orbit share of failures [CI] |
|---|---|---|---|---|---|---|

**P2 repair.** Pooled and per family; rows for prior S, prior S_mid and UnivFD-orbit:

| Scorer | Family | ORBIT_ERROR n | Success | Exact s* = s_gt | GOOD n | Harm |
|---|---|---|---|---|---|---|

**P3 detection (after repair).**

| Family | Failure type | n fail / n total | AUROC S | AUROC S_img | AUROC p_fake | AUROC ρ_signed | AUROC ρ_clipped |
|---|---|---|---|---|---|---|---|

**P4 executable policies.** Per source × family × K:

| Source | Family | K | Policy | Mean | Median | Good25 | Bad20 | Bad15 | q05 | Min |
|---|---|---|---|---|---|---|---|---|---|---|

Paired contrasts (mean difference [CI], Good25 McNemar p):
- E1−E0:
- E2−E1:
- E2−E3:

## 6. Decision table (computed mechanically; the planner decides)

| Decision | Rule | Value | Status |
|---|---|---|---|
| D-material | orbit share ≥ 0.20 for FAST or DAPS | | |
| D-repair | success ≥ 0.90 and harm ≤ 0.01 (FAIL if < 0.75 or > 0.03) | | |
| D-detect | AUROC(S or S_img) − AUROC(p_fake), FAST and DAPS | | |

## 7. Secondary and EXPLORATORY results
Per-t curves, complexity-bias statistics, cost per candidate, FAST timing (sampling-only vs wall), optional B26/A1.6 rows. Label everything that is not pre-registered as EXPLORATORY.

## 8. Anomalies, deviations, open questions
Every deviation from the spec, with its reason and its effect on the results.

## 9. Artifacts

- **PAC roots:** one per stage.
- **Return zip:** path and SHA-256.
- **Committed compact outputs:** `docs/track_r/executors/r0/returns/R0_<UTCSTAMP>/`.
- **Contact sheets:** paths.

## 10. Executor opinion (optional, clearly separated)
