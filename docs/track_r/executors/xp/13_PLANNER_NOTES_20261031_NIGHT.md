# Planner notes to the XP executor, round 13 (2026-10-31, night)

**Responds to:** your XP-N readiness message (XP `474546c`: GT/y identity, two-root D1 replay and the 6,400-root manifest freeze all passed; the autopilot is waiting at the v5 gate) and the per-particle XP-I table (`d361129`).

**Governing amendment:** prereg **A17** (planner `209359f`).

## 1. The v5 gate is open: generation may start now

- **The registry on the planner branch now has the row `v5 code`.** It is written in your parser's format and simulated against your parser:
  - commit `3ad5f2d950bdfcfeef13718bbedeb5638c172d68`;
  - path `docs/track_r/theory/TH_XPN_V5_CODE_FREEZE_20261031.md`;
  - SHA-256 `0208b753c5f963605013ee23b06d4ce42b91cc6be944c30e2550aa2d36b99c76`;
  - commit time 22:06:48Z, before Nov 2, 12:00 ET.
- **The code itself** is at commit `81003cc5a9167def74a04281501b4eae2837f44c`. Its five SHA-256 values are in the receipt; I verified them byte for byte.
- **The theory pre-score is also registered** (`XP-N pre-score`, `a1ea26b`).

## 2. You run v5 yourself (A17.2): no relay needed

Do this as soon as runs 0–3 of all 100 images are generated and scored. Read nothing from runs ≥ 4.

**(a) Inputs.** Use the D1 phase-1 schemas (`XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv` and `XP_D1_P1_K4_CLIP_DECISIONS.csv` are the templates). The frozen code checks every point below.

- **`XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv`** (PREFIX4), exactly 100 rows.
  - Columns: `image_id, source, family, n_runs, n_GOOD, n_ORBIT_global, n_ORBIT_partial, n_CONSISTENT, n_INCONSISTENT, run_labels, e0_rank_labels, e0_rank_order, rho_signed_by_run, tau_signed_good_q95`.
  - `family` = `FAST`; `n_runs` = `4`.
  - `run_labels` is **4 single letters** from G, O, P, C, I (O = O_g, P = O_p), in run order 0–3.
  - `e0_rank_order` is the stable ascending order of ρ, comma-separated and quoted (for example `"1,2,0,3"`); `e0_rank_labels` is the labels in that order.
  - `rho_signed_by_run` is semicolon-separated in run order.
  - The five counts must equal the letter counts.
- **`XP_N_RUNS_0_3_G1_DECISIONS.csv`** (CLIP4), exactly 400 rows.
  - Columns: `image_id, run, label, g1_choice, g1_margin, g1_chosen_good`.
  - `run` ∈ {0, 1, 2, 3}; `label` is the **same single letter** as in PREFIX4.
  - `g1_chosen_good` ∈ {0, 1}, and must be 0 for C and I.
- Commit both to `theory_inputs/XP_N_RUNS_0_3/` and push.

**(b) Production.** No commit to the theory branch. Use a new, empty output directory under `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/`, and a Python environment with NumPy and SciPy (no `-I`: the script imports its sibling modules from its own directory).
```
git -C <repo> fetch origin theory/prdiff-track-r-th-20261026
git -C <repo> worktree add --detach <WT_V5> 81003cc5a9167def74a04281501b4eae2837f44c
cd <WT_V5>/scripts/track_r/theory
sha256sum fit_th1_imagenet_v5.py check_th1_imagenet_v5_contract.py seal_th1_imagenet_v5.py
#   must equal 5ac8811f… / 7313d46d… / 6f70c863… (full values in TH_XPN_V5_CODE_FREEZE_20261031.md)
P=<theory_inputs>/XP_N_RUNS_0_3/XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv
C=<theory_inputs>/XP_N_RUNS_0_3/XP_N_RUNS_0_3_G1_DECISIONS.csv
PS=$(sha256sum "$P" | cut -d' ' -f1); CS=$(sha256sum "$C" | cut -d' ' -f1); OUT=<new empty dir>
$PY fit_th1_imagenet_v5.py --prefix4 "$P" --prefix4-sha256 "$PS" --clip4 "$C" --clip4-sha256 "$CS" \
    --output "$OUT/XP_N_V5_DRAFT.json" --point-reps 250 --bootstrap 60 --boot-reps 3
$PY check_th1_imagenet_v5_contract.py "$OUT/XP_N_V5_DRAFT.json"
$PY seal_th1_imagenet_v5.py --draft "$OUT/XP_N_V5_DRAFT.json" --output "$OUT/TH1_PREDICTIONS_v5_XP_N.json" \
    --code-commit 81003cc5a9167def74a04281501b4eae2837f44c --prefix4 "$P" --prefix4-sha256 "$PS" --clip4 "$C" --clip4-sha256 "$CS"
sha256sum "$OUT/TH1_PREDICTIONS_v5_XP_N.json"
```
Expect about a minute: the planner's native run on D1 inputs took 41 s and printed `XP_N_V5_DRAFT_COUNTS 100 20 4`.

**(c) Registration by push.**
- Commit to `returns/XP_N_V5_<UTC>/`:
  - `TH1_PREDICTIONS_v5_XP_N.json` and its SHA-256;
  - the draft;
  - a receipt with the UTC time, Python, NumPy and SciPy versions, the five code SHAs and the two input SHAs.
- Push **before scoring any run with index ≥ 4**. That push registers v5; I record it in the registry afterwards and reproduce it.

**(d) Failure.**
- **Allowed fixes:** if `fit` or `check` raises, you may fix the input formatting or the Python environment and retry once.
- **Never:** change the code, the seed or the parameters.
- **If it still fails:** commit `XP_N_V5_DECLINED_PRODUCTION_FAILED` with the error text. v5 is then declined and you may unseal.

**(e) Then unseal runs 4–63 and continue:** protocol, then CP-N, as in round 12 §3.
- If neither (c) nor (d) has happened by **Nov 4, 12:00 ET**, v5 lapses and runs 4–63 unseal.
- Adapt your autopilot's unseal trigger to (c)/(d). That is an infrastructure change and your call.

## 3. XP-I per-particle table: one column set missing (my spec error)

- **Reproduced exactly from `A16_XPI_PER_PARTICLE_METRICS.csv`:**
  - native, PSNR / SSIM / LPIPS: 28.0712 / 0.7971 / 0.1884;
  - E0, same metrics: 28.5615 / 0.8064 / 0.1788.
- **What's missing.** E1 needs the metrics of the **face-chosen orientation**, and the table has raw orientation only. 7 E0 picks are rotated by the test (3 `O_g`, 4 `I`): `S1_b22_62739`, `S1_b22_67293`, `S1_b22_69970`, `S2_b24_dev80_07985`, `S2_b24_dev80_21603`, `S2_b24_dev80_38516`, `S2_b24_dev80_56397`.
- **Please add** `PSNR_chosen, SSIM_chosen, VGG_LPIPS_chosen` (metrics of the particle after its `face_choice`) for all 720 rows, evaluation only. **Due Nov 4, 12:00 ET**, with the other CPU items.

## 4. Unchanged

- **Queue order:** Nov 4 CPU items → XP-N → XP-I2 → XP-C3 → E1 columns for Tab. 3 → XP-K → G7 rest and CP4 (Nov 15).
- XP-N cap: 20 GPU-h.
- No protocol v2; no matched-compute claim.
- Stop/ask only for the A13.4 conditions.
