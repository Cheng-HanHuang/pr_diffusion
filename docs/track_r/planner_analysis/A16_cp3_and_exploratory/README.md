# A16: CP3 reproduction and post-CP3 exploratory analyses (planner, 2026-10-31)

Everything here is **CPU-only and re-run from committed inputs**. Ground truth enters only through the evaluation labels. Except for the CP3 reproduction, every analysis is **exploratory**: designed after CP2/CP3 outcomes were seen, run on panels already seen, and never used to change a frozen rule.

**Inputs.**
- XP CP2 return (`ef1cf10`): `D4_64RUN_STRINGS.csv` (SHA-256 `7f162733…`), `D4_64RUN_G1_CLIP.csv` (`cbf2a1cc…`).
- XP CP3 return (`1981dea`, `returns/XP_A17_CP3_20261031T173255Z/`): all files verified against its `SHA256SUMS`.
- F8 and B16 strings and face-CLIP decisions (CP1, `29ef1f5`).

**Code.**
- The frozen protocol `gtfree_protocol.py` at `c598176` (SHA-256 `d5f973fe…`).
- The three short scripts in `code/`.

## Results

| File | What it is | Label |
|---|---|---|
| `out/CP3_REPRODUCTION.json` | All 7 per-cohort CP3 protocol outputs (G2 σ = 0.02 and 0.10, G4, G5) rerun with the frozen code. The maximum absolute difference from the executor's JSONs is **0** in every field. | Confirmatory reproduction |
| `out/PC_DAPS_IMAGE_LEVEL.json` | XP-I GOOD25 shares reproduced from strings and face-CLIP decisions (native 0.7903, E0 0.8278, E1 0.8444; paired bootstrap on E1 − native, 20,000 reps: [0.026, 0.085] against the executor's [0.028, 0.085], a difference within Monte Carlo error). Image-level split: 135 images with all 4 particles GOOD, 27 with none GOOD in any orientation, 18 mixed. **E1 picks a GOOD particle in 17 of the 18 mixed images**, against 7.25 for a random particle; E1 reaches 84.4% against 85.0% for the 4-particle orbit oracle. | Descriptive (A15.3a) |
| `out/ORACLE_GAP_DECOMPOSITION.json` | O_orbit − E1 split into **selection loss** (the residual pick is C or I although an orbit-GOOD run exists) and **orientation loss** (the pick is orbit-GOOD but the output is not GOOD). Exact all-subset weights. | Exploratory diagnostic |
| `out/D4_BUDGET_FRONTIER.json` | Frozen protocol v1 on the 64-run ImageNet panel with N_MAX ∈ {2, …, 64}, runs drawn without replacement from the 64. | Exploratory frontier (allowed by A15.6) |
| `out/D4_CONFIDENCE_STOP_REPLAY.txt` | Reviewer-suggested variant: stop only when 2 runs pass **and** the pick's \|CLIP margin\| ≥ c. c = 0.00763, the FLAG_O cut of the D1-P2 primary panel, fixed in advance and not tuned. With c = 0 the replay reproduces the frozen protocol exactly. | Exploratory, one pre-fixed setting |
| `out/LABEL_FREE_TAU_CROSSFIT.json` | Label-free `self_mad3` threshold at σ = 0.02 and 0.10 (XP-G), and on G4 and G5. Computed both on the cohort itself and **cross-fitted** (from the other cohort, or the other half of G5 by SHA-256 parity of the image id). | Exploratory |

## Headline numbers

| Panel (K) | O_orbit | E1 | Gap | Selection | Orientation | of which partial orbit |
|---|---|---|---|---|---|---|
| DAPS FFHQ F8 (8) | 94.4 | 93.9 | 0.6 | 0.0 | 0.6 | 0.6 |
| SITCOM FFHQ B16 (16) | 92.2 | 91.1 | 1.1 | 1.1 | 0.0 | 0.0 |
| DAPS-4K FFHQ G4 (8) | 96.7 | 95.6 | 1.1 | 1.1 | 0.0 | 0.0 |
| PC-DAPS FFHQ (4 particles) | 85.0 | 84.4 | 0.6 | 0.6 | 0.0 | 0.0 |
| DAPS-4K ImageNet G5 (8) | 82.0 | 77.0 | 5.0 | 1.0 | 4.0 (3.0 good picks broken, 1.0 rotations unfixed) | 0.0 |
| FAST ImageNet XP-D4 (16) | 71.9 | 60.6 | 11.3 | 6.6 (6.5 from C) | 4.8 (3.1 unfixed, 1.5 broken) | 0.2 |
| FAST ImageNet XP-D4 (64) | 78.0 | 65.0 | 13.0 | 8.0 (all C) | 5.0 (4.0 unfixed, 1.0 broken) | 0.0 |

- **Budget frontier (XP-D4, 64 runs).**
  - The gain against fixed best-of-K at equal mean runs is +0.5 to +0.8 points at N_MAX = 3–6 and +0.1 at 8.
  - It is −0.5 at 12, −0.9 at 16, −2.3 at 32 and −3.4 at 64.
  - The protocol's success rises only from 44.0% to 49.2% between N_MAX = 2 and 64, against 44.1% to 65.0% for fixed best-of-K at K = N_MAX.
- **Confidence-conditioned stopping is worse** at equal mean runs:
  - −0.2 at N_MAX = 8;
  - −1.8 at 16;
  - −4.1 at 32;
  - −6.5 at 64.

  It spends extra runs on low-margin images, which rarely turn into confident GOOD outputs. **Abandoned, without tuning.**
- **Label-free τ.** σ-scaling failed (H-G1), but the label-free rule does not need scaling.
  - **Distance from the labeled τ:** within 0.1–3.5% at σ = 0.02 and within 0.6–1.2% at σ = 0.10.
  - **Protocol gain** (pooled; own threshold / cross-fitted threshold):

    | Panel | Own | Cross-fitted | Mean runs |
    |---|---|---|---|
    | σ = 0.02 | +3.3 | +4.1 | 3.2–3.3 |
    | σ = 0.10 | +1.1 | +0.9 | 2.2–2.3 |
    | G4 | +3.9 | +3.7 | |
    | G5 | +0.1 | +0.4 | |
