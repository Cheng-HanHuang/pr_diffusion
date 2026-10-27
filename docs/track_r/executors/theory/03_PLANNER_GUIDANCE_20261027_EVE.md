# Planner guidance to the theory executor, round 3 (2026-10-27, 17:15 ET)

**Responds to:** `TH_A6_PLANNER_RESPONSE_20261027.md`, head `1e3e8d9`.

**Governing amendments:** XP prereg **A6** and **A7**.

## 1. Your inputs exist and are verified

Cheng will upload these files to you, since your runtime cannot read the raw files from GitHub. Check each SHA-256 on arrival.

| File (XP branch) | Commit | SHA-256 |
|---|---|---|
| `docs/track_r/executors/r0n/returns/R0N2_A8B_20261027T041156Z/R0N2_A8B_PER_IMAGE.csv` | `c638b54` (2026-10-27 12:01 ET) | `e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901` |
| `…/R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv` | `c638b54` | `a86d625ec574614ad42131ae648b84beefe7ce8cb340516ce6a2ae65f20db9f5` |
| `docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv` | `41d4121` (2026-10-27 12:01:51 ET) | `c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1` |

**A8b arrived before your Oct 28, 12:00 ET gate, so the gate is satisfied.**

**Planner QA of A8b:**
- The A8 columns are identical, row by row, to A8 (SHA `197d0c8e…`).
- The ρ order agrees with `e0_rank_order` in all 760 rows. 19 rows tie at 6 significant digits, so take the order from `e0_rank_order`.
- Every C and I label agrees with its frozen τ, across all 3,240 runs.
- τ by family:

  | Family | Cohort A (B22) | Cohort B (DEV80) | ρ★ |
  |---|---|---|---|
  | DAPS | 0.0474649 | 0.0475986 | ≈ 0.05 |
  | SITCOM | 0.0448503 | 0.0449706 | ≈ 0.05 |
  | FAST | 0.0500829 (both cohorts) | | ≈ 0.05 |

  So τ < ρ★ ≈ σ for DAPS and SITCOM; for FAST, τ ≈ σ.
- Within-image gaps (ρ_C − ρ_G)/σ over G/C pairs:

  | Family | Pairs | Median | Minimum | Share negative |
  |---|---|---|---|---|
  | SITCOM | 158 | 6.8e-3 | −2.4e-4 | 3.2% |
  | DAPS | 21 | 3.3e-3 | +5.5e-4 | 0 |
  | FAST | 26 | 2.2e-3 | +8.2e-5 | 0 |

  These gaps are tiny relative to σ, yet decisive in the paired comparison, as your Proposition 3 predicts.

## 2. Mechanism forecasts v1.1: registered

- `5abad48`, SHA-256 `11262301…758d`, committed 15:56:47Z; registry row added.
- A5.5 (the historical decomposition) may now be shared with you once the executor produces it.
- **Note for interpretation:** the signs of Δ and C for DAPS and SITCOM GOOD runs follow already from N1c (ρ < σ ⇒ Δ < 0 ⇒ C > D/2). The informative forecasts are the ratios D/C ≈ 1.02 and 1.04, and FAST's sign.
- **Expect** the flat model to miss D/C. A GOOD run at about 28 dB still differs from x★ by much more than the fitted noise, so D contains a reconstruction-error term that the model omits. A miss there is informative, not embarrassing. Say so in the paper text.

## 3. Production run of your A6 code on the raw A8 (planner, head `1e3e8d9`)

- `fit_th1_prefix_v1.py` (without `--freeze`): **runs** in 3 min 17 s and writes 976 prediction rows.
- `check_th1_exact_prefix_oracles.py`: **PASS** (5 cases).
- `check_a6_panel_contract.py`: **FAIL.** 5 rows have E1_CLIP > O_orbit, all DAPS cohort A, by at most 4.4e-4:

  | Variant | K |
  |---|---|
  | race | 6 |
  | race_tau1 | 4, 6, 8 |
  | tail | 4 |

  One anatomy row has a negative selection loss (−6.5e-5).
- **Cause.** The oracle points are now exact beta-hypergeometric means, while E0 and E1 are still Monte Carlo means. Where E1 ≈ O_orbit (DAPS cohort A), the Monte Carlo error crosses the exact value. The degenerate selection intervals (lo = hi = point) have the same cause.
- **Fix: control variates.** Set E = exact O − Monte Carlo mean of the per-panel loss (O − E), and shift each E draw by the same Δ as its oracle.
  - This is unbiased and has lower variance.
  - It makes E0 ≤ O_raw and E1 ≤ O_orbit hold exactly, both for the points and for every draw.
  - With this patch the planner reran everything: `TH1_A6_PANEL_CONTRACT_PASS predictions 976 anatomy 252 crossovers 60`.

```diff
@@ -277,12 +277,19 @@
                     point[:,2:]=oracle
+                    delta_raw=oracle[:,0]-simulation_mean[:,2]
+                    delta_orb=oracle[:,1]-simulation_mean[:,3]
+                    point[:,0]=simulation_mean[:,0]+delta_raw
+                    point[:,1]=simulation_mean[:,1]+delta_orb
@@
                     draws[:,:,2:]+=(oracle-simulation_mean[:,2:])[None,:,:]
+                    draws[:,:,0]+=delta_raw[None,:]
+                    draws[:,:,1]+=delta_orb[None,:]
```

Apply it (or an equivalent fix of your own), rerun the validator on the raw A8, and record the PASS in the freeze.

## 4. Before the v1 freeze (target Oct 28, 18:00 ET; hard Oct 31)

1. **Apply the fix above.**
2. **A8b.** Either use it, or state why not.
   - Minimum: QA only (as in §1).
   - Optional: a gap-calibrated tail variant, as an additional sensitivity row and not the primary. For example, a per-family lower tail of within-image (ρ_C − ρ_G) fitted from A8b.
   - Do not change the primary model at the last minute.
3. **Recommended: per-image point forecasts in a companion file.** List per image, variant, panel and K the values of E0, E1_CLIP, O_raw and O_orbit. Then, if a B22 SITCOM image fails the B16 lineage gate (A7.2), the comparison can drop it without refitting. Commit it with the freeze and include its SHA-256.
4. **Freeze and report.** Commit `TH1_PREDICTIONS_v1.json`, plus the companion file if you make it, and send Cheng:
   - the commit SHA;
   - the SHA-256 of each file;
   - the UTC commit time.

## 5. v2: ImageNet FAST

- **Due:** 48 h after `41d4121`, i.e. **Oct 29, 12:01 ET** (A7.4).
- **Inputs:** the D1 phase-1 strings above (100 images × runs 0–3; Good20; τ_IN = 0.050183).
- **Panel:** n = 16, K ∈ {8, 12, 16} (the 64-run panel only if XP-D4 runs).
- **CLIP G1 decisions:** pending (XP-C worker). If they have not reached `theory_inputs/`, freeze E0, O_raw and O_orbit, and for E1_CLIP either abstain or state your φ_g and h assumption.

  ImageNet orientation is less canonical than FFHQ's, so do not assume FFHQ's 58/58.

## 6. Then

- A5.5, when released: compare against v1.1, and explain any miss through the reconstruction-error term.
- Integration text for Secs. 4–5 and the supplement proofs, by Nov 3–6.
- **Stop condition:** if anything needs `xp/returns` or K > 4 data, stop and ask.
