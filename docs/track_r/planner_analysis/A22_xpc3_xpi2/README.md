# A22: XP-C3 readout and XP-I2 reproduction (planner, 2026-11-03)

CPU-only, from committed inputs. Ground truth enters only through the evaluation labels and metrics.

## Inputs (SHA-256)

| Input | Source | SHA-256 |
|---|---|---|
| XP-N prior decisions (6,400) | XP `91fe7a3`, `theory_inputs/XP_C3_A20_XP_N_6400_PRIOR_DECISIONS.csv` | `fc77b073…` |
| G5 prior decisions (800) | XP `a6c6a64`, `theory_inputs/XP_C3_A20_G5_800_PRIOR_DECISIONS.csv` | `ab20baf1…` |
| XP-D4 prior decisions (6,400) | XP `591a28f`, `theory_inputs/XP_C3_A20_XP_D4_6400_PRIOR_DECISIONS.csv` | `65f2595d…` |
| XP-N strings, G1 decisions | XP `2d791c5`, `theory_inputs/XP_N_RUNS_0_63/` (match its SHA256SUMS) | `4b256e53…`, `f63c3d05…` |
| XP-D4 strings, G1 decisions | XP `ef1cf10`, `returns/XP_A15_CP2_20261030T185023Z/` | `7f162733…`, `cbf2a1cc…` |
| G5 strings, G1 decisions | XP `1981dea`, `returns/XP_A17_CP3_20261031T173255Z/` (match its SHA256SUMS) | `14201083…`, `ca5d0d7e…` |
| XP-I2 per-particle, policies, DAPS comparison, G1, strings | XP `1eae9bf`, `returns/XP_I2_A15_5_20261103T183150Z/` | `d05b390a…`, `9c872f1d…`, `a1a94a3b…`, `e082a5de…`, `32b9922c…` |

## 1. XP-C3: `code/xpc3_readout.py` → `out/XPC3_READOUT.json`

**Integrity.**
- Each panel has 100 images × n runs.
- The label in the prior file equals the frozen string label and the CLIP label. The D4 CLIP file's old `O`/`P` labels are mapped to `O_g`/`O_p`.
- `prior_choice` = 111 iff margin > 1e-7.
- `*_chosen_good` is GT-evaluated at the chosen orientation:
  - C and I candidates are never GOOD;
  - G kept upright and O_g flipped are always GOOD;
  - G flipped or O_p flipped can be GOOD on near-symmetric images;
  - wherever the two tests chose the same orientation, the outcomes agree.

**Readout.** Exact all-subset E0, E1-CLIP, E1-prior, O_raw and O_orbit. Paired image bootstrap of E1-prior − E1-CLIP (10,000 replicates, seed 20261103).

| Panel | Result |
|---|---|
| XP-N | Prior ahead at every K (+1.9 to +5.3): K = 4 +3.8 [+0.9, +7.0]; K = 16 +5.3 [+1.1, +9.8]; K = 64 +5.0 [−1.0, +11.0]. Orientation loss at K = 64: 7.0 (CLIP) vs 2.0 (prior) |
| XP-D4 | −0.6 to +1.0; every interval contains 0 |
| G5 | +0.75 to +1.44; every interval contains 0 |

**Candidate level, share of G runs flipped and of O_g runs fixed:**

| Panel | G flipped, prior | G flipped, CLIP | O_g fixed, prior | O_g fixed, CLIP |
|---|---|---|---|---|
| XP-N | 6.3% | 11.2% | 82.8% | 78.1% |
| XP-D4 | 10.4% | 8.2% | 85.2% | 86.8% |

**Margins.** The median |margin| is 3–9e-5 for G/O_g and about 1e-5 for C/I.

## 2. XP-I2: `code/xpi2_repro.py` → `out/XPI2_REPRODUCTION.json`

**Reproduction.**
- The policies are recomputed from the per-particle CSV and match the executor's to 3.6e-15.
- The DAPS-4K comparators, recomputed from the frozen G5 strings and G1 decisions, are identical per image.

**Outcome.** Native mean 30.75% / 17.40 dB; E0 37%; E1 53% / 21.50 dB; O_orbit 55%.
- E1 − native (paired, 20,000 replicates):
  - GOOD +22.25 [+14.8, +30.0];
  - PSNR +4.11 [+2.61, +5.64];
  - SSIM +0.110 [+0.067, +0.155];
  - LPIPS −0.091 [−0.126, −0.057].
- **Bag anatomy:** 55 bags hold an orbit-GOOD particle, 45 of them only orbit-GOOD particles. E1 is GOOD in 44 of those 45 and in 9 of the 10 mixed bags.
