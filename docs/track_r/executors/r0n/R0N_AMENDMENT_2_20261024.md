# R0-N amendment 2 (planner, 2026-10-24, after reading PC-DAPS)

**Scope.**
- Adds two CPU-only items to R0-N.
- Changes no frozen R0 or R0-N v1 definition.
- **Excluded:** GPU work, new reconstructions or measurements, and any access to confirmation305.

**Non-blocking.** If either item would delay the N1a–N1h return, return without it and write "N1i deferred" or "N3b deferred". Optional items not returned by 2026-11-06 are dropped from the submission.

**Why.**
- PC-DAPS (arXiv 2610.31592; code `github.com/voilalab/PC-DAPS` @ `59b2f04`, MIT) reports DAPS phase-retrieval outcomes on FFHQ with an orientation-only label: 14.5% of runs are upside-down or color-inverted.
- We report 3.5% (B22) and 5.3% (DEV80) of DAPS runs as *repairable* orbit errors.
- To make our tables comparable and to reconcile these numbers, compute their label on our runs.
- Planner context: `docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`.

## A4 — N1i: outcome crosswalk and protocol ladder (CPU; same PAC block as N1 if possible)

**Inputs.** Use the same candidate images and GT tensors that the R0.7 PSNR_raw replay used, per source.
- Record which representation each source uses: float after clamp, or RGB8.
- GT is used for evaluation labels only, as in R0.7. It is never used in a deployable step.

### A4.1 PC-DAPS outcome label

This is the exact rule of `pr_outcomes.py` @ `59b2f04`. You may reimplement it, or copy it with the MIT notice.

1. **Map to [0, 1].** Map x and g from [−1, 1] by v·0.5 + 0.5, then clamp to [0, 1].
2. **Per-channel PSNR.** PSNR_c(x, r) = 10·log10(1 / max(MSE_c, 1e-10)), with the MSE taken over H×W for channel c.
3. **Shift search.** For r ∈ {g, rot180(g)}, where rot180 flips both spatial axes, take the maximum of PSNR_c over the 9 circular shifts (dy, dx) ∈ {−1, 0, 1}² of r. The shift is `torch.roll` on the spatial dims.
4. **Channel assignment.**
   - upright_c ⇔ best(g) > best(rot) + 1.0 dB;
   - rotated_c ⇔ best(rot) > best(g) + 1.0 dB.
5. **Run label.**
   - **faithful:** all channels upright;
   - **flipped:** not faithful, and every channel is upright or rotated;
   - **failed:** otherwise.
6. **Unrecoverable image** at a given K: none of its K runs is faithful.

**Unit test (synthetic, CPU).** A smooth non-symmetric test image should be labeled faithful; its rot180, flipped; a version with only channel G rotated, flipped; and uniform noise, failed.

### A4.2 Tables (DESCRIPTIVE)

1. **Label rates.** For each family × cohort (and per source where a source spans both cohorts, i.e. S5): faithful / flipped / failed rates per run. Give n and image-cluster bootstrap CIs using the R0.7 convention.
2. **Cross-tab.** PC-DAPS label × R0.7 label, as counts per family × cohort. R0.7 labels: GOOD, ORBIT-global, ORBIT-partial, CONSISTENT, INCONSISTENT.
3. **Unrecoverable images.** The rate per image at maximal K, next to 1 − O_orbit Good25 for the same stratum.
4. **Selected outputs.** The same labels for the E0-selected and the E1-selected output per image at maximal K.

### A4.3 Protocol ladder

For each (source, family), at K = 1 and at maximal K, report Good25 / mean / q05 for:
- **mean-of-K:** each image's mean PSNR over its K runs, then the mean over images. Also give the per-run Good25 rate.
- **E0, E1, E2, O_raw, O_orbit:** the existing frozen policies, reproduced.
- **E_P (EXPLORATORY), prior-only selection:**
  1. Take the stored S of each candidate's identity member (000).
  2. Pick the argmin over the K candidates; ties go to the lowest run index.
  3. Apply the frozen E1 orbit repair to that pick.

  Also report E_P − E0 and E_P − E1. E_P is the post-hoc analogue of selecting by prior consistency alone, with no residual.

**Planner reference values** (the R0.8 P4 matrix; use them to sanity-check your ladder, not as targets):

| Stratum | First run, K=1 E0 (dB) | O_raw at max K (dB) | E1 at max K (dB) |
|---|---|---|---|
| B22 DAPS | 27.23 | 29.42 (K=2) | 30.07 |
| B22 SITCOM | 22.97 | 26.66 (K=4) | 26.64 |
| B22 NP | 25.58 | 29.40 (K=8) | 29.12 |
| DEV80 DAPS | 23.29 | 28.83 | 28.82 |
| DEV80 SITCOM | 19.59 | 24.12 | 24.70 |
| DEV80 NP | 22.32 | 26.07 | 25.86 |
| FAST (S5) | 23.81 | 29.00 | 28.99 |

### A4.4 DAPS configuration record (feeds N1e)

From the historical receipts of S1, S2 and S3, record for DAPS:
- annealing steps;
- ODE steps;
- NFE per run.

The planner reads the B21/B24 launch defaults as 400 annealing × 5 ODE steps, i.e. 2,000 NFE per run. Confirm or correct this from receipts, not from the scripts.

### A4.5 Outputs and labeling

- **Status.** Every N1i number is DESCRIPTIVE or EXPLORATORY; no decision depends on it.
- **Evidence file.** Add the numbers to `R0N_LETTER_EVIDENCE.json` under `n1i_*`.
- **Tables.** One CSV per table.
- **Figure (optional).** A small PDF/PNG of the cross-tab as a stacked bar for DAPS on B22 and DEV80.

## A5 — N3b: bounded held-out-data inventory (CPU)

Record presence and size only. No enumeration inside directories, no downloads.

**Checks:**
1. Held-out face data, depth 1:
   ```
   ls -d /egr/research-pac/huang248/data/celeba* /egr/research-pac/huang248/data/*CelebA* /egr/research-pac/huang248/data/*face* 2>/dev/null
   ```
2. A DAPS-style 100-image ImageNet test subset, using `test -d` only:
   - `dataset/test-imagenet` or `datasets/test-imagenet` under the DAPS checkout(s) recorded in R0.0;
   - the same paths under `/egr/research-pac/huang248/external/FAST_DIPS_A15`.
3. ImageNet validation images by fixed file name, without listing the directory. Only check a val directory under an `imagenet*` path that N3 already found at depth 1.
   - **Flat layout:** `test -f <val dir>/ILSVRC2012_val_00000001.JPEG` and `…00000100.JPEG`.
   - **Per-class layout** (detected with `test -d <val dir>/n01440764`): record "per-class layout" and stop. Do not search the class folders.

**Output.** Add the results to the N3 section of the return. Each item is PRESENT (with size) or ABSENT.

## A6 — explicitly not authorized here

**N4** (clean-image orientation probe) is a planner *proposal* (planner note §9) that needs Cheng's explicit go.
- Do not write or run GPU code for it under this amendment.
- If Cheng forwards a go, the planner will issue amendment 3 with the frozen N4 spec.

## Return additions

Add sections **N1i** and **N3b** to `R0N_EXECUTOR_RETURN.md` §4, and record the amendment-2 commit SHA in §2 (Identity).
