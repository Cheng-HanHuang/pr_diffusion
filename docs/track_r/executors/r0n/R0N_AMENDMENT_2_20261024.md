# R0-N amendment 2 (planner, 2026-10-24): the R0-N.2 follow-up CPU block

**Status.**
- The R0-N N1 return has been received and ruled on. It is `R0N_20261024T061111Z` on `codex/prdiff-track-r-r0n-20261023` @ `5af47db`; the planner's ruling is in `docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md` §9.
- This amendment defines **one follow-up CPU block, R0-N.2**. It collects everything still owed:
  - items from amendment 1 that were not executed: the R0-N run merged planner `6686322`, which predates amendment 1;
  - N2 on CPU, because the GPU gate returned UNAVAILABLE;
  - two new CPU items prompted by concurrent work PC-DAPS.

**Scope.**
- CPU only (`CUDA_VISIBLE_DEVICES=""`).
- Changes no frozen R0 or R0-N v1 definition.
- **Excluded:** GPU work, new reconstructions or measurements, and any access to confirmation305.
- Existing R0-N outputs are immutable. Write to a new `r0n2_<stage>_<UTCSTAMP>/` root.

**Priority order.** If time runs short, finish B1–B4 and return; A4/A5 may be marked "deferred". Optional items not returned by 2026-11-06 are dropped from the submission.

**Planner context for A4/A5:** `docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`.

## B1 — branch

1. Merge `origin/claude/prdiff-track-r-planner-20261021` (≥ the commit that adds this file) into `codex/prdiff-track-r-r0n-20261023`, or into a new `codex/prdiff-track-r-r0n2-<YYYYMMDD>` created from `5af47db`. This should be conflict-free, because planner-owned files are unchanged on the executor branch.
2. Record both SHAs in the return.

## B2 — amendment 1 items not yet executed (required)

1. **N1h, training-membership lookup.** Run amendment 1 §A2 exactly:
   - look up the FFHQ `category` by id only, for B22 and DEV80;
   - report repair success and harm for prior S and S_mid, split by category;
   - record any documentation of `ffhq_10m.pt`'s training set, from targeted greps of `external/dps` and `external/DiffFPR`; otherwise write "UNKNOWN".
2. **A3 image identities.** List the image ids behind:
   - each post-selection fix;
   - each selection miss;
   - each uncorrected orbit error;
   - each generation failure

   in the max-K N1b strata. These are the 15 fixes, 5 misses, 1 uncorrected and 93 generation failures counted by the planner from `N1b_POSTSELECTION.csv`.
3. **N0 receipt.** `R0_7_COMPLETE_RECEIPT.json` (SHA `50ff1550…79c8`) was not committed with N0.
   - If it is path-free, commit it and verify the SHA.
   - If not, record "withheld: contains paths".

## B3 — N2 CLIP zero-shot on CPU (required: kill criterion K2)

1. Run `scripts/track_r/r0n_2_clip_orientation.py` on CPU. Keep the frozen config `n2_clip_zero_shot` unchanged (`cpu_allowed: true`): same model SHA, prompts, preprocessing, score and decision.
2. No GT in scoring.
3. Measure CPU seconds per encoding.
4. **Report** success on GLOBAL orbit errors, harm on GOOD, and the partial orbit errors left unfixed, for:
   - CLIP;
   - prior global-only (S);
   - prior global-only (S_mid);
   - **the N1d-selected configuration** (`single_300|J1|global2`, 2 forwards);
   - UnivFD global-only.

   Give image-cluster CIs. Add measured or projected cost per candidate for each.
5. **Stop rule.** If the CPU projection exceeds 3 hours wall time, stop and report the projection instead of running.

## B4 — N1d miss identity (required, one line)

Give the id and orbit pattern of the one DEV80 orbit error that `single_300|J1|global2` missed (35/36). The planner expects it to be the single partial-orbit error (DAPS, DEV80); confirm or correct.

## A4 — N1i: outcome crosswalk and protocol ladder (optional, non-blocking)

**Why.**
- PC-DAPS (arXiv 2610.31592; code `github.com/voilalab/PC-DAPS` @ `59b2f04`, MIT) reports DAPS phase-retrieval outcomes on FFHQ with an orientation-only label: 14.5% of runs are upside-down or color-inverted.
- We report 3.5% (B22) and 5.3% (DEV80) of DAPS runs as *repairable* orbit errors.
- To reconcile these, compute their label on our runs.

**Inputs.** Use the same candidate images and GT tensors that the R0.7 PSNR_raw replay used, per source.
- Record the representation each source uses: float after clamp, or RGB8.
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

**Planner reference values** (R0.8 P4 / N1g; use them to sanity-check your ladder, not as targets):

| Stratum | First run, K=1 E0 (dB) | O_raw at max K (dB) | E1 at max K (dB) |
|---|---|---|---|
| B22 DAPS | 27.23 | 29.42 (K=2) | 30.07 |
| B22 SITCOM | 22.97 | 26.66 (K=4) | 26.64 |
| B22 NP | 25.58 | 29.40 (K=8) | 29.12 |
| DEV80 DAPS | 23.29 | 28.83 | 28.82 |
| DEV80 SITCOM | 19.59 | 24.12 | 24.70 |
| DEV80 NP | 22.32 | 26.07 | 25.86 |
| FAST (S5, both cohorts) | 23.81 | 29.00 | 28.99 |

### A4.4 DAPS configuration record (feeds N1e)

From the historical receipts of S1, S2 and S3, record for DAPS:
- annealing steps;
- ODE steps;
- NFE per run.

The planner reads the B21/B24 launch defaults as 400 annealing × 5 ODE steps, i.e. 2,000 NFE per run. Confirm or correct this from receipts, not from the scripts.

### A4.5 Outputs and labeling

- **Status.** Every N1i number is DESCRIPTIVE or EXPLORATORY; no decision depends on it.
- **Evidence file.** Write `R0N2_LETTER_EVIDENCE.json` with the numbers under `n1i_*`.
- **Tables.** One CSV per table.
- **Figure (optional).** A small PDF/PNG of the cross-tab as a stacked bar for DAPS on B22 and DEV80.

## A5 — N3b: bounded held-out-data inventory (optional, non-blocking)

Record presence and size only. No enumeration inside directories, no downloads.

**Checks:**
1. Held-out face data, depth 1:
   ```
   ls -d /egr/research-pac/huang248/data/celeba* /egr/research-pac/huang248/data/*CelebA* /egr/research-pac/huang248/data/*face* 2>/dev/null
   ```
2. A DAPS-style 100-image ImageNet test subset, using `test -d` only:
   - `dataset/test-imagenet` or `datasets/test-imagenet` under the DAPS checkout(s) recorded in R0.0;
   - the same paths under `/egr/research-pac/huang248/external/FAST_DIPS_A15`.
3. ImageNet validation images by fixed file name, without listing the directory. N3 found `/egr/research-pac/huang248/data/imagenet`. Only test a val directory directly below it: `val` or `ILSVRC2012_val`, using `test -d`.
   - **Flat layout:** `test -f <val dir>/ILSVRC2012_val_00000001.JPEG` and `…00000100.JPEG`.
   - **Per-class layout** (detected with `test -d <val dir>/n01440764`): record "per-class layout" and stop. Do not search the class folders.

**Output.** Add the results to the return. Each item is PRESENT (with size) or ABSENT.

## A6 — explicitly not authorized here

**N4** (clean-image orientation probe) is a planner *proposal* (planner note §9) that needs Cheng's explicit go.
- Do not write or run GPU code for it under this amendment.
- If Cheng forwards a go, the planner will issue amendment 3 with the frozen N4 spec.

## Return (`R0N2_EXECUTOR_RETURN.md`)

1. **Status for Cheng** — at most 8 lines.
2. **Identity** — branch, SHAs and interpreter.
3. **Sections:** B2 (N1h, A3 identities, N0 receipt), B3 (N2 / K2 inputs), B4, A4 (N1i), A5 (N3b).
4. **Anomalies.**
5. **Artifacts** — PAC roots, committed outputs and SHA256SUMS.
