# A38 — G7 ground-truth upright-retention closeout for CP4

**Executor return, 2026-11-04.** Governed by planner round 19 / prereg A23.5 @ `6f5ee7f9c7a952472ace86cb3d3c301c92937d08`. **Ready for planner verification, not yet planner-accepted.** G7 is a GT-only diagnostic, not a new selector, inverse-solver benchmark, matched-compute comparison, or new main-paper result.

## Source and completed results

All four remotely published per-image CSVs have **exactly 280 rows**, with no duplicate indices. An independent cross-file check found identical `(index, dataset, basename, uid, ground-truth source SHA256)` for every row. Original frozen GT image manifest SHA: `d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde`. Cohort: 180 FFHQ + 100 ImageNet.

| Frozen policy, in A23 order | FFHQ upright | ImageNet upright | Pooled upright | Immutable publication commit |
| --- | ---: | ---: | ---: | --- |
| S_mid global2 | 180/180 | 95/100 | 275/280 (98.21%) | [`da130b5`](https://github.com/Cheng-HanHuang/pr_diffusion/commit/da130b50856fd6c495fe73049651b746ab67b56b) |
| S_mid all8 | 180/180 | 95/100 | 275/280 (98.21%) | [`967e73c`](https://github.com/Cheng-HanHuang/pr_diffusion/commit/967e73cc7adcc5cc3347d257572bd779ff0a6dea) |
| 2-forward | 180/180 | 93/100 | 273/280 (97.50%) | [`6d2afbf`](https://github.com/Cheng-HanHuang/pr_diffusion/commit/6d2afbfd8c405cbf7db4a6f193ba21c885c7eaa4) |
| UnivFD | 172/180 | 89/100 | 261/280 (93.21%) | [`9d379d2`](https://github.com/Cheng-HanHuang/pr_diffusion/commit/9d379d2f275a9feeb175e1f18800fe2568609db2) |

All four method readouts and stage-terminal files report complete 280-image grids, and each publication preserves its aggregate/per-image CSV SHA256. The final UnivFD publication was verified by pushing the **existing** local commit after original push timeouts; no images were recomputed. This return is a consolidation, not a fifth experiment.

### Paired-image descriptive checks

- S_mid global2 and S_mid all8 chose **exactly the same orbit for all 280 GT images**, including the same five ImageNet global flips at indices `197,248,258,261,271`.
- 2-forward flipped seven ImageNet images: `189,197,225,248,258,261,274`. Four overlap the five S_mid flips; one S_mid flip is not selected by 2-forward.
- UnivFD changed the upright decision on 19 images total (eight FFHQ, eleven ImageNet). Its selected orbits are `000` for 261 images, `111` for 18, and `010` for one. Two of its 19 non-upright decisions overlap the S_mid five (`258,261`).
- These paired comparisons are **descriptive, not a preregistered hypothesis test**. Better GT upright retention does not, by itself, establish better selection of inverse-problem reconstructions.

## Budget and provenance

| Ledger event | Added physical GPU-h | G7 cumulative GPU-h | Project cumulative GPU-h |
| --- | ---: | ---: | ---: |
| Previously accepted A22 G7 C2 | 2.626529895 | 2.626529895 | 244.457094760 |
| A33 B2/B8 paired canary | 0.017988794 | 2.644518689 | 244.475083555 |
| A34 S_mid global2 remaining GT evaluation | 2.512508263 | 5.157026952 | 246.987591818 |
| A35 S_mid all8 | 0.744200808 | 5.901227760 | 247.731792625 |
| A35 2-forward | 0.070124751 | 5.971352510 | 247.801917376 |
| A36/A37 UnivFD on CPU | 0.000000000 | 5.971352510 | 247.801917376 |

**Final charged G7:** `5.971352510419156 / 10.0 GPU-h` (remaining `4.028647489580844`). **Final charged project:** `247.80191737622505 / 400.0 GPU-h` (remaining `152.19808262377495`). These are G7/project cumulative ledger values from verified stage receipts, not a claim that all other simultaneous project activities outside these receipts have been globally reconciled.

The A33 pilot confirmed paired original A20 versus A21 fixed-t B8 loss differences below `1e-5`; the A35 native FFHQ and ImageNet canaries passed their corresponding numerical gates. No A21.10 optional replicate was run.

## UnivFD dependency-only recovery

The original A35 CPU worker stopped before evaluating an image because external pinned UniversalFakeDetect CLIP required `pkg_resources`. A36 isolated official `setuptools==80.9.0` then found another missing dependency `ftfy`. A37 isolated pinned `ftfy==6.3.1` and `wcwidth==0.2.13`; synthetic eight-orbit CPU model inference passed. The **original unchanged A35 UnivFD worker/scorer/weights** subsequently generated all 280 GT image receipts in the existing A36 recovery run, and the A35 publisher created local commit `9d379d2`. Its first two pushes timed out; the operator verified science/readout and pushed the exact same commit, with GitHub exact-head confirmation. New GPU time: zero. No earlier accepted GT policies rerun.

## CP4 remainder / limitations

The A23 G7 upright-retention requirement is complete **as an executor return, pending planner acceptance**. CP4 also requests XP-K against registered mechanism forecasts, the final broader ledger, and an explicit unrun inventory. XP-K (SITCOM, FAST-DIPS, DAPS) has separate native source one-parameter, historical identity replay, and DAPS F8 ordering gates; **no XP-K outcome is established by this G7 report**. The optional A21.10 replicate was not run; protocol v2/new selector and matched-compute claims are not authorized.

**Recommended planner action:** verify the four publication SHA receipts, accept/reject the paired 280-image G7 GT table, and keep XP-K/remainder ledger as separate CP4 items. Do not rerun any accepted G7 generation.
