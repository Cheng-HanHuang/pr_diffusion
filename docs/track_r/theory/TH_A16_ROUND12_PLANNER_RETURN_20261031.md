# Track R theory — Round 12 return, 2026-10-31

## Provenance

Verified **all 11 entries** in SHA256SUMS.txt from theory_upload_20261031_PM.zip before opening 12_PLANNER_GUIDANCE_20261031_PM.md, its integration diffs, or A16_PLANNER_EXPLORATORY_SUMMARY.md. No XP-N pilot or later outcome data have been accessed; no `xp/returns` access. The CP3/XP-D4 findings used below are already released, explicitly post hoc.

## 1. Upstream S1/S2/S3 integration edits: DONE

- docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex: math-mode FLAG_C paragraph title, anonymized post-hoc wording, every 'equal-compute' replaced by **'fixed K at equal mean runs'** or equivalent (not matched NFE or wall time). H-G1 10.97% discrepancy is relative to **linearly scaled reference**; relative to the newly labeled threshold at sigma=.02 it is about 9.9%. An explicitly post-hoc label-free tau paragraph explains why the four-run residual-minimum median can follow the labeled GOOD residual quantile without sigma-linearity, provided GOOD dominates minimum ranks. Current own/cross-fitted data are descriptive, not a new prior.
- docs/track_r/theory/TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex: changed planner-confirmed to source-specific phrase; removed rendered internal-code filename; commented internal unresolved-figure note. All underlying symmetry propositions and stale empirical placeholders remain untouched.
- docs/track_r/theory/TH_SUPPL_S3_COMPLETE_DRAFT_20261031.tex and separately the dated A15 addendum: two literal backslash-n sequences replaced by actual newlines; replaced planner attribution in rendered text with 'We also report'. All outcome counts, classical 16-moment witness and earlier registered claims preserved.
- Double-blind check: occurrences of 'planner' and internal paths in the rendered S1/S2/S3 are removed; remaining occurrences are LaTeX comments (not typeset). The original S2 relocated symmetry source is preserved.

These adopted edits are **theory upstream text**, not a new planner PDF build. The planner reported its previous integration as 15 pages, clean; re-run the integrated build after copying these updated files.

## 2. Image-disjoint ImageNet XP-N v5 code-only freeze: DONE before receiving any XP-N run

- **Code source commit**: 81003cc5a9167def74a04281501b4eae2837f44c. Full source SHA-256 manifest and gate: docs/track_r/theory/TH_XPN_V5_CODE_FREEZE_20261031.md (committed at 3ad5f2d950bdfcfeef13718bbedeb5638c172d68).
- Implementation: scripts/track_r/theory/fit_th1_imagenet_v5.py, check_th1_imagenet_v5_contract.py, seal_th1_imagenet_v5.py, run_th1_imagenet_v5.sh and test_th1_imagenet_v5_synthetic.py.
- Input: exactly 100 arbitrary image IDs, four FAST Good20 labels/residual ranks per image and 400 per-run G1 decisions, both input SHA-256 checked; **no D1 ID assertions**, no XP-N held indices 4–63 permitted.
- Refit the **v2 five-class Dirichlet and class-score race** using only the new authorized four-run prefix. No structural-zero spike. Variants race (primary), S2, tail stress, race_tau1.
- Targets one 64-run completed panel, K4/8/16/32/64, E0/E1_CLIP/O_raw/O_orbit. Expected contract: 100 predictions (E1 class and separate image-cluster sensitivity), 20 anatomy records, 4 exact-oracle-anchor audit groups. Require oracle MC/exact shifts <.01 and paired ordering E0<=O_raw and E1<=O_orbit. Same 95% model-conditional finite-panel predictive interval coverage rule as v2.
- Python syntax and bash -n checks **PASS**. Synthetic 100 arbitrary IDs and 400 fabricated G1 flags **PASS 100/20/4**, and an illegal future index 4 correctly rejected. Local smoke used **simplified stub fits** and therefore is **not native production QA**. The planner must run unchanged code in the actual repo with genuine 0–3 inputs and preregister the sealed result before any 4–63 outcome is opened. If the Nov 2 noon ET production gate fails, decline v5 instead of adjusting after unblinding.
- **No v5 numerical forecast is sealed/registered yet.**

## 3. XP-N pre-score mechanism prediction: FROZEN

Before any XP-N run was supplied, theory commit **a1ea26b0bd153fffec64a26ef730bd0aeb84d8a0** recorded docs/track_r/theory/TH_XPN_PROTOCOL_PRESCORE_20261031.md.
- **N-H1a (Nmax16 gain >=-1 point): predict PASS**, rough +0.2 pp central (judgment range [-1.8,+2.2]).
- **N-H1b (Nmax64 gain < -1 point): predict PASS**, rough -2.7 pp central (range [-5.5,+0.7]).
- This is a low-confidence S1 mixture argument: a small near-zero-pass tail can absorb a linearly growing fraction of the run budget while stopped easy-image quality saturates; no resampling, no fitting to the new panel, no claim of true structural zeros.

## 4. Remaining tasks

Final S1/S2/S3 integration by **Nov 6** with the latest source. S2's stale XP-C2/generic-detector/quote sentences remain unresolved by design. The planner owns paper wrappers, external M- references, 15-page PDF build and any later S4+ additions. No protocol v2 or paper main-text rewrite proposed.
