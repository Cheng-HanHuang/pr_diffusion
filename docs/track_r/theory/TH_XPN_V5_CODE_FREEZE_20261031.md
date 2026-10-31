# XP-N v5 prospectively frozen implementation — 2026-10-31

**Status: CODE FROZEN; prediction values NOT YET FITTED OR REGISTERED.** Frozen before this executor received any XP-N candidate runs, including calibration runs 0–3. The image-disjoint 100-image experiment remains fully held at this handoff. Version A16.5. No data from XP-N or `xp/returns` accessed.

The first complete v5 implementation (including synthetic test) exists by repository code commit **81003cc5a9167def74a04281501b4eae2837f44c**. The underlying algorithms and their sibling-fit imports are frozen here. The theory-owned supplement wording edits committed later do not change forecast code. Any future model-code change requires a dated erratum **before** the first XP-N pilot is analyzed, and a new code SHA/commit.

## SHA-256 of source files (exact bytes)

```text
5ac8811f0948ee143f0b2cd90406626a01bc9421121472fccb241292e2745292  scripts/track_r/theory/fit_th1_imagenet_v5.py
7313d46ddb5df33e2b3dfb7bae8c301e66584b0d503b6da117560e41bde4d33c  scripts/track_r/theory/check_th1_imagenet_v5_contract.py
6f70c8638eda541ce10786b905403a3ac82c2fe2eefef9bfcf25647819ff3f71  scripts/track_r/theory/seal_th1_imagenet_v5.py
e795ff2233f5d57cac6c4ec12649e5e31271c00fb864a74ff2b586088dcaec8e  scripts/track_r/theory/run_th1_imagenet_v5.sh
8fba9cf6ba1cdfdaea2f93933d6168490eec8dd58495e971eeeebd69095c25b6  scripts/track_r/theory/test_th1_imagenet_v5_synthetic.py
```

## Nonnegotiable v5 design

- Input **100 arbitrary image IDs** from the run-0–3 FAST Good20 ImageNet pilot; no D1 ID or frozen D1 cohort overlap/provenance assertion. Exactly four labels and residual ranks per image, hashed in production.
- Input 400 independent per-run G1 chosen-orientation Good20 decisions for those exact image IDs and indices 0–3 only; first four decisions kept as observed. No 4–63 labels or other held data may be read to produce a v5 forecast.
- **No structural-zero spike**, five-class Dirichlet generator fitted on the new 0–3 data; solver-class BT weights with race, S2, transferred C-tail, race_tau1 (all from existing v2 fitting code); G1 beta(1,1) per-class success rates fit only on 0–3.
- For **one 64-run completed panel** and uniform K-subset K=4/8/16/32/64, emit E0, E1_CLIP, O_raw, O_orbit on the 100 fixed image IDs. Predictive realized-panel 95% intervals; primary selection race; the same frozen coverage and standardized-residual tie-break used before.
- Exact Dirichlet finite-panel oracle anchor for O_raw/O_orbit shared across selection variants. Require each pair of MC and exact oracle calculations to disagree by **less than 0.01** at every K before sealing. Equal paired oracle/selector shifts preserve E0<=O_raw and E1<=O_orbit for every draw.
- **Contract:** 100 prediction records (four variants x five K x E0/E1 beta class/E1 cluster/O_raw/O_orbit), 20 generation/selection records, four five-K oracle-anchor groups. Abort on mismatched hashes, illegal indices, invalid labels/ranks, missing decisions, wrong IDs, oracle-anchor violation or failed interval contract.

## QA and registration gate

The source passed Python compilation and Bash syntax checks. A wholly synthetic 100-image 400-decision fixture with **arbitrary non-D1 IDs** passed the contract 100/20/4. A deliberately inserted run-index 4 decision was rejected. The local end-to-end run used simplified stub fitting/simulation modules; it **does not certify numerical identity to the actual repository's native v2 siblings**. Final production must be run on the native repository with real authorized prefixes at point_reps=250, bootstrap=60, boot_reps=3, and pass the 0.01 oracle-anchor gate.

At production use a clean theory branch, export `PY`, `PREFIX4`, `PREFIX4_SHA256`, `CLIP4`, `CLIP4_SHA256` (and optional `WORK`), then run `bash scripts/track_r/theory/run_th1_imagenet_v5.sh`. The launcher validates, writes an immutable sealer output, commits it, and prints the SHA-256 and full commit SHA. **Push and register the output before opening XP-N indices 4–63.** If source QA or production fails near Nov 2, 12:00 ET, explicitly decline v5. Never repair its model after outcome unblinding.

Separately, the prospective protocol hypotheses N-H1a/N-H1b are recorded in `TH_XPN_PROTOCOL_PRESCORE_20261031.md`, commit **a1ea26b0bd153fffec64a26ef730bd0aeb84d8a0**, and are immutable.
