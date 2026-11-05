# Supplement: sixth integration build (planner, 2026-11-05)

**Build.** Run from this folder, after building the main paper once (the cross-references read `../main.aux`):
```
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

**Result.** 22 pages (2026-11-05: new section `SUPP_SELECTION_PROTOCOLS.tex`, "Selection protocols in recent work", input after S4 so the theory section numbers stay unchanged; paper names applied in the planner-owned files, see below. 2026-11-04: S2 adds the equivariance-defect table; S4's ImageNet table gains the E1 column. 2026-11-03, evening: S4 adds the prior orientation test, XP-C3, and PC-DAPS on ImageNet, XP-I2): S3 has the dated test-6 addendum; S4 includes the ImageNet four-run table and the image-disjoint panel; **new S6 shows all 15 images without a good run** (two float pages). No LaTeX errors, no undefined references, no overfull boxes.

## Sources

| File | Source | Planner integration edits |
|---|---|---|
| `TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex` | Theory `7e5c4b8` (round 12: the earlier integration edits adopted upstream; "equal mean runs" wording; H-G1 denominators named; a post-hoc label-free τ paragraph; round 13: typo fixed; round 15: a dated note on N-H1a) | None: identical to the theory file. |
| `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex` | Theory `061c618` | References to main-paper labels prefixed `M-` (`\externaldocument[M-]{../main}`). **2026-11-03:** two stale empirical sentences resolved with committed data: the ImageNet orbit-error share (XP-D4 `ef1cf10`, G5 `1981dea`, XP-N `2d791c5` strings) and ImageNet orientation (C-A11; the prior test, `tab:s4-prior`). **2026-11-04:** the equivariance-defect measurement (XP-C2, `9090a50`; new `tab:s2-c2` and a sentence linking it to `thm:tv`), the N2 prompts quoted (verified at `b1637f7c`), and the GT-only CLIP false-flip counts (C-A14). **2026-11-04, evening:** theory's wording for the two interpretive sentences (round 19 review); new `tab:s2-gt` (every orientation test on the ground-truth images; C-A15); the UnivFD sentence resolved. **2026-11-05 (planner):** R_s → R_b; "DAPS" → DAPS-2K for the FFHQ runs; "first image set"/"new images" → ImageNet/ImageNet′; the G1/G2 labels replaced by descriptions; the last `\pending` (per-solver source audit) removed, the sentence stands without it. Still open: the per-solver source audit as a supplement item. |
| `TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex` | Theory `061c618` | <ul><li>"planner-confirmed" → "in the pinned source".</li><li>An internal code path replaced by "the lightweight SITCOM reimplementation in our code base" (overfull line, double-blind).</li><li>The internal paragraph "Remaining source/figure checks for paper integration" commented out.</li></ul> |
| `TH_SUPPL_S3_COMPLETE_DRAFT_20261102.tex` | Theory `ecea698` (round 18: one dated sentence qualifying the XP-C3 score as an ELBO-motivated proxy); `2619eec` (round 17: test 1's E1 points); before that `7e5c4b8`. Its prefix is byte-identical to the integrated 20261031 copy (whose `\n` and double-blind fixes the theory executor adopted); a dated test-6 addendum follows | None: identical to the theory file. |
| `S4_PROTOCOL_EMPIRICAL.tex` | **Planner** (new), generated from `docs/track_r/planner_analysis/A16_cp3_and_exploratory/out/` | **2026-11-05:** DAPS → DAPS-2K; "bag" → PC-DAPS sample / particles; ImageNet/ImageNet′ names; L_b, R_b. |
| `SUPP_UNSOLVED_IMAGES.tex` (renders as S6) | **Planner** (new, 2026-11-03), from the 15 Nov 4 panels (XP `953e2f0`; SHA-256 checked against the receipt; tiles cropped losslessly by `../fig/unsolved/crop_tiles.py`) | — |
| `supp.tex` (wrapper) | Planner | Restates the orbit lemma (`lem:orbit`), since the main text no longer contains it; inputs S4, then the protocols section. **2026-11-05:** orbit index renamed s → b (R_b), as in the main text. |
| `SUPP_SELECTION_PROTOCOLS.tex` | **Planner** (new, 2026-11-05): the selection protocols of DPS, DAPS, SITCOM, FAST-DIPS, PC-DAPS and TGD (C-P1) | — |

**S4 contents:**
- every registered protocol hypothesis, with its outcome and the image-bootstrap interval of its gain;
- the XP-D4 budget frontier and the confidence-conditioned variant;
- the label-free threshold at σ = 0.02 and 0.10, own and cross-fitted;
- the decomposition of the oracle gap;
- PC-DAPS: same particles, with compute labels;
- the published operating points (A8.6);
- the registration record.

## Open items (before Nov 23)

1. **Done (theory `4f5fad0`):** the integration edits are adopted upstream; S1 uses "equal mean runs" and names its H-G1 denominators. S2 audit and S3 at `4f5fad0` match the integrated copies.
2. **S2, stale empirical sentences** (from the old main-text Sec. 4). Resolve or delete. Resolved: the ImageNet orbit-error share and ImageNet orientation (2026-11-03); the equivariance defect and the N2 prompts (2026-11-04). Resolved on 2026-11-04, evening: UnivFD. Open: the per-solver source audit:
   - "Fig. 1b reports the equivariance defect": the teaser no longer has that panel; XP-C2 is in G7.
   - The ImageNet orbit-error share: D1 has 279 O_g and 14 O_p among 1,600 runs (XP `5f6b213`).
   - Quote the frozen N2 face prompts.
   - UnivFD.
3. **Sections still to write:**
   - S5: the selection-protocol table removed from the main text (C-P1): **done 2026-11-05** (`SUPP_SELECTION_PROTOCOLS.tex`);
   - S1–S3: internal names (F8, B22, DEV80, D1-P2, XP-D4, XP-G, A8, bag), imperative notes and dated interpretations: theory round 21; define the two race sensitivity variants (log-weight prior scale 1.0 vs 2.5; pooled C-over-G rate 5/205) in S3;
   - S7: the full forecast tables (CP1, CP2, B16, D1-P2, FAST64) and the best-of-K curves per test;
   - S9: the controls (XP-K, if run);
   - XP-N and XP-C3, if they run.
4. **The registration table** gets the XP-N forecast row if XP-N runs, and `v5` if it is registered.
