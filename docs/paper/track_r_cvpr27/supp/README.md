# Supplement: fifth integration build (planner, 2026-11-03)

**Build.** Run from this folder, after building the main paper once (the cross-references read `../main.aux`):
```
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

**Result.** 20 pages: S3 has the dated test-6 addendum; S4 includes the ImageNet four-run table and the image-disjoint panel; **new S6 shows all 15 images without a good run** (two float pages). No LaTeX errors, no undefined references, no overfull boxes.

## Sources

| File | Source | Planner integration edits |
|---|---|---|
| `TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex` | Theory `7e5c4b8` (round 12: the earlier integration edits adopted upstream; "equal mean runs" wording; H-G1 denominators named; a post-hoc label-free τ paragraph; round 13: typo fixed; round 15: a dated note on N-H1a) | None: identical to the theory file. |
| `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex` | Theory `061c618` | References to main-paper labels prefixed `M-` (`\externaldocument[M-]{../main}`). |
| `TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex` | Theory `061c618` | <ul><li>"planner-confirmed" → "in the pinned source".</li><li>An internal code path replaced by "the lightweight SITCOM reimplementation in our code base" (overfull line, double-blind).</li><li>The internal paragraph "Remaining source/figure checks for paper integration" commented out.</li></ul> |
| `TH_SUPPL_S3_COMPLETE_DRAFT_20261102.tex` | Theory `7e5c4b8`. Its prefix is byte-identical to the integrated 20261031 copy (whose `\n` and double-blind fixes the theory executor adopted); a dated test-6 addendum follows | None: identical to the theory file. |
| `S4_PROTOCOL_EMPIRICAL.tex` | **Planner** (new), generated from `docs/track_r/planner_analysis/A16_cp3_and_exploratory/out/` | — |
| `SUPP_UNSOLVED_IMAGES.tex` (renders as S6) | **Planner** (new, 2026-11-03), from the 15 Nov 4 panels (XP `953e2f0`; SHA-256 checked against the receipt; tiles cropped losslessly by `../fig/unsolved/crop_tiles.py`) | — |
| `supp.tex` (wrapper) | Planner | Restates the orbit lemma (`lem:orbit`), since the main text no longer contains it; inputs S4. |

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
2. **S2, stale empirical sentences** (from the old main-text Sec. 4). Resolve or delete:
   - "Fig. 1b reports the equivariance defect": the teaser no longer has that panel; XP-C2 is in G7.
   - The ImageNet orbit-error share: D1 has 279 O_g and 14 O_p among 1,600 runs (XP `5f6b213`).
   - Quote the frozen N2 face prompts.
   - UnivFD.
3. **Sections still to write:**
   - S5: the selection-protocol table removed from the main text (C-P1);
   - S7: the full forecast tables (CP1, CP2, B16, D1-P2, FAST64) and the best-of-K curves per test;
   - S9: the controls (XP-K, if run);
   - XP-N and XP-C3, if they run.
4. **The registration table** gets the XP-N forecast row if XP-N runs, and `v5` if it is registered.
