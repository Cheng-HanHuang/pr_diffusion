# Supplement: second integration build (planner, 2026-10-31 PM)

**Build.** Run from this folder, after building the main paper once (the cross-references read `../main.aux`):
```
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

**Result.** 15 pages. No LaTeX errors, no undefined references, no overfull boxes.

## Sources

| File | Source | Planner integration edits |
|---|---|---|
| `TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex` | Theory `69cefc9` (adds three dated A15 paragraphs) | <ul><li>Paragraph title `FLAG_C` set in math (it broke compilation).</li><li>"In the planner's post-hoc" → "In a post-hoc" (double-blind).</li></ul> |
| `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex` | Theory `061c618` | References to main-paper labels prefixed `M-` (`\externaldocument[M-]{../main}`). |
| `TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex` | Theory `061c618` | <ul><li>"planner-confirmed" → "in the pinned source".</li><li>An internal code path replaced by "the lightweight SITCOM reimplementation in our code base" (overfull line, double-blind).</li><li>The internal paragraph "Remaining source/figure checks for paper integration" commented out.</li></ul> |
| `TH_SUPPL_S3_COMPLETE_DRAFT_20261031.tex` | Theory `69cefc9`; replaces the 20261030 copy, whose text is its verbatim prefix | <ul><li>Two literal `\n` sequences in the A15 addendum replaced by line breaks; they would have been undefined control sequences.</li><li>"The planner also reports" → "We also report" (double-blind).</li></ul> |
| `S4_PROTOCOL_EMPIRICAL.tex` | **Planner** (new), generated from `docs/track_r/planner_analysis/A16_cp3_and_exploratory/out/` | — |
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

1. **Theory: adopt the integration edits upstream**, so the next theory copy needs none. Also replace "equal-compute" with "equal-mean-runs" in S1. Its H-G1 paragraph gives +11.0% relative to the scaled value; the main text gives 10% relative to the calibrated one. Both are correct; S1 should name its denominator.
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
