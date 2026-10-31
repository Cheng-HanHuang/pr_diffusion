# Supplement: first integration build (planner, 2026-10-31)

**Build.** From this folder, after building the main paper once (the cross-references read `../main.aux`):
```
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

**Result.** 10 pages, no LaTeX errors, no undefined references, no duplicate citations.

## Sources

Theory branch @ `061c618`, copied into this folder. Planner integration edits (layout only, no mathematical content):

| File | Planner edits |
|---|---|
| `TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex` | Paragraph title `FLAG_C` set in math (`$\textsc{flag}_C$`); it broke compilation |
| `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex` | References to main-paper labels prefixed `M-` (`\externaldocument[M-]{../main}`). Otherwise byte-identical to the theory copy (original SHA `fc34f945…`) |
| `TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex` | None |
| `TH_SUPPL_S3_COMPLETE_DRAFT_20261030.tex` | None |
| `supp.tex` (wrapper) | Restates the orbit lemma (`lem:orbit`), since the main text no longer contains it |

## Open items (before Nov 23)

1. **S2, stale empirical sentences** (from the old main-text Sec. 4). Resolve or delete them:
   - "Fig. 1b reports the equivariance defect": the teaser no longer has that panel; XP-C2 is in Tier C (G7).
   - The ImageNet orbit-error share: D1 has 279 O_g and 14 O_p among 1,600 runs, from XP `5f6b213`.
   - Quote the frozen N2 face prompts.
   - UnivFD.
   - The per-solver source audit pointer: now in S2's proof audit.
2. **New sections:**
   - **S4 (protocol, empirical):** the frozen spec, every hypothesis outcome, the frontier, the label-free threshold, the noise-level panels, the XP-D4 budget decomposition.
   - **S5:** the selection-protocol table removed from the main text (C-P1).
   - **S6 (registry):** hashes and freeze times for v1, v1 companion, v2, v2 E1, v3, v4, protocol v1, and the H2 pre-score note.
   - **S7:** the full forecast tables (CP1, CP2, B16, D1-P2, FAST64).
   - **S8:** PC-DAPS, descriptive (A15.3 analyses a and b).
   - **S9:** the controls (XP-K, if run).
3. **S3** cites numbers relayed up to Oct 29. Add a dated addendum for v4/XP-D4 (CP2) and for F8; do not edit the registered statements.
