# B21.6 visual GX interpretation

Status: human/assistant visual interpretation of B21.6 contact sheets, paired with `docs/b21/b21_6_gx_summary.md`.

Primary numeric source:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/gx_summary.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/gx_sensitivity.csv
```

Contact sheets:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/contact_sheets/b21_6_00046_contact_sheet.jpg
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/contact_sheets/b21_6_00171_contact_sheet.jpg
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/contact_sheets/b21_6_00480_contact_sheet.jpg
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/contact_sheets/b21_6_00746_contact_sheet.jpg
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics/contact_sheets/b21_6_00971_contact_sheet.jpg
```

## Visual labels for GX rows

| image | numeric GX observation at 0.5x median cut | visual dominant type | interpretation |
|---|---|---|---|
| `00046` | 120 bad candidates, 40 clusters, largest share 0.208, cross-seed yes, cross-arm no | mixed upside-down / symmetry-like attractors plus heavy ghosting | There are repeated bad basins, but not one all-arm dominant basin. LF/base arms enter related but not identical bad modes. This looks like a combination of phase-retrieval symmetry ambiguity and unstable prior hallucination, not a single clean repulsion target. |
| `00171` | 120 bad candidates, 84 clusters, largest share 0.033, cross-seed yes, cross-arm yes | diffuse smeared face hallucination / texture collapse | The contact sheet is visually similar in texture, but pairwise clustering fragments it. The failure is not a single exact attractor; it is a broad low-quality hallucination region. A repulsion memory against one prototype is unlikely to fix it. |
| `00480` | 89 bad candidates, 34 clusters, largest share 0.483, cross-seed yes, cross-arm yes | very stable upside-down / rot180-like symmetry basin | This is the cleanest shared bad basin. Nearly all low-PSNR candidates are a plausible face in the wrong phase-retrieval symmetry orientation. B21.2 final selector / symmetry-aware scoring is directly relevant. |
| `00746` | 120 bad candidates, 46 clusters, largest share 0.450, cross-seed yes, cross-arm yes; at 0.6x largest share 0.575 | stable wrong-identity/colorized face basin after early smeared modes | This is a strong cross-arm basin, but it is not pure rot180. It looks like a plausible face/identity basin that survives across base/LF arms. Candidate generation or basin-escape interventions matter here. |
| `00971` | 56 bad candidates, 47 clusters, largest share 0.036, cross-seed no, cross-arm yes | visually near-identical upside-down / rot180-like symmetry basin | The numeric average-linkage cut under-clusters because the median pairwise distance is extremely tiny. Visually this is a single symmetry-dominated basin. B21.2 final selector / symmetry-aware scoring is directly relevant. |

## Updated GX interpretation

The numeric GX table alone is conservative. It identifies `00480` and `00746` as large cross-seed/cross-arm basins, with `00746` crossing the repulsion-candidate threshold only at the 0.6x sensitivity cut. Visual review strengthens the symmetry interpretation for `00480` and `00971`, even though `00971` is numerically over-fragmented.

Practical conclusions:

1. **Symmetry/final-selector track remains high value.** `00480` and `00971` are visually dominated by upside-down / rot180-like outputs. These are not necessarily failures of generating a plausible face; they are failures of selecting the correct ambiguity representative under a symmetry-blind exact operator loss.
2. **Candidate-generation / basin-escape track remains necessary.** `00171` and `00746` are not fixed by a pure symmetry-aware selector. `00171` is broad texture/hallucination collapse; `00746` is a stable wrong-identity basin.
3. **Repulsion memory is not a universal next step.** It may be useful for `00480`/`00746`-like repeated basins, but `00046`/`00171`/`00971` show that visual failure modes are either mixed, diffuse, or symmetry-dominated. Repulsion should not replace selector-v2 and resampling/branching experiments.

## B21.2 consequence

B21.2 selector-v2 should prioritize symmetry-aware final scoring on candidate images where paths exist. Since B19.20/B19.16 replay CSVs do not expose candidate sample paths, the next prerequisite is to locate original candidate PNGs or rerun a small audited candidate panel with explicit sample-path logging.

## B21.3/B21.4 consequence

B21.3 reallocation and B21.4 guided variants should be evaluated against different failure categories rather than one aggregate bad25 count:

- symmetry-dominated: `00480`, `00971`;
- wrong-basin / identity attractor: `00746`;
- diffuse hallucination collapse: `00171`;
- mixed symmetry/ghosting: `00046`.
