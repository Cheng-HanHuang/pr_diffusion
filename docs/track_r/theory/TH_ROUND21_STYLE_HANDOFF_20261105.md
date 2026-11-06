# Theory round 21 — S1–S3 nomenclature and style handoff (2026-11-05)

**Scope:** Style-only edits to theory-owned S1–S3 LaTeX. Scientific claims, prospective forecasts, registries, measured values and the stopping/race/TV derivations are unchanged. **No** `xp/returns/` was accessed. The theory executor did **not** edit `PREDICTIONS_REGISTRY.md` or the main manuscript. The paper is double-blind; citations to SITCOM remain in the third person.

## Source files to integrate

| Supplement | Theory file | What changed |
|---|---|---|
| S1 | `docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex` | Cohorts A/B; DAPS-2K; FAST-DIPS on ImageNet and ImageNet′; PC-DAPS **sample**; no internal stage names, dates or process language; shorter H2/H1/threshold sentences; forecast checks distinct from percentage-point gains. |
| S2 relocated source | `docs/track_r/theory/TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex` | Adopted planner integration at `ed35f11` including b-indexed orbit rotations, numeric equivariance table, GT orientation-test table, G1/G2 descriptions, ImageNet′ and DAPS-2K/4K names, removal of unresolved source-audit pending marker. Existing TV theorem and its formulas are byte-identical to the pre-edit theory source. Filename retained for the existing `supp.tex` input; its content is no longer a verbatim copy of the oldest theorem source. |
| S2 proofs | `docs/track_r/theory/TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex` | Adopted the earlier integrated anonymization edits; removed process comments, pinned commit hashes and source-specific local paths; DAPS-2K/DAPS-4K and implementation-neutral SITCOM wording. |
| S3 | `docs/track_r/theory/TH_SUPPL_S3_COMPLETE_DRAFT_20261102.tex` | Paper names and concise sentences; no internal project or dated headings, commit IDs or process imperatives. Names of the race sensitivities are defined explicitly; forecast coverage is counted as checks, not “percentage points.” Old numerical outcomes and v5 scoring rule unchanged. |

**Race specification:** $w_G=1$, $w_I=0$, global and partial orbit errors share $w_O$. The primary independent zero-mean Gaussian priors on log $w_O,w_C$ use scale 2.5. The tighter-prior sensitivity (`race_tau1`) uses scale 1.0, shrinking both toward 1. The transferred-tail sensitivity fixes the pooled C-over-G rate at $5/205$ for every solver.

**Source consistency checks:**
- S1: 12 original `\\label` strings preserved.
- S2 relocated: all 6 original `\\label` strings preserved, plus planner's `tab:s2-c2` and `tab:s2-gt`.
- S2 proofs: 1 original `\\label` preserved.
- S3: all 16 original `\\label` strings preserved. They contain historical date identifiers; their identifiers alone remain unchanged because cross-references depend on them.
- Text scan across all four files: no B22/DEV80/F8/B16/D1-P2/XP-D4/XP-N/XP-G/XP-C3/A8, “bag”, “authorized”, “released”, “planner”, “executor”, `R_s` or unresolved `\\pending` tokens. No visible GitHub URLs, author names, local cluster paths, or dated headings. All LaTeX brace and environment counts balance.
- All S1 and S2-proofs math display blocks unchanged, byte-for-byte. In S3 the only math display change renames the two cohort labels inside the existing population-bound `cases` environment; numeric endpoints and mathematical expressions are identical. The S2 relocated TV theorem was separately checked byte-identical to the older theorem body.
- No new integrated LaTeX build is claimed. The planner's previously accepted 21-page PDF remains the last confirmed compilation. After copying these sources, build the supplement and check tables, floats, cross-references and line wrapping.

## Three main-text sentence checks (requested §3)

**1. Intro: shared noise energy cancels.** Correct for the *difference of two squared measurement residuals on the same noisy measurement*. The shared realized $\\|\\eta\\|^2$ cancels; the cross term $-2\\langle\\eta,e_i-e_j\\rangle$ remains. **Precision note:** Sec. 4 currently says “the noise energy $m\\sigma^2$ cancels.” Since $m\\sigma^2$ is the *expected* noise energy rather than the realized term, “the common $\\|\\eta\\|^2$ term cancels” is strictly more precise. This is an optional main-text wording change for the planner, not a challenge to the identity.

**2. Intro: only first n moments are identified.** Correct under the idealized image-mixture model with conditionally iid Bernoulli runs and known *population* distribution of n-run counts. In a finite panel, even those moments are estimated rather than known exactly. Optional clearer wording: “Even if the distribution of n-run counts were known exactly, it would determine only the first n moments of the image-wise success rates. These do not determine the fraction with rates near zero.” Do not interpret finite unsuccessful runs as proof of $p=0$.

**3. Sec. 5: race ceiling.** Correct as the $K\\to\\infty$ per-image limit of the class-conditional exponential-rank surrogate with $w_G=1$, $w_{O_g}=w_{O_p}=w_O$, $w_I=0$ and a positive denominator:
$E_0(\\infty)=p/[p+(o_g+o_p)w_O+cw_C]$.
The formula is *not* an exact physical-residual law for native diffusion solvers, nor the average of an image-mixture ratio. The supplement states those limitations.

**Integration:** The planner may copy the four theory LaTeX files. Only the planner should update the main paper and prediction registry. This handoff commits no new forecasts, calibration data, or selection methods.
