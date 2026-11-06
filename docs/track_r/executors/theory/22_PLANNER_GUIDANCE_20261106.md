# Theory guidance, round 22 (planner, 2026-11-06)

**Context.** Cheng ran two AI reviews of the Nov 5 draft (`docs/track_r/reviews/`; triage in `AI_REVIEWS_20261105_TRIAGE.md`). The planner answered most items in writing. Five new sentences carry theory content. Please check them with your round-21 reply (due Nov 12). Round 21 itself is unchanged.

## Sentences to check

1. **Sec. 5, after the race is introduced:** "This is the tail competition of best-of-n [yano]: a CONSISTENT run is picked only if it beats every GOOD run, not merely if it passes."
2. **Sec. 5, mixture:** "Heterogeneity across images enters through a Dirichlet mixture over the class probabilities; it is the only dependence between the runs of an image."
3. **Supplement §Dependence:** runs are exchangeable by construction (fresh seeds, shared measurement only). By de Finetti's theorem they behave as independent draws given the image's class rates, which is what Prop. 1 and the mixture assume. Finite exchangeability makes this approximate. Is the wording safe?
4. **Sec. 6 and supplement §Other verifiers:** "Under Gaussian noise, a run's likelihood depends on the measurement only through its residual, so telling a CONSISTENT run from a GOOD one needs the prior, not another residual."
5. **Sec. 7 and supplement §Other operators:** with a large null space (inpainting), we expect failed runs to be CONSISTENT, so stopping cannot help (Prop. 1). Lower oversampling should mean more consistent failures. These are stated as expectations, not results.

Reply with "OK" or a replacement sentence for each. No new theory is requested.

## Unchanged

- No reads of `xp/returns`.
- Only the planner edits `PREDICTIONS_REGISTRY.md`.
- Double-blind.

## Planner's view

The reviews found no flaw in the theory. They asked for intuition in the main text, which these sentences supply. The supplement freezes Nov 22.
