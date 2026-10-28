# Track R registered freeze packaging review — October 28, 2026

**Review status:** PASS — no numerical erratum or refreeze requested. **Theory executor did not access xp/returns, FFHQ K>4 outcomes, XP-D1 phase-2 outcomes, or XP-D2/D3 outcomes.** The source is the user-uploaded archive theory_upload_20261028.zip (all NINE individual file SHA-256 checks passed) and the planner's guidance 04_PLANNER_GUIDANCE_20261028.md. The remote theory branch was independently checked to be at 340dc5dcbcde96df4edc938f1b807faedc18936b **before** the theory executor's new writing commits.

## Registered immutable artifacts

- FFHQ v1: commit 8ef5f94d6de8aab4c3cc25b17fa797db0de1560b; SHA-256 c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753; 976 predictions, 252 anatomy curves and 60 crossover entries. Source A8, A8b and tau hashes preserved, and A8b was used ONLY for QA, not residual-gap fitting.
- FFHQ fixed-image companion: commit 5782149155056ea06869de4262dc2b9f259c3d10; SHA-256 74190e6722a11d1e7e16118984f89bbc5281f665629902db244b5740cd783462; 14,800 per-image records. Companion is for prespecified image-lineage exclusion, not a separate forecast.
- ImageNet FAST Good20 v2: commit 340dc5dcbcde96df4edc938f1b807faedc18936b; SHA-256 8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55; 36 forecasts and 12 anatomy rows. It abstains on E1_CLIP and leaves a separate A8.3 addendum route.

All three registered files match their planner registry hashes. Across their policy predictions, every point and 95% predictive interval satisfies 0 <= lo95 <= point <= hi95 <= 1.

## Code-level packaging review

1. The planner's cfefd00 change in seal_th1_a6_v1.py replaces an impossible near-exact float64-versus-six-significant-digit tau comparison by equality after formatting the exact tau with .6g. It does not modify a model parameter, random number, output forecast or original-label classification.
2. The planner's dd5c5e6 ImageNet v2 sealer and launcher enforce the authorized phase-1 CSV SHA, run the v2 draft contract, stamp the source/code hashes, runtime/seed context, original E1 abstention and UTC freeze time, refuse overwriting the existing output, and require a clean theory worktree. This is packaging, not a refit or a numerical model alteration.
3. ImageNet v2's declared code_commit is dd5c5e6 (the packaging commit) and draft_sha256 is retained. The statement that the planner executing production had earlier seen XP-D2/D3 aggregates is disclosed in v2 source_data_disclosure, while the original fixed model did not use them. This provenance caveat should remain visible in the paper and is not an arithmetic error in v2.
4. The source archive and registry assert contract PASS 976/252/60 and 36/12. The theory executor inspected these receipts and registered JSONs but did not independently reproduce production execution in this packaging-review round. Do not conflate document review with a second end-to-end rerun.

## Naming note — literature/supplement correction, not an artifact erratum

The ImageNet v2 model variant named tail is **the pooled FFHQ 5/205 rare-C rate carried over**, not a family-specific ImageNet heavy-tail fit and not necessarily a pessimistic stress arm. Its registered ImageNet E0 points are:
- K8 tail 0.413169 vs race 0.388202;
- K12 tail 0.417322 vs race 0.399266;
- K16 tail 0.415167 vs race 0.404939.

Thus **tail is more optimistic than race at every registered ImageNet K**. In narrative and plots, label it FFHQ-tail-transfer sensitivity, retain the immutable JSON model_variant field tail, and avoid claiming that it provides a lower envelope. No refreeze or mutation of registered text/values is justified.

## Pre-outcome interpretation and addendum

The one-page prediction interpretation was committed independently at TH1_PRE_RESULTS_INTERPRETATION_PLAN_20261028.md before this theory executor saw any post-prefix outcome. v1 is not refit. The optional D1 E1 addendum has separate source+contract+seal code; it is not registered until the executor's 400-run G1/G2 phase-1 input SHA is provided and production passes. Its completion never changes v2.

**Disposition:** No erratum to a frozen number detected. Record this review and wording note. If later evidence identifies a package defect, issue a dated erratum, never overwrite the original artifact.
