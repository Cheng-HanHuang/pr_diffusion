# CP3 — Scientific readout and A12 XP-I matched-NFE blocker

**Checkpoint:** CP3 / A13; executor status **INTERNAL, NOT RELEASED**.
**Scientific freeze:** `docs/planning/track_r/XP_PREREG_v1_1.md`, amendment A12 §4 and A13 §4–5.
**Readout executor head:** `f6ccc548f2087eeb94834954767214e13710ff9e`.
**Result:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a16_cp3_joint_20261031T034143Z/XP_A16_CP3_JOINT_READOUT.json`.
**Status:** `XP_A16_CP3_INDEPENDENT_JOINT_READOUT_PASS_UNMATCHED_NFE`; `CP3_RETURN_AUTHORIZED=false`.

## 1. Validated results

Native receipts and GT-free scoring: five panels, 5,840 candidates in total. Independent A15 audit reproduced all seven frozen protocol executions and all per-image label/CLIP decisions. A16 independent metric readout verified 2,960 native-float PIQ VGG-LPIPS / SSIM scores and their aggregation (G4 1,440, G5 800, G3 720), plus one independent source-tensor spot check per stage. All 100 ImageNet GT/y identities passed CPU reconstruction with zero reported RMS discrepancy. GT was used only for evaluation.

- **G2 σ=0.02:** H-G1 **FAIL**, scaled τ=0.02003316, GT τ=0.02223162 (9.89% error); H-G2 **PASS**, success 0.927778 **at mean 8.0/8 runs** (no savings).
- **G2 σ=0.10:** H-G1 **FAIL**, scaled τ=0.1001658, GT τ=0.09808984 (2.12% error); H-G2 **PASS**, success 0.6032125 at mean 2.20162 runs.
- **G4 FFHQ DAPS-4K:** H1 **PASS**; success 0.939614, mean 2.80337 runs, gain +0.039341 over fixed equal-mean-run comparator.
- **G5 ImageNet DAPS-4K:** H1 **PASS**; success 0.638853, mean 2.40444 runs, gain only +0.001043 over fixed equal-mean-run comparator.
- G4/G5 frozen DAPS-4K per-metric best-of-4 and best-of-8 PSNR, PIQ SSIM and LPIPS were recomputed; all are **GT-only per-metric oracles**, not GT-free policy outputs. Do not equate them with first-run or with E0/E1.

## 2. XP-I PC-DAPS vs DAPS (180 paired FFHQ images / identical signed y)

| Analysis | PSNR dB | SSIM | PIQ VGG-LPIPS | GOOD25 | Measurement |
|---|---:|---:|---:|---:|---|
| DAPS-4K one run | 25.2171 | 0.6901 | 0.2598 | 0.68333 | one native DAPS run |
| PC-DAPS native mean of four **metrics** | 28.0712 | 0.7971 | 0.1884 | 0.79028 | one four-particle bag |
| PC-DAPS E0 residual-selected particle | 28.5615 | 0.8064 | 0.1788 | 0.82778 | same bag, no GT |
| PC-DAPS E1 CLIP orientation of E0 particle | 29.0971 | 0.8225 | 0.1656 | 0.84444 | same bag, no GT |

PC-DAPS E1 versus DAPS first run: +3.8800 dB, +0.13239 SSIM, −0.09421 LPIPS, +0.16111 GOOD25 (29 extra successes / 180). E1 versus E0 **alone**: +0.5356 dB and +0.01667 GOOD25 (3 extra successes / 180). No GT was used for E0/E1 selections.

**This is descriptive near-equal nominal-work evidence, NOT a verified matched-NFE test.**

## 3. The frozen-rule obstruction — stop/ask under A13 §4

A12 preregistered XP-I as **at matched denoiser evaluations per image**, comparing PC-DAPS's four-particle bag (native method, E0/E1) with **DAPS under the frozen protocol** with `N_MAX` chosen to match PC-DAPS's mean NFE. Protocol v1 is frozen at **M=2**. A DAPS-4K run is 400×5=2,000 nominal NFE, so **the frozen stopping rule costs at least 4,000 NFE per image**, irrespective of `N_MAX` (minimum two runs). The native PC-DAPS bag's base ODE schedule costs 4×(130×1+70×5)=**1,920 nominal particle-equivalent denoiser calls**; native receipts recorded 126 pruning events over 180 bags, establishing **at least 1,928.4 average** when counting logged extra calls. Some prior-consistency calls were not logged, so exact PC-DAPS NFE is not available. An exact match to a frozen M=2 DAPS protocol has **not** been demonstrated and is not supported by the audit.

We computed a same-image, same-measurement, **one-DAPS-run** 2,000 nominal NFE control; this is **outside protocol v1's M=2 selection**. No `N_MAX` manipulation may disguise it as the preregistered matched-NFE policy. The one-run comparison remains valuable as an explicitly descriptive **near-nominal** comparison.

This is the A13 §4 condition: **a frozen rule cannot be applied as written**. The executor must stop and request a planner disposition rather than declare CP3 complete or retrofit M=1/retune the policy post hoc.

## 4. Decision requested from planner

Choose explicitly and record the choice:

1. **Allowed descriptive-only CP3:** accept G2/G4/G5 as frozen-confirmatory, retain XP-I only as **descriptive near-nominal work with unmatched actual NFE** and explicitly mark the original matched-NFE XP-I comparison **unresolved**; authorize an honestly labeled CP3 return (not a matched-NFE claim).
2. **Strict preregistration:** defer full CP3 acceptance until independent per-bag network forward accounting and a genuinely matched comparator are available. This may require new runs or explicit protocol amendment; do not silently construct a frozen-M=2 result from the one-run comparator.

In either case, A13's ≤20-line checkpoint return should disclose **both** H-G1 failures, low-noise G2's zero run savings, the marginal G5 gain, and the XP-I NFE limitation. The planner independently reexecutes protocol outputs with tolerance ≤0.002. **No new GPU, model inference, CLIP or SSIM/LPIPS scoring is needed for the completed descriptive evidence.**

## 5. Local evidence handles (not publishable paths)

- CP3 frozen protocol: `xp_a15_cp3_protocol_20261030T210259Z`.
- Independent A15 protocol audit: `xp_a15_cp3_protocol_20261030T210259Z/XP_A15_CP3_INDEPENDENT_AUDIT.json`.
- PC-DAPS NFE ledger: `xp_a16_cp3_nfe_20261031T024828Z/XP_A16_CP3_NFE_LEDGER.json`.
- ImageNet GT/y identity: `xp_a16_cp3_gt_y_20261031T024828Z/XP_A16_CP3_IMAGENET_GT_Y_IDENTITY.json`.
- Native metric receipts: `xp_a16_cp3_piq_{G4,G5,G3}_20261031T030003Z/XP_A16_CP3_NATIVE_PIQ_METRICS_RECEIPT.json`.
- Joint readout: `xp_a16_cp3_joint_20261031T034143Z/XP_A16_CP3_JOINT_READOUT.json`.

**No sanitized CP3 `returns/` commit is authorized by this note.**
