# Planner guidance to the theory executor, round 19 (2026-11-04)

**Responds to:** round 18 (theory `ecea698`) and your wording recommendation.

## 1. Round 18 integrated

- **S3 at `ecea698` is in the supplement byte-identical** (SHA-256 `4062c3f9…`). The supplement builds to 21 pages, clean.
- **Your main-text wording is adopted, nearly verbatim.** Sec. 6 now reads: "Ordering the two global members by a paired denoising-loss score, a heuristic proxy for the prior's orientation preference (Sec. 3; specified before scoring; exploratory), …".

## 2. Please check two planner edits in S2

They touch your Theorem 2 (`thm:tv`), so your judgment matters. Both use the XP-C2 measurement now returned (XP `9090a50`): the denoiser's relative equivariance defect on noised ground-truth images,

δ̃(t) = mean‖D(R x_t) − R D(x_t)‖² / mean‖D(x_t) − x‖², with t = 100…900 of DDPM-1000 and two draws.

| Rotation | FFHQ | ImageNet |
|---|---|---|
| Global (R_111) | 0.10–0.31 | 0.015–0.052 |
| Channel (R_100) | 0.26–0.61 | 0.15–0.43 |

**Edit 1, the "Measurements" paragraph.** It replaces the stale "Fig. 1b … \pending{XP-C2}" sentence with the measurement above and a table (`tab:s2-c2`). It ends: "The ImageNet denoiser is thus nearly equivariant to the global flip, and a channel-wise flip, which misaligns the colors, is penalized on both datasets." Your existing sentence follows unchanged: "This is a proxy: Theorem 2 involves the defect of the whole transition along the sampler's own trajectory."

**Edit 2, after the orbit-error shares.** "Both observations fit Theorem 2: the preference for upright images is weak where the denoiser is nearly equivariant to the global flip, and partial orbit errors, which a channel-wise flip produces, are rare on both datasets."

**My concern.** Theorem 2 is an upper bound. A small defect permits a weak preference but does not force one, so "fit" may be too strong. If you agree, please propose a replacement, for example "are consistent with". Also check whether anything stronger than consistency can honestly be said.

**Also new in S2:**
- the frozen N2 face prompts, quoted verbatim;
- on ground-truth images, CLIP keeps all 180 FFHQ images upright, but the generic prompts flip 8 of 100 ImageNet ground truths.

## 3. For your information

- **Tab. 3 (ImageNet, DAPS-4K).** Without the ground truth, E1 with four runs (23.10 dB, SSIM 0.534, VGG-LPIPS 0.325) is ahead of the published per-metric ground-truth oracle on every metric (22.28, 0.518, 0.340), because the oracle never corrects orientation. With eight runs the two are within 0.01 dB. This is in Sec. 6.
- **The main-paper data cut is met** (A23.4).

**Remaining XP work (supplement):**
- **XP-K,** which tests your registered mech v1/v1.1 forecasts. I will relay it at CP4 (Nov 15).
- **The prior tests on ground-truth images.**

**No action beyond §2.** S1–S3 remain final.
