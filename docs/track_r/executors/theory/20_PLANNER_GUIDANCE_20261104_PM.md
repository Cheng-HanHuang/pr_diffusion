# Planner guidance to the theory executor, round 20 (2026-11-04, evening)

**Responds to:** your round-19 review of the two S2 edits.

## 1. Your wording is adopted

**Both sentences are replaced as you proposed.** I split each into shorter sentences. Cheng prefers short sentences in the paper.

**Measurement paragraph:**
> "In this diagnostic, the normalized global-flip defect of the ImageNet denoiser is much smaller than that of the FFHQ denoiser. Single-channel rotations produce larger defects on both datasets."

**Link to Theorem 2:**
> "These observations are qualitatively consistent with denoiser-dependent orientation preferences. They do not test Theorem 2 directly. Its bound concerns the sampler's transition defects along its own trajectories, not denoiser defects on noised ground-truth images."

**Your rearranged bound** (an observed asymmetry η implies a lower bound on the total transition defect) is already stated in S2: "the bound turns an observed orientation preference into a lower bound on the sampler's lack of equivariance along its own path". I left it unchanged.

The supplement builds to 21 pages, clean.

## 2. For your information: every orientation test on the ground-truth images

This is new in S2 as `tab:s2-gt`; the G7 return was completed today.

**Upright kept, FFHQ (180) / ImageNet (100):**

| Test | FFHQ | ImageNet |
|---|---|---|
| CLIP face | 180 | — |
| CLIP G1 | 180 | 92 |
| CLIP G2 | 180 | 92 |
| S_mid global2 | 180 | 95 |
| S_mid all8 | 180 | 95 |
| 2-forward (t = 300) | 180 | 93 |
| UnivFD (all eight) | 172 | 89 |

**Notes on the table:**
- S_mid all8 chooses exactly as global2 on every image. It never picks a channel-wise flip on a ground truth.
- The ImageNet errors overlap little. No image is turned by all of G1, S_mid global2, 2-forward and UnivFD, and 20 are turned by at least one.

**One question, no action needed.** If you think the "all8 = global2 on every ground truth" fact belongs next to your Theorem 2 discussion of channel-wise flips, propose one sentence for S2. Otherwise nothing is needed.

**No other action.** S1–S3 remain final.
