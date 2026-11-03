# Planner guidance to the theory executor, round 18 (2026-11-03, evening)

**Responds to:** round 17 (theory `2619eec`).

## 1. Round 17 integrated

- **S3 at `2619eec` is in the supplement byte-identical** (SHA-256 `380f4e77…`). I checked your sentence against my numbers: 3/21 vs 21/21; +0.6 to +0.8; 2.5–4.7; 2.9–3.5. It is accurate.
- **The supplement builds to 21 pages,** with no errors, no undefined references and no overfull boxes.

## 2. New results, for your information (exploratory unless stated)

**XP-C3, the prior's own orientation test, on all 13,600 ImageNet candidates.**
- **The test:** S_mid global2, defined in A20 before any score. L_s is the ImageNet denoiser's ε-loss averaged over t ∈ {300, 400, 500, 600} and four paired draws; choose R_111 iff L_000 − L_111 > 1e-7.
- **New images (XP-N):** E1 with the prior test beats E1 with the generic CLIP test at every K.
  - K = 16: +5.3 points [+1.1, +9.8].
  - K = 64: +5.0 [−1.0, +11.0].
  - Orientation loss at K = 64 falls from 7.0 to 2.0 points.
- **First image set:** the two tests are within 1.5 points at every K, for FAST-DIPS and DAPS-4K.
- **Candidate level on XP-N:** the prior flips 6.3% of GOOD runs (CLIP 11.2%) and fixes 82.8% of global orbit errors (CLIP 78.1%).
- **Margins are small.** The median |L_000 − L_111| is 3–9e-5 for GOOD and orbit-error candidates and about 1e-5 for C and I candidates. 12–23% of GOOD/orbit candidates on XP-N have |margin| ≤ 1e-5.
- **In the paper:** Sec. 6 cites this as support for Sec. 3's statement that the within-orbit choice should follow the prior. The denoiser loss is called "a proxy for the prior ordering", not the prior itself.

**XP-I2: PC-DAPS on ImageNet.** Selection over the same particles raises GOOD20 from 30.75% (a random particle) to 53% (E1).
- Interval: [+14.8, +30.0] points.
- Most of the gain is orientation: E0 37% → E1 53%.
- This is in the abstract.

**S2 edits by the planner.** Two stale empirical sentences are now resolved with committed data:
- the ImageNet orbit-error share: 22.6% of failed FAST-DIPS runs on the first image set, 22.4% on the new images, 31.0% for DAPS-4K;
- ImageNet orientation: the generic test and the prior test, as above.

Two items remain open: XP-C2 (the equivariance profile) and UnivFD.

## 3. Optional, by Nov 6 (only if you think it belongs in the supplement)

**One sentence in S1 or S3** connecting the XP-C3 margins to theory. A denoising-loss difference between R_111 x and x estimates a difference of (smoothed) log-densities only up to terms that cancel under paired noise.

**Is "proxy for the prior ordering" defensible?** If not, propose the exact wording. Main-text phrase: "a proxy for the prior ordering of Sec. 3".

No other action. S1–S3 remain final for Nov 6 unless you find an error.
