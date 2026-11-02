# Planner guidance to the theory executor, round 16 (2026-11-02, afternoon)

**Responds to:** round 15 (theory `7e5c4b8`): the S3 test-6 addendum, the complete S3 draft dated 20261102, and S1 with the N-H1a note.

## 1. Round 15 accepted and integrated

- **Integrated without edits.** S1 and S3 from `7e5c4b8` go in as they are. S3's prefix is byte-identical to the integrated 20261031 copy. The supplement builds to **17 pages** with no errors, no undefined references and no overfull boxes. The PDF is in this upload.
- **Your overlap check is reproduced:**
  - 67 images have both GOOD and orbit-error runs; 6 have GOOD only; 2 have orbit errors only; 25 have neither;
  - the correlation of the per-image counts is +0.369.
- **The diagnosis is accepted as written.**
  - **Non-identifiability** explains why the joint event is hard to extrapolate from four runs.
  - **Misspecification** says what went wrong here: a single Dirichlet imposes negative covariance between class shares, but the counts are positively associated across images.
- **One request:** keep "misspecification" as the headline of the diagnosis. The main text states the same thing in plain words: an image with an orbit error usually also has a GOOD run.

## 2. Status of S1–S3

**Final for Nov 6 unless you find an error.** Optionally, a short S2 pass: the stale empirical sentences stay marked; I resolve them when XP-C2 exists, or delete them by Nov 20.

## 3. For your information

- **XP-C3** (a prior orientation test on the ImageNet candidates, XP-N included) is now defined: the S_mid global2 test from v1.1's XP-C row (A20). Results by Nov 5.
- **What the result would mean either way:**
  - if the prior's own ordering beats generic CLIP on the new images, that supports Sec. 3's "the within-orbit choice must come from the prior";
  - if not, the main text says so.
- **No action needed now.**
