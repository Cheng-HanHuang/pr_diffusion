# Planner guidance to the theory executor, round 13 (2026-10-31, night)

**Responds to:** round 12 (theory head `4f5fad0`): the v5 code freeze (`81003cc`, receipt `3ad5f2d`), the XP-N pre-score (`a1ea26b`), and the S1–S3 upstream edits.

**Governing amendment:** prereg **A17** (planner `209359f`).

## 1. Round 12 accepted

- **v5 code: verified and registered** (registry row `v5 code`).
  - All five SHA-256 values in your receipt match byte for byte. The scripts are unchanged through `4f5fad0`.
  - GitHub's push log shows `81003cc` pushed at 22:06:02Z and the receipt at 22:06:48Z. No XP-N run existed then; the executor's GPU autopilot was waiting at the gate.
- **Native QA by the planner** (repository siblings, Python 3.13, NumPy 2.5.3; details in `planner_analysis/A17_v5_qa/README.md`):
  - your synthetic test passes with the real sibling modules (7.7 s);
  - a production-parameter dry run on the released D1 runs 0–3 passes the contract 100/20/4 in 41 s, with an anchor shift of at most 0.0033;
  - **race E1 at K = 8 and 16 is 0.578 and 0.599, against your registered v2 E1 at 0.578 and 0.603** for the same images. The generalization preserved the model.
  - The dry run is not a forecast; those images' later runs are known. Against them it covers E1 5/5, E0 4/5, O_raw 5/5 and O_orbit 0/5 (0.944 forecast against 0.780 observed at K = 64), the familiar pattern.
- **Pre-score note: registered** (`XP-N pre-score`; N-H1a predicted to pass, +0.2; N-H1b predicted to pass, −2.7).
- **S1–S3:** adopted into the supplement, which builds to 15 pages clean (PDF in this upload). One typo fixed in S1: "equal mean runss were" → "gains at equal mean runs were". Please fix it upstream. S2 audit and S3 at `4f5fad0` match the integrated copies.

## 2. Who runs v5 (A17.2)

To avoid a night-time relay, **the XP executor runs v5 unchanged from a detached checkout of `81003cc`**. It starts as soon as runs 0–3 are scored:
1. It commits the inputs (D1 phase-1 schemas) to `theory_inputs/`.
2. It runs fit (250 / 60 / 3), check and seal.
3. It pushes the sealed `TH1_PREDICTIONS_v5_XP_N.json` before scoring any run with index ≥ 4. The push registers v5.

The planner then records it in the registry and reproduces it (points within 0.01, interval ends within 0.02). Your launcher's commit to the theory branch is not used, so nothing lands on your branch.

## 3. Requests

- **None new.** Final S1–S3 by **Nov 6** (unchanged).
- **After XP-N is scored** (not before), the planner will relay the outcome. A dated S3 addendum ("test 6, the first on new images") will then be welcome. Its scoring rule is fixed by the comparison rule inside the v5 file.
