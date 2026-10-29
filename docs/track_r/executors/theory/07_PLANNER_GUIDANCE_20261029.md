# Planner guidance to the theory executor, round 7 (2026-10-29, 04:00 ET)

**Responds to:** round 6 (`2fd1c02`), accepted in the round-6 relay. v3 is frozen at `e71026e` and registered.

**Governing amendment:** XP prereg **A11**.

## 1. The paper's direction is decided

Cheng chose option (a).

**The paper now delivers:**
1. a **ground-truth-free best-of-K protocol**: stop at M = 2 measurement-consistent runs, keep the residual minimizer, correct orientation with a zero-shot test, and flag images with no consistent run or an uncertain orientation;
2. the finding that **GT-free selection is predictable, while unsolved images are not**: your race model holds where it is tested, and both prospective forecasts miss on generation.

**For your work, this means:**
- **The orbit lemma is presented as classical.** In the main text it shrinks to a short paragraph (Sec. 3): the symmetry, orientation-blindness of measurement-only selection, and the posterior ratio within an orbit.
- **Prop. ratio, Cor. blind, Thm. equivariant and the TV bound move to Supplement S2.** The former main-text section is kept verbatim in `sec/X_symmetry_moved_to_suppl.tex` (this upload) as your source.
- **The selection model is now the theoretical core** (main-text Sec. 5): S1–S3, the race, the estimators, the two prospective tests, the identification limit. Your C-T8 data bounds are cited there as exploratory (B22 [0.870, 1], DEV80 [0.611, 1]).

## 2. Protocol v1 (frozen; in this upload)

- `GTFREE_PROTOCOL_v1.md` (SHA `7e9a5fd8…`) and `gtfree_protocol.py` (SHA `d5f973fe…`), frozen at planner `c598176`, with development outputs in `dev_v1/`.
- **Held-out panels:**
  - D1-P2 (ImageNet FAST, 16 runs, τ transferred from FFHQ);
  - B16 (SITCOM, 16);
  - F8 (DAPS, 8).
- **Hypotheses:**
  - H1: no loss against fixed best-of-K at equal mean runs (≥ −1 point);
  - H2: the gain on DAPS is ≥ +2 points and larger than on SITCOM and ImageNet;
  - H3: FLAG_C precision for unsolved images ≥ 0.5;
  - H4: the ImageNet orientation flag raises selective success by ≥ 5 points.
- **Dev (FAST FFHQ, in-sample):**
  - 91.3% at 8.4 runs per image, against 89.1% for fixed E0 at equal runs and 91.2% for best-of-16;
  - better than fixed K at 12/12 operating points;
  - DAPS cohort B +4.9 points, SITCOM cohort B −0.1.

## 3. Requests, in priority order

**(a) D1-P2 half-page note** (already requested). Under your locked plan, with no refit.

**(b) Protocol theory: new, and now central.** The method section needs a short formal account (main text: one proposition; details in Supplement S1). Under your class model (S1, the race, per-image class probabilities, runs i.i.d. given the image):
1. **Closed forms for the stop-at-M rule truncated at N_max:**
   - its success probability;
   - its expected runs, a truncated negative binomial in the consistency rate s;
   - the probability of FLAG_C.

   Note that consistency is ρ ≤ τ, so about 5% of GOOD runs are not consistent by construction (τ is the 95th GOOD percentile). Model it as a per-class pass probability.
2. **A condition under which stop-at-M beats fixed best-of-K at equal mean runs.** The intuition to formalize: fixed K spends K runs on every image; stopping moves runs toward images with small s, so its advantage grows with the heterogeneity of s across images. Its success, given M consistent candidates, is a race among those candidates that does not depend on s.
3. **The failure-type mechanism.** Show how the gain at equal compute depends on the share of failed runs that are CONSISTENT (c) rather than INCONSISTENT (u). As c/(c + u) → 1, consistency stops predicting correctness and the gain should vanish. Dev: DAPS cohort B (12% C) +4.9; SITCOM cohort B (60% C) −0.1.
4. **(Optional) `v3P`.** A registered forecast of the protocol's success and mean runs on D1-P2 (your v2/v2 E1 model) and on B16/F8 (v3), at M = 2 and N_max = panel size. The estimand is the protocol's random-order average on the realized panel (spec §2), so it is the same conditional-prefix situation as v1.
   - **Register by Nov 1, 12:00 ET.**
   - Code must run end to end on the real inputs with a contract; I run production.
   - If I run it after protocol outputs exist, the registry will say so.

   If it does not fit, decline: (1)–(3) matter more.

**(c) Supplement:**
- **S1 (protocol):** (b)(1)–(3).
- **S2 (symmetry):** from `X_symmetry_moved_to_suppl.tex`, plus the per-solver source audit (C-T3).
- **S3 (selection model and the identification bounds).**

  Also one paragraph on why blur or misalignment cannot break the twin ambiguity: |F(x∗k)| = |F x|·|F k| for both twins. Coded masks and references can (Candès–Li–Soltanolkotabi 2015; Barmherzig et al. 2019; DOLPH uses a random mask). Cheng asked about this, and a reviewer may too.

**Deadlines:**
- (a) and (b)(1)–(3): **Nov 3**;
- `v3P`: **Nov 1, 12:00 ET**, if attempted;
- Supplement drafts: Nov 6.

## 4. Reads and stops

- **You may read:**
  - this upload;
  - `theory_inputs/`;
  - the planner-branch status files;
  - the paper draft.
- **Still not `xp/returns/`.** Protocol outputs reach you through me, or through `theory_inputs/` once committed (A11.4).
- **Stop and ask** if `v3P` would need any B16, F8 or D1-P2 run beyond runs 0–3 (D1-P2) or the historical prefix (B16/F8).
