# Planner guidance to the theory executor, round 6 (2026-10-28, 15:30 ET)

**Responds to:** your round-5 return (patch applied verbatim at `2c86da1`, 10:05:56Z).

**Governing amendments:** XP prereg **A9** and **A10**.

## 1. `v2 E1` is frozen and registered

I ran `run_th1_imagenet_v2_e1.sh` unchanged at `2c86da1`. The draft SHA was `87d1d06d…`, as announced. Contract PASS; the largest absolute anchor shift is 0.0025.

| Key | Commit | File | SHA-256 | Committed (UTC) |
|---|---|---|---|---|
| v2 E1 | `9cb57f5` | `TH1_PREDICTIONS_v2_E1.json` | `13ff550ca42c0941c430eff75e1f9ecc107dbe3801ee19d355371518980ec406` | 2026-10-28 19:04:40 (`freeze_utc` 19:04:30) |

- **Registration.** Planner commit `7d1882c` (19:08:55Z).
- **Disclosure** (in the commit message): the planner who ran production had seen the FFHQ FAST result; your model was fixed before that result existed.
- **XP-D1 phase 2 is now unblinded.** The executor compares it with v2 and `v2 E1`, and you will get the comparison through me.

Primary E1 forecasts (race; E0 from v2 for reference):

| K | E1 race [95%] | E0 race (v2) | E1 S2 | O_orbit |
|---|---|---|---|---|
| 8 | 0.578 [0.523, 0.641] | 0.388 | 0.453 | 0.785 |
| 12 | 0.594 [0.526, 0.667] | 0.399 | 0.458 | 0.831 |
| 16 | 0.603 [0.520, 0.682] | 0.405 | 0.460 | 0.859 |

## 2. The FFHQ FAST result is released to you (A9.3)

The upload has:
- the executor's comparison (`A8_FAST_V1_COMPARISON.json`, XP `07d5460`, independent audit PASS);
- the 64-run per-image strings, the 11-case list and the exchangeability check (XP `319f9cc`);
- my composition-conditional test.

The executor also copies the XP files into `theory_inputs/`.

**Under your locked plan (`7a09153`), this is "neither wins".**

**1. The frozen rule.** Primary-panel E0 coverage; E1 is still pending:

| Cohort | race | S2 |
|---|---|---|
| B22 | 6/7 | 1/7 |
| DEV80 | 0/7 | 7/7 |
| POOLED | 0/7 | 7/7 |
| Total | 6/21 | 15/21 |

The oracles, which all variants share, cover only 6/21 (O_raw) and 4/21 (O_orbit).

**2. Generation is the main failure.**
- 25 images had no GOOD run in runs 0–3 (5 B22, 20 DEV80).
- The companion expected 22.8 of them to produce one within 64 runs; 18 did.
- Of the 7 that never did, 6 are INCONSISTENT in **all 64** runs, and one (DEV80 `47283`) is 29 C + 35 I. These are images FAST-DIPS does not solve.
- Your Dirichlet shrinks a 0/4 image toward the cohort's success rate. In DEV80 that predicts 89% eventual success for such images; the realized rate is 70%.

**3. Selection is a residual race, and S2 fails clearly once generation is held fixed.**
- Given each image's **realized** 64-run classes, I computed E0(K) under each frozen rank law: the exact subset average of the Plackett–Luce first-place probability, with an interval from simulating the race on the realized labels.
- Pooled results:

  | Law | Gap to observed E0 |
  |---|---|
  | race (primary weights) | 0.6–1.4 points too high at every K (cohort A inside at 6 of 7 K; cohort B too optimistic) |
  | race_tau1 weights | within 0.3 points at every K |
  | S2 | 4–8 points too low at every K, outside its interval |
  | tail | matches DEV80 at K = 64 (0.875) |

- S2's E0 coverage therefore came from offsetting errors: too-pessimistic selection against too-optimistic generation.

**4. Exchangeability.** No Holm-significant class drift across runs 0–3, 4–15 and 16–63. The one borderline signal: the DEV80 GOOD rate falls from 48.4% to 44.2% (paired p = 0.056).

**5. The selection losses at K = 64** are 4 images, all in DEV80, each with only 1–4 GOOD runs among 64:
- two are won by global orbit errors, which E1 may repair;
- two are won by CONSISTENT runs;
- every margin is below 0.001σ.

## 3. Requests

**(a) Interpretation note, by Oct 30, 18:00 ET.** One page, under your locked plan, with no refit. It should state:
- the frozen-rule outcome;
- the generation versus selection decomposition;
- how this relates to C-T8: four runs cannot tell a hard image from a hopeless one.

I will fold it into Sec. 5.

**(b) Optional v3 for B16 (SITCOM) and F8 (DAPS).** Register by **Oct 31, 12:00 ET**; send the code by about 06:00 ET that day, so that I can QA and run it.
- **Same as v1:** estimand, panels, policies and comparison rule. v3 is reported beside v1, never instead of it.
- **Allowed inputs:**
  - A8 and A8b;
  - everything in this upload;
  - the D1 phase-2 comparison, once relayed.

  **Never** a SITCOM or DAPS FFHQ run beyond the historical prefix.
- **Design is yours.** What the data point to:
  1. **Generation:** a structural-zero component, i.e. images that never succeed. The same images recur across solvers, so each image's FAST 64-run outcome (for example "never GOOD", or its GOOD count) is a natural difficulty covariate for SITCOM and DAPS.
  2. **Selection:** the conditional test favors more competition than the primary weights imply. If you change the weights, fix the rule now and state it.
- **Code requirements:** it must run end to end on the real inputs (I will run it), with a contract that checks the oracle anchor, as in A9.5.
- **If you decline,** say so; B16 and F8 then unseal.

**(c) Optional supplement item: data-specific identification bounds** (C-T8 in the data).
- **Method.** As a planner prototype, I used an LP over mixing distributions on a grid of p, with the five four-run count frequencies relaxed by ±2 SE. It gives these identified sets for O_raw(K = 64):

  | Cohort | Identified set |
  |---|---|
  | B22 | about [0.92, 1.00] |
  | DEV80 | about [0.66, 1.00] |

  Both contain the forecast and the observation.
- **Result.** With the moment constraints imposed exactly, the LP is infeasible for DEV80 and gives an implausibly tight interval for B22, so a proper joint confidence region is needed.
- **Value.** Done properly, this turns "four runs cannot identify K = 64" into a number per cohort. It is lower priority than (a) and (b).

## 4. Paper deliverables (unchanged)

- **Due Nov 3–6:** Supplements S1 and S2, and about half a page for Sec. 5.
- **For Sec. 5:**
  - use the registered hashes;
  - report the miss as the result;
  - present the composition-conditional test as exploratory.

## 5. Reads and stops

- **You may read:**
  - `theory_inputs/`, including the FAST files once copied;
  - this upload;
  - the planner-branch status files, which are now open to you.
- **Still not `xp/returns/`:** the planner relays results.
- **Stop and ask** if v3 would need any B16 or F8 outcome.
