# B22.3 scientific closeout: fixed-baseline evaluation

## Decision

**B22 fixed-baseline evaluation is scientifically complete.**

The validated 100-image panel, deterministic CPU analysis, and manual visual review support a stable three-way conclusion:

1. **Fresh2 is the central-quality leader.** It has the highest executable mean and median raw PSNR and wins raw PSNR against NP-8-RS on 91 of 100 images.
2. **NP-8-RS is the executable reliability and lower-tail leader.** It reaches 95/100 raw good25, removes all sub-10 dB failures, and has the strongest executable q05, at the highest compute cost.
3. **SITCOM-4S is a lower-cost stabilizer.** It reaches 93/100 raw good25 at substantially lower GPU cost than Fresh2 or NP-8-RS, but its ordinary-image PSNR is lower.

No executable policy dominates quality, reliability, and cost simultaneously.

```text
B22.0 inventory: SIGNED OFF
B22.1 reproducibility smoke: SIGNED OFF
B22.2 100-image execution: VALIDATED
B22.3 numerical analysis: SIGNED OFF
B22.3 visual atlas: SIGNED OFF
B22 fixed-baseline study: COMPLETE
New GPU experiments inside B22: NOT AUTHORIZED
Cross-method selector tuning on this panel: PROHIBITED
```

## Validated executable results

| Policy | Mean raw PSNR | Median | Min | q05 | raw good25 | raw bad20 | Mean GPU-s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh1 | 27.232 | 30.605 | 6.482 | 9.614 | 80/100 | 18 | 156.1 |
| Fresh2 | **29.299** | **30.899** | 6.360 | 15.860 | 92/100 | 8 | 312.6 |
| SITCOM-1 | 22.972 | 27.038 | 6.699 | 7.791 | 71/100 | 25 | 49.3 |
| SITCOM-4S | 26.442 | 27.166 | 6.428 | 23.415 | 93/100 | 4 | 196.1 |
| NP-1 | 25.585 | 29.390 | 6.062 | 10.435 | 75/100 | 24 | 52.7 |
| NP-8-RS | 29.269 | 29.941 | **10.995** | **25.146** | **95/100** | **3** | 411.8 |

The 3-point raw-good25 advantage of NP-8-RS over Fresh2 is not statistically decisive on 100 paired images: NP-8-RS rescues seven Fresh2 threshold failures and harms four, with exact McNemar p = 0.5488. The scientifically stronger distinction is distributional:

- Fresh2 gives higher raw PSNR on 91/100 images;
- NP-8-RS prevents several rare catastrophic Fresh2 failures;
- the mean raw-PSNR difference is effectively zero: NP-8-RS minus Fresh2 = -0.029 dB, image-bootstrap 95% CI [-1.018, +1.129].

Therefore NP-8-RS should not be described as a universally better reconstruction method. It is a more expensive tail-risk policy.

## Multi-run value

Each frozen population policy materially improves its corresponding one-run method:

| Comparison | Mean PSNR gain | good25 rescues / harms | Interpretation |
|---|---:|---:|---|
| Fresh2 vs Fresh1 | +2.067 dB | 12 / 0 | Frozen two-run DAPS selector is strongly supported |
| SITCOM-4S vs SITCOM-1 | +3.470 dB | 22 / 0 | Four-run population substantially stabilizes SITCOM |
| NP-8-RS vs NP-1 | +3.684 dB | 20 / 0 | Eight-run residual selection substantially stabilizes NP |

This is a robust result: all three population policies improve threshold reliability without harming a previously good one-run case at the 25 dB threshold.

## Selector versus candidate generation

### SITCOM

- selected good25: 93/100;
- oracle4 good25: 95/100;
- exact oracle PSNR matches: 39/100;
- selected-bad/oracle-good cases: two.

Visual review separates these two misses:

- `60140`: severe 180-degree orientation selector miss; the selected candidate is upside-down while oracle4 is correctly oriented and good;
- `64518`: marginal threshold miss, 24.80 versus 25.01 dB, with visually similar outputs.

The remaining SITCOM-4S failures are candidate-generation, orientation-resolution, or quality-floor failures rather than selector errors.

### NP

- selected good25: 95/100;
- oracle8 good25: 96/100;
- exact oracle PSNR matches: 26/100;
- selected-bad/oracle-good cases: one.

The single threshold selector miss is:

- `65269`: NP-8-RS selects a blurrier 24.37 dB candidate while oracle8 reaches 27.75 dB.

The other NP-8-RS failures lack a good candidate in the frozen eight-run population.

The pooled selector statistics correlate strongly with candidate PSNR in the expected direction:

- SITCOM correction norm versus PSNR: Spearman -0.713;
- NP residual statistic versus PSNR: Spearman -0.736.

These correlations support the selectors as useful population-level heuristics, but the isolated misses show that neither statistic is a certificate of reconstruction quality.

## Visual failure taxonomy

The 16-image union of executable multi-run failures was inspected in full. Detailed labels are stored in `docs/b22/b22_3_visual_failure_taxonomy.csv`.

### Fresh2: exact confirmation of the frozen B21.12 taxonomy

Fresh2 has eight raw failures:

- **three 180-degree orientation failures:** `62908`, `66715`, `68539`;
- **two chromatic/illumination overlay failures:** `65553`, `66889`;
- **two structured twin/ghost failures:** `65365`, `67293`;
- **one high-complexity shared prior collapse:** `65003`.

This exactly reproduces the B21.12 conclusion: three rot180-resolvable failures and five persistent failures split into two chromatic/illumination, two twin/ghost, and one high-complexity collapse. B22 therefore validates rather than revises the B21 failure anatomy.

### NP-8-RS

NP-8-RS has five raw failures:

- `61252`: structurally correct but blurred/noisy quality floor across the population;
- `61669`: circular structured-twin basin collapse across the population;
- `65003`: shared high-complexity collapse;
- `65269`: selector miss with a good oracle candidate;
- `67520`: structurally correct but blurred/noisy quality floor across the population.

Thus one failure is selector-level and four are candidate-generation or quality-floor failures.

### SITCOM-4S

SITCOM-4S has seven raw failures:

- `60140`: severe orientation selector miss;
- `64518`: marginal threshold selector miss;
- `62739`: borderline quality-floor case with no good SITCOM candidate;
- `62908`: raw orientation ambiguity; a rotated candidate is visually correct;
- `65003`: shared high-complexity collapse;
- `66591`: structured twin/ghost population failure;
- `67520`: blurred/noisy quality floor.

Thus two are selector misses and five are population-generation, symmetry-resolution, or quality-floor failures.

## Cross-method complementarity

The diagnostic best-of-{Fresh2, SITCOM-4S, NP-8-RS} oracle reaches 99/100 raw good25.

- Fresh2 is the oracle winner on 91 images.
- NP-8-RS is the oracle winner on nine images.
- SITCOM-4S is never the highest-PSNR method on this panel, although it provides a cheaper reliability point and rescues several Fresh2 failures.
- `65003` is the only failure shared by all three methods.

The NP-8-RS wins are disproportionately important because several are large rescues of Fresh2 catastrophes: `62908`, `65365`, `65553`, `66715`, `66889`, `67293`, and `68539`.

This establishes a real cross-method routing opportunity, but not an executable routing policy. Any rule fitted to these 100 final images would invalidate prospective evaluation.

## Shared hard case `65003`

Image `65003` contains unusually complex, colorful, high-frequency clothing and head ornamentation. Visual inspection shows:

- Fresh2 preserves part of the face and silhouette but remains dominated by severe structured/chromatic superposition;
- SITCOM candidates collapse to diffuse multicolor twin/ghost states;
- NP candidates show severe circular-boundary and structured ghost artifacts;
- all corresponding oracles remain below 25 dB.

This is a shared candidate-generation and prior-basin failure, not a selector failure. It is the strongest evidence that the current prior/solver family has a genuine hard-image regime that cannot be repaired merely by selecting among the present candidates.

## Reporting recommendation

The primary B22 story should be reported as a **quality–reliability–cost frontier**:

- **Fresh2:** best central reconstruction quality;
- **SITCOM-4S:** cheaper reliability stabilizer;
- **NP-8-RS:** strongest executable lower-tail reliability at highest cost.

The paper/report should avoid:

- declaring a universal winner from 92%, 93%, and 95% good25;
- treating raw residual/correction statistics as certificates;
- presenting any ground-truth oracle as executable;
- tuning a cross-method selector on the final panel;
- obscuring the large compute difference between the population policies.

The main positive system insight is:

> Independent solver families fail in visibly different basins, and a high-cost NP population can remove most catastrophic DAPS failures even though DAPS remains better on ordinary images.

## Next research decision

B22 should now be closed rather than extended with more panel-specific experiments.

A future integration project is justified only under a new protocol:

1. develop clean-free routing or confidence features using separate historical/development images;
2. freeze the router and all thresholds before evaluation;
3. construct a new untouched locked image panel;
4. evaluate the router prospectively against Fresh2, SITCOM-4S, NP-8-RS, and fixed-cost portfolios;
5. retain `65003`-type shared candidate-generation failure as a distinct target from cross-method selection.

That future project should be treated as a new checkpoint/project, not as further tuning of B22.

## Minor atlas metadata note

The returned `atlas_index.csv` records panel paths under `failure_atlas.tmp` because paths were serialized before the atomic directory rename. The actual panels are present under `failure_atlas/panels/`, and the contact sheets and manual review are valid. This is a path-metadata usability defect only; it does not affect images, metrics, or conclusions. Future atlas generation should serialize final post-rename paths.
