# B21.5 HIO warm-start five-image pilot plan

Status: runner and analyzer ready; GPU experiment not yet executed.

## Motivation

The three-case image-00046 smoke passed its implementation, cost, and calibration quality gates, but the mean gain was dominated by one large rescue. This pilot tests whether the frozen HIO warm-start generalizes across distinct failure modes.

## Frozen panel

```text
images: 00046 00171 00224 00746 00971
cases per image: 8
cases total: 40
measurement seed: 5001
base seeds: 7400--7439
HIO seeds: 8400--8439
warm DAPS noise seeds: 9400--9439
HIO: 240 iterations, beta 0.9, ER every 20, final ER 10
DAPS: ann400, diff5
warm injection step: 200
```

Image `00046` is the smoke/tuning image. The other four images are reported separately as `HELDOUT4`.

## Comparison

For every case:

- `base_full`: one ordinary full ann400 DAPS run;
- `hio_warm`: one clean-free HIO generation followed by DAPS transitions 200--399.

The HIO generator sees only the locked Fourier-magnitude measurement and known support. Ground truth is used only by the offline analyzer.

## Primary registered gate

The registry criterion for `WARM_hio` is retained:

```text
pooled warm good25 rate - pooled base good25 rate >= 0.10
pooled (HIO generation + warm DAPS) / base DAPS wall ratio <= 0.70
```

With 40 cases, the rate criterion requires at least four net additional good25 cases.

## Generalization safeguard

Because image `00046` was used for the smoke, promotion to broader validation additionally requires the held-out four-image panel to have nonnegative net good25.

## Secondary diagnostics

Report by image and overall:

- base and warm good25 counts;
- warm-only and base-only wins;
- exact paired McNemar p-value;
- mean and median PSNR difference;
- PSNR win/loss counts and exact sign-test p-value;
- wall-time ratio;
- raw-HIO PSNR and measurement residual.

No HIO hyperparameter or injection-step tuning is permitted after seeing this pilot. A failed gate rejects this frozen configuration; a passed gate promotes it to a broader fresh validation.
