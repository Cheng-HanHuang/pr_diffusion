# B21.5 frozen three-arm fresh validation plan

Status: runner and analyzer ready; fresh GPU validation not yet executed.

## Development result motivating validation

On the 40-case development panel, the frozen sequential selector achieved:

```text
base+LF gated: 17/40 good25
base+LF+HIO:   22/40 good25
incremental HIO rescues: 5
incremental HIO harms:   0
HELDOUT4 net beyond base+LF: +4
```

This supports one fresh validation. It does not authorize any retuning.

## Fresh panel

Use ten deterministic, previously untouched IDs from the official FFHQ validation range `60000--69999`. The list is the first ten SHA-256-ranked IDs under salt `B21.5-three-arm-fresh-v1`:

```text
68263 63803 66452 63282 66892
69293 68924 65808 62802 65960
```

Each image receives eight paired cases, for 80 total cases.

Fresh seeds:

```text
measurement panel seed/tag: 5101
base seeds:                 10000--10079
HIO seeds:                  11000--11079
warm DAPS noise seeds:      12000--12079
```

Measurements are generated once with the local DAPS preprocessing and operator:

```text
ToTensor -> Resize(256) -> CenterCrop(256) -> [-1,1]
PhaseRetrieval(oversample=2.0, sigma=0.05).measure(x)
```

A deterministic per-image Gaussian-noise seed is derived from the panel seed and image ID. Saved tensors and hashes are recorded before solver execution.

## Frozen candidates

For every case:

1. `base_full`: ann400, diff5, full DAPS.
2. `lf050`: same initial/base seed, ann400, diff5, `alpha=0.50`, `frac=0.35`, `radius_frac=0.12`.
3. `hio_warm`: 240 HIO iterations, beta `0.9`, ER every `20`, final ER `10`, injected at global DAPS step `200`, then DAPS steps 200--399.

## Frozen clean-free selector

Use exact operator loss only:

1. Select LF iff `loss_lf < loss_base - 0.7`.
2. Select HIO iff `loss_hio < loss_current - 0.7`.

No threshold, HIO, LF, schedule, or injection-step tuning is permitted after seeing the fresh results.

## Fresh-validation gates

The three-arm method passes only when all hold:

- all 80 cases complete with valid HIO states;
- incremental three-arm net beyond base+LF is at least `+6/80`;
- incremental HIO harms are at most `2`;
- at least `5/10` images have positive incremental net and at most `2/10` are negative;
- the clean-free selector captures at least `80%` of oracle incremental HIO successes;
- marginal HIO cost is at most `0.70x` one full base run;
- total three-arm cost is at most `1.75x` one full base run.

A pass freezes the policy and promotes it to a larger disjoint official-validation benchmark. A failure retires HIO and retains the frozen base+LF portfolio.
