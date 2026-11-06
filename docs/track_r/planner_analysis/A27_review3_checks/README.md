# A27: planner checks prompted by the third AI review (2026-11-06)

All checks are CPU only, run on committed data, exploratory and post hoc. No new run was drawn. Ledger: C-R8, C-V4, C-V5.

## Inputs

**Forecast interval scores (C-R8).** `code/forecast_interval_scores.py <dir> <out.json>` reads the same six comparison JSONs as `fig/make_fig_forecast_calibration.py` (sources are listed in that script's docstring). It keeps both registered E1 decision models, so the counts match Tab. 2 (111 selected-run forecasts).
- Interval score: the Gneiting–Raftery score at α = 0.05.
- Absolute error is taken against the point forecast.

**Protocol sweeps (C-V4).** These use the frozen protocol code, `docs/track_r/protocol/gtfree_protocol.py` at planner `c598176` (SHA `d5f973fe…`), unmodified, and these inputs:
- **DAPS-4K:** CP3 G4/G5 strings and CLIP decisions, XP `1981dea` (`theory_inputs/XP_A17_CP3_20261031T173255Z/`).
- **DAPS-2K:** F8 strings and face decisions, XP branch `prdiff-track-r-xp-a13-cp1-20261030T010953Z` (`theory_inputs/XP_A13_F8_8RUN_*`).
- **SITCOM:** B16 strings and face decisions, XP branch `prdiff-track-r-xp-a11-b16-return-20261029`.

The scripts expect the inputs staged as `in/<G4|G5|F8|B16>_{STRINGS,CLIP}.csv` next to a copy of the frozen code, with an `out/` folder.

**Reproduction checks.**
- G5 reproduces the registered output exactly: 0.638853 at 2.40444 runs, gain +0.001043.
- G4 per cohort reproduces the registered per-cohort outputs: B22 0.978875 at 2.492, DEV80 0.8905 at 3.192.
- F8 per cohort with the labeled τ reproduces the registered weighted gain, +3.38.

## Findings

**Forecasts.** Over all six tests, the race forecasts the selected run with:
- a mean interval score of 30.3 points, against 181.8 for S2;
- a mean absolute error of 3.1 points, against 8.0;
- coverage of 76/111, against 53/111.

The race loses test 1 clearly (60.9 against 11.4) and test 4 narrowly (9.5 against 7.8). For the oracles, the two models share point forecasts; their interval scores are 47.8 (race) and 44.8 (S2).

**Threshold.** The DAPS stopping gains need τ_S within about 1%.
- At ×0.98, no run passes: the rule never stops (8.0 runs) and flags every image.
- At ×1.02, the rule stops at M runs and keeps +0.7 to +1.9 points.
- The labeled, label-free and cross-cohort thresholds lie within 0.33% of each other and give +2.6 to +5.0 points.

**Target M (held-out panels).** On DAPS, M = 1 gives +7.2 to +11.2 points and M = 4 gives +0.5 to +1.9. On SITCOM and DAPS-4K ImageNet, every M gives about 0.

**Pass probability s against class probability κ (C-V5).**
- SITCOM, 16 runs: 82 of 1,688 GOOD runs and 11 of 94 orbit errors fail τ_S.
- DAPS-4K with the label-free τ_S: 40 of 350 INCONSISTENT runs pass on FFHQ, and 139 of 210 on ImageNet.
