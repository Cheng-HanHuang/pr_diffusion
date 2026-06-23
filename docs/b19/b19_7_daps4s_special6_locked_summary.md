# B19.7 DAPS4S exact-loss selection on locked special-six measurements

## Purpose

Evaluate whether DAPS with four independent candidates, selected by exact DAPS phase-retrieval operator loss, gives an executable clean-free solver on the special-six images.

Images:

- `00013`
- `00017`
- `00018`
- `00027`
- `00028`
- `00034`

Measurement seeds:

- `3000`
- `3001`
- `3002`

Each case uses a locked phase-retrieval measurement, then runs DAPS4S and selects the candidate with minimum exact DAPS operator loss.

## Top-line result

| metric | value |
|---|---:|
| cases | 18 |
| mean selected PSNR | 30.930 |
| min selected PSNR | 29.581 |
| mean oracle PSNR | 30.979 |
| max gap to oracle | 0.252 dB |
| contains 30 dB candidate | 14 / 18 |
| selected 30 dB candidate | 11 / 18 |
| selected bad25 | 0 / 18 |
| selected bad20 | 0 / 18 |

## Interpretation

The strict 30 dB threshold is not the best success criterion here, because some cases have oracle PSNR below or barely above 30 dB.  The clearest example is `00027`, where the oracle DAPS4S candidate is only around `29.66--29.74` dB, and the selected candidate is essentially oracle-quality.

The main conclusion is that DAPS4S plus exact operator-loss selection is near-oracle on this special-six locked-measurement audit.  No selected candidate is catastrophic, the minimum selected PSNR is about `29.58` dB, and the maximum selected-vs-oracle gap is only about `0.252` dB.

This changes the Branch B direction:

- SITCOM has a candidate-generation ceiling on hard locked `00028`.
- DAPS has better access to high-quality reconstruction basins.
- Exact DAPS operator loss is a usable clean-free selector for DAPS candidates on this audit.

## Per-image summary

| image | cases | mean selected | min selected | selected 30 | contains 30 | mean gap | max gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00013` | 3 | 29.922 | 29.858 | 1 | 2 | 0.081 | 0.244 |
| `00017` | 3 | 32.833 | 32.803 | 3 | 3 | 0.000 | 0.000 |
| `00018` | 3 | 29.970 | 29.934 | 1 | 3 | 0.148 | 0.252 |
| `00027` | 3 | 29.660 | 29.581 | 0 | 0 | 0.028 | 0.084 |
| `00028` | 3 | 31.081 | 30.952 | 3 | 3 | 0.024 | 0.073 |
| `00034` | 3 | 32.114 | 32.058 | 3 | 3 | 0.011 | 0.032 |
