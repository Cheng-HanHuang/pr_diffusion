# B19.0 SITCOM special-six initial result

Run:

`/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver/B19_0_sitcom_special6_noise005_npsitcomroot`

Setup:

- SITCOM root: `external/sitcom_ode_npsitcom`
- images: `00017,00027,00013,00028,00034,00018`
- noise: `0.05`
- runs per image: `4`
- anneal steps: `200`
- diffusion steps: `5`
- seed: `19050`

Observed final PSNR pattern:

- `00013`: all four runs good, around 29.3--29.9 dB.
- `00018`: all four runs good, around 29.5--29.8 dB.
- `00027`: three good runs around 29.0 dB and one medium-bad run around 24.4 dB.
- `00017`: one good run around 32.0 dB and three catastrophic runs around 5.8 dB.
- `00028`: one good run around 30.3 dB and three bad runs around 12--17 dB.
- `00034`: one good run around 31.3 dB and three bad runs around 11.9 dB.

Interpretation:

This run is mainly a within-population selector/certificate testbed.  There is no all-four SITCOM generation failure in this seed, because each difficult image has at least one good candidate.  Simple residual-like features are suspicious because the good candidate can have larger residual values than the bad candidates, especially on image `00028`.

Next analysis:

Run `scripts/b19/analyze_sitcom_tau_selectors.py` to evaluate tau=0.8 correction and residual selectors.

## Tau=0.8 selector analysis

Analysis script:

`scripts/b19/analyze_sitcom_tau_selectors.py`

Output folder:

`/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver/B19_0_sitcom_special6_noise005_npsitcomroot/tau08_selector_analysis`

At tau=0.8, the executable clean-free selectors successfully chose good candidates for all six images.

Summary:

| selection method | n images | mean PSNR | min PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|---:|
| oracle_best_psnr_diagnostic | 6 | 30.393 | 29.151 | 0 | 0 |
| min_x0y_full_residual_tau | 6 | 30.360 | 29.141 | 0 | 0 |
| min_x0y_lowfreq_residual_tau | 6 | 30.360 | 29.141 | 0 | 0 |
| min_correction_tau | 6 | 30.326 | 29.023 | 0 | 0 |
| min_x0hat_x0y_disagreement_tau | 6 | 30.326 | 29.023 | 0 | 0 |
| max_x0y_full_residual_tau | 6 | 19.121 | 5.824 | 4 | 3 |

Interpretation:

This special-six run is a strong within-population selection test.  Images `00017`, `00028`, and `00034` each have one good candidate and three bad/catastrophic candidates, yet the tau=0.8 minimum-correction and minimum-residual selectors recover good candidates.  The deliberately wrong `max_x0y_full_residual_tau` selector fails badly, confirming that the features are meaningful rather than the candidate set being trivially all-good.
