# B19.3 DAPS smoke on special images

Purpose:

Test whether official DAPS, using its default FFHQ pixel-DDPM phase-retrieval configuration, can rescue or reproduce the hard/special B19 images.

Setup:

- DAPS root: `external/daps`
- env: `/egr/research-pac/huang248/conda-envs/daps`
- model: `ffhq256ddpm`
- task: `phase_retrieval`
- sampler: `edm_daps`
- task group: `pixel`
- num runs: `4`
- annealing steps: `200`
- diffusion steps: `5`
- operator noise: `0.05`
- DAPS seed: `42`

Important caveat:

These are DAPS-generated measurements, not yet same-measurement comparisons with the SITCOM B19 runs.

## Results

| image | per-run PSNRs | mean PSNR | max PSNR | interpretation |
|---|---:|---:|---:|---|
| `00013` | `30.198, 30.024, 30.055, 29.900` | `30.045` | `30.198` | stable DAPS success |
| `00017` | `5.783, 32.763, 32.659, 5.872` | `19.269` | `32.763` | mixed population; good candidates exist |
| `00028` | `13.643, 15.093, 12.375, 12.891` | `13.501` | `15.093` | DAPS failure on the hard image |
| `00034` | `12.344, 32.099, 31.977, 11.939` | `22.090` | `32.099` | mixed population; good candidates exist |

## Interpretation

DAPS is not globally broken in this environment.  It gives stable good reconstructions on `00013` and finds high-quality candidates on mixed-population images `00017` and `00034`.

However, DAPS also fails on image `00028`: all four candidates stay around `12--15` dB.  This supports the current B19 diagnosis that `00028` is a shared hard image/measurement/prior case, not just a SITCOM selector failure.

Compared with SITCOM:

- SITCOM 4S on `00028 / seed19053` had oracle4 around `12.78` dB.
- SITCOM 12S partially rescued the case to around `27.66` dB.
- DAPS 4-run smoke on `00028` did not rescue it, reaching only around `15.09` dB max under its own generated measurement.

Next steps:

1. Use DAPS metrics as evidence that DAPS works on easier/mixed images.
2. Do not spend effort patching DAPS for same-measurement testing unless needed for a final fair comparison.
3. Focus B19 next on clean-free detection of hard low-ceiling cases and conditional escalation, because both SITCOM and DAPS can fail on `00028`.
