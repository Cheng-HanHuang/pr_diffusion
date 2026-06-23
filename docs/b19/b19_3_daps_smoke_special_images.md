# B19.3 DAPS smoke on special images

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

Caveat:

These are DAPS-generated measurements, not same-measurement comparisons with SITCOM.

## Results

| image | per-run PSNRs | mean PSNR | max PSNR | interpretation |
|---|---:|---:|---:|---|
| `00013` | `30.198, 30.024, 30.055, 29.900` | `30.045` | `30.198` | stable DAPS success |
| `00017` | `5.783, 32.763, 32.659, 5.872` | `19.269` | `32.763` | mixed population; good candidates exist |
| `00028` | `13.643, 15.093, 12.375, 12.891` | `13.501` | `15.093` | DAPS failure on the hard image |
| `00034` | `12.344, 32.099, 31.977, 11.939` | `22.090` | `32.099` | mixed population; good candidates exist |

## Interpretation

DAPS is not globally broken in this environment. It succeeds on `00013` and finds high-quality candidates on `00017` and `00034`.

However, DAPS fails on `00028`, with all four candidates around `12--15` dB. This supports the B19 diagnosis that `00028` is a shared hard image/measurement/prior case, not merely a SITCOM selector failure.

Next step:

Before running more solver sweeps, make the comparison measurement-locked: save/load the same phase-retrieval measurement `y` for SITCOM and DAPS.
