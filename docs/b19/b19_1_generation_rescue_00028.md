# B19.1/B19.2 generation rescue: image 00028 seed 19053

Case:

- image: `00028`
- noise: `0.05`
- root: `external/sitcom_ode_npsitcom`
- original special-six seed: `19053`

## Multiseed regime audit

Across the special-six multiseed audit, the tau-window selector produced no selector failures.

Summary:

- solved oracle4-good + selector-good cases: 23
- selector failures: 0
- oracle4 generation failures: 1

The only oracle4 generation failure was:

- seed `19053`
- image `00028`
- oracle4 PSNR: `12.780699`
- tau-window selected PSNR: `12.364407`

## 12S rescue

Output:

`/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver/B19_1_generation_rescue_00028_seed19053_12S_npsitcomroot`

Result:

- oracle12 best PSNR: `27.655651`
- tau-window selected PSNR: `27.632565`
- bad25 after selection: `0`
- bad20 after selection: `0`

Interpretation:

Increasing the SITCOM population from 4 to 12 partially rescues the case, but it does not recover the usual 30+ dB regime observed for easier seeds/images.  This suggests a hard-generation or low-ceiling SITCOM case rather than a simple selector failure.

## diff_steps sweep

For 4S at seed `19053`, increasing `diff_steps` gives:

| diff_steps | mean PSNR | max PSNR | min PSNR | bad25 | bad20 |
|---:|---:|---:|---:|---:|---:|
| 5 | 17.077 | 27.633 | 12.136 | 3 | 3 |
| 10 | 18.410 | 26.578 | 14.153 | 3 | 3 |
| 20 | 18.534 | 25.849 | 15.412 | 3 | 3 |

Interpretation:

More SITCOM inner refinement improves the floor but lowers the best candidate.  Therefore simply increasing `diff_steps` is not a promising path to the missing 30+ dB reconstruction.  The next diagnostic should test a decoupled/exploratory method such as DAPS, or a separate measurement-refinement branch, on this exact case.
