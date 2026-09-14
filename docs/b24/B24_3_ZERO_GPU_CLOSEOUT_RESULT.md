# B24.3 zero-GPU development closeout result

## Status

The authorized zero-GPU development closeout completed successfully on 2026-09-14 from B24 head:

`c3f13963cd267094231302bf9bc8d3a6e8c754c9`

PAC run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

The launcher reported:

- `B24_3_CLOSEOUT_SOURCE_GATE_PASS`;
- `B24_3_ZERO_GPU_CLOSEOUT_TESTS_PASS`;
- `B24_3_ZERO_GPU_ENV_PASS`;
- `B24_3_ZERO_GPU_CLOSEOUT_GATE_PASS`;
- `B24_3_ZERO_GPU_CLOSEOUT_COMPLETE`.

Scope guards remained intact:

- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `confirmation_exposed=false`.

The binding development decision remains:

`STOP_B24_METHOD_REFINEMENT`

This closeout does not reopen method development and does not authorize confirmation exposure.

## Historical Fresh2 derivation

Fresh2 was derived exactly from stored DEV80 DAPS reps 0 and 1 using the pre-existing clean-free rule:

`choose rep1 iff exact_loss(rep1) < exact_loss(rep0) - 0.7; otherwise rep0`.

Ground truth was not used for selection.

DEV80 results:

- `FRESH2_SELECTED`: Good25 `57/80` (`0.7125`), mean PSNR `26.1596 dB`, median `29.9255 dB`;
- `DAPS2_ORACLE`: Good25 `59/80`, mean `26.8807 dB`, median `30.2805 dB`;
- Fresh2 selector gap to the two-trajectory oracle: mean `0.7212 dB`, median `0 dB`;
- Fresh2 accepted rep1 on `29/80` images.

Against DAPS1, Fresh2 had mean paired delta `+2.8692 dB`, Good25 rescues/harms `10/0`, and >=5 dB rescues/harms `14/0`.

Against executable NP4 selection, Fresh2 had mean paired delta `+0.4249 dB`, median `+1.0939 dB`, Good25 rescues/harms `14/12`, >=5 dB rescues/harms `14/13`, and PSNR wins/ties/losses `63/0/17`.

Against the DAPS4 oracle ceiling, Fresh2 remained below by mean `-2.6671 dB` and median `-0.0474 dB`, with Good25 harms/rescues `11/0`.

Fresh2 is historical project methodology, not a newly fit B24 method. These DEV80 results are therefore descriptive closeout evidence, not a new B24 advancement decision.

## Hard-subset complementarity

The fresh-both-baseline-oracles-fail subset contained `10` DEV80 images.

Unique Good25 successes within that subset were:

- `NP4_SELECTED`: `2`;
- `EPP321_SELECTED`: `2`;
- `FRESH2_SELECTED`: `1`;
- `PE3_SCORE_SELECTED`: `0`;
- `PE3_RANDOM_SELECTED`: `0`;
- `DAPS1`: `0`;
- `SITCOM1`: `0`.

This supports the narrow interpretation that the methods can reach different basins, while the prospectively frozen PE3 gate still failed to establish a robust NP-native improvement over independent NP4.

## Compute closeout

Dispatch-supported dynamic FLOPs from the development-only audit were retained with explicit guards against treating unsupported FFT/custom work as zero:

- DAPS1: `840267171895544`;
- DAPS2 / Fresh2 two-trajectory equivalent: `1680534343791088`;
- DAPS4 independent equivalent: `3361068687582176`;
- PE3_SCORE: `3413820886220800`;
- SITCOM1: `387934191616000`;
- SITCOM4 independent equivalent: `1551736766464000`.

Ratios to PE3_SCORE in the dispatch-supported metric:

- DAPS1: `0.2461`;
- DAPS2 / Fresh2: `0.4923`;
- DAPS4: `0.9845`;
- SITCOM1: `0.1136`;
- SITCOM4: `0.4545`.

No exact total-FLOP equivalence is claimed. The closeout explicitly records nonzero unsupported Fourier/custom work, including DAPS phase-retrieval FFT structure and SITCOM's `20000` autograd backward calls. For NP PE3_SCORE, the static inventory records `29184` forward complex 2-D FFT calls plus `2796` inverse calls under the frozen accounting derivation.

## Closeout artifacts

The PAC closeout emitted:

- `FRESH2_PER_IMAGE.csv`;
- `FRESH2_SUMMARY.json`;
- `DEV80_CLOSEOUT_PER_IMAGE.csv`;
- `PAIRWISE_EXECUTABLE.csv`;
- `HARD_SUBSET.csv`;
- `COMPLEMENTARITY.json`;
- `COMPUTE_CLOSEOUT.json`;
- `B24_3_DEV_CLOSEOUT.md`;
- `B24_3_DEV_CLOSEOUT.json`;
- `SHA256SUMS.txt`;
- `LAUNCH_IDENTITY.txt`;
- `.tar.gz` archive plus `.sha256` sidecar.

Archive path:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z.tar.gz`

## Planner-facing conclusion

B24 development should remain closed under the frozen rule. NP-native pruning/reallocation and the final protected-explorer variants did not robustly improve over independent NP4 at matched NP UNet work. Historical Fresh2 is a strong descriptive comparator on DEV80 and is notably competitive with NP4 at substantially lower dispatch-supported compute, but it is not a new B24 method and the present closeout does not convert it into a post-hoc advancement candidate.

Confirmation remains locked unless the planner separately authorizes a new project direction.
