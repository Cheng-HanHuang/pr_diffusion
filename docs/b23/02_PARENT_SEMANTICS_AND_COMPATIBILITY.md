# Parent semantics and compatibility

## Frozen parent matrix

| Parent | Native source/state | Native coordinate | Measurement policy | Terminal candidates | B23.0 result |
|---|---|---|---|---:|---|
| Fresh1 | patched DAPS `posterior_sample.py`/`sampler.py`; `DAPSNativeState` | annealing sigma/index | exact raw locked tensor | 1 | parent frozen; donor unclassified |
| LF-v1 | same DAPS state with post-update LF magnitude intervention | annealing sigma/index | exact raw locked tensor | 1 | parent frozen; donor unclassified |
| NP-1 | B22 DiffFPR-guided proposal/ranking path; `NoisePickingNativeState` | diffusion timestep | verify raw tensor, then `clamp_min(0)` in memory | 1 | parent frozen; donor unclassified |
| SITCOM-1 | official SITCOM-ODE path; `SITCOMNativeState` | annealing sigma, diffusion substep, LGVD iteration | exact raw tensor, no clipping | 1 | parent frozen; donor unclassified |

`manifests/b23/parent_semantics.json` and the four `configs/b23/*_frozen.yaml` files are the
machine-readable source of truth.

## Fresh1

Fresh1 is the first trajectory from the frozen B21.11 DAPS protocol: FFHQ 256, oversampled Fourier
phase retrieval at `2.0`, `sigma_y=0.05`, `ffhq256ddpm`, `edm_daps`, `ann400`, `diff5`, LF off,
HIO off, batch one, one retained reconstruction.

The historical panel tag is `5401`; its measurement seed is SHA-256 over
`B21.5-fresh-measurement:{panel_seed}:{image_id}`, first eight bytes big-endian modulo `2^63-1`.
Historical first-trajectory seeds are `22000 + panel row_id`. New B23 seeds are not inferred from
that historical range: they must come from a signed future registry and the B23 named-stream rule.

Potential native boundaries are prior prediction, measurement-conditioned optimization,
post-update `x0y`, re-noising/transition, and trace/output. They are hypotheses about boundaries,
not donor eligibility. B23.1 replay must establish exact operation and state contracts.

## LF-v1

LF-v1 acts after the native DAPS measurement update on `x0y`. It pads and transforms `x0y`, keeps
its current Fourier phase, blends low-frequency magnitudes toward the locked measurement inside
radius fraction `0.12`, then crops the inverse transform. Frozen `alpha=0.50` decays linearly over
the first `0.35` of 400 annealing steps; ordinary DAPS follows.

This means “LF early then DAPS late” is the LF-v1 control, not a novel B23 composition. Patch default
values are not substituted for the frozen validated values. The semantic audit is
`docs/b21/b21_1_lf_patch_capture.md`.

## NP-1

NP-1 is not a cheap generic gradient. At each relevant timestep its native logic proposes five soft
noise candidates, denoises/scores them using the frozen low-frequency magnitude score, retains the
stable winner, and may apply late low-frequency magnitude projection beginning at step 300. Every
proposal, denoiser evaluation, score, projection, RNG draw, and discarded branch is charged.

Frozen values include 1000 diffusion steps, soft/hard candidates `5/1`, score radius `0.6`,
projection radius `0.2`, `300:0.2`, LF score, seed 100, and in-memory nonnegative measurement
clipping. NP-1 uses the clean DiffFPR checkout at
`a45ffe58f18fed8a63d3446600424e2b08733524` and the B22 repository entrypoint.

## SITCOM-1

SITCOM-1 uses the official implementation at
`275ab67efbd8146bffca20155171ba6be1169c09`, not
`prdiffusion/algorithms/hybrid_np_sitcom.py`. It couples LGVD optimization of the model input,
network gradients, measurement consistency, forward noising, and its annealing schedule.

Frozen values are 200 linear annealing steps from sigma 100 to 0.1, five linear diffusion substeps
with minimum 0.01, 100 LGVD steps, learning rate `5e-5`, minimum ratio `0.01`, `tau=0.01`, poly-7
timestep, seed 43, one retained trajectory. The only intended tracked text difference is robust
recursive output-directory creation. Bytecode/cache changes are evidence noise, not parent logic.

A “late SITCOM polish” is only a conjectured truncation. Unless native replay, explicit adapter,
state validity, information accounting, RNG reconciliation, and module replay all pass, SITCOM is
`BASELINE-ONLY`.

## Compatibility outcomes

Post-B23.1 classifications are limited to:

- `NATIVE-REPLAYED`
- `DAPS-NATIVE-DONOR`
- `ADAPTER-QUALIFIED-DONOR`
- `BASELINE-ONLY`
- `REJECTED-PROTOTYPE`

B23.0 assigns none of them beyond “native parent frozen.” Cross-family Track A requires at least
one NP or SITCOM operation to become `ADAPTER-QUALIFIED-DONOR`; forcing a lossy adapter to make H0
pass is prohibited.
