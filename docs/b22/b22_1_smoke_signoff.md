# B22.1 reproducible one-image smoke sign-off

## Decision

**B22.1 is signed off.**

The recovered one-image fixed-baseline smoke passed the independent validator for
both `SITCOM-1` and `NP-1`. The full 100-image baseline implementation may now be
prepared, but no full-panel GPU launch is authorized until that implementation,
its sharding/accounting contract, and its dry-run validation are reviewed.

```text
B22.0 inventory: SIGNED OFF
B22.1 one-image smoke: SIGNED OFF
B22.2 full-panel implementation: AUTHORIZED
Full 100-image GPU launch: BLOCKED pending B22.2 implementation review
```

## Returned recovery archive

```text
B22_1_smoke_20260727_142201_npjson_recovery_20260727_144738.tar.gz
```

PAC run root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/
B22_1_smoke_20260727_142201
```

Recovery repository head:

```text
aa48845bb822e94c25fa3af92a141aa07cb24f11
```

The input manifest records the original execution head
`6852c84132159028f1a8104e2e9ed56db71ae7ee`. The recovery changed only artifact
serialization and validator tolerance/reporting; the exact NP reconstruction
content hash matched the first attempt, establishing that no method result
changed.

## Nondiscretionary smoke input

```text
selection rule: lexicographically_first_locked_measurement
selection used method outcome: false
image_id: 60044
measurement: ffhq60044_phase_noise005_meas5401.pt
measurement file SHA-256:
98ad24d664df0366e81a8b9f5418ff0cb7c14fb967d92bb40d72cbebfdc344de
measurement tensor-content SHA-256:
877ab671eb3fc53d4f8311dee9a424d2acb7f9e4927eb9c90cf32b8ab06e2f93
model SHA-256:
81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa
```

The locked measurement has shape `(1,3,384,384)`, dtype `float32`, and 157,327
negative entries. SITCOM consumed it raw. NP used the frozen in-memory
`clamp_min(0)` preprocessing, producing clipped-measurement content SHA-256:

```text
26d521a5ae28163598d5c9828803eb369ff1f1943265834dca1b4e6df26552ba
```

## Validation result

`validation.json` reports `PASS` with every required check true:

- locked measurement identity;
- shared ground-truth identity;
- finite outputs;
- independent offline metric recomputation;
- runtime recording;
- GPU-memory recording;
- exact frozen configurations;
- raw locked measurement for SITCOM;
- in-memory clipping for NP.

The metric comparison tolerance is `1e-5 dB`, which is strict relative to the
observed CPU/GPU reduction drift and does not alter reported scores.

## Results

| method | raw PSNR | rot180 PSNR | ambiguity-aware PSNR | reconstruction time | total observed time | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| SITCOM-1 | 26.8377895 dB | 8.3543310 dB | 26.8377895 dB | 89.7891 s | 90.9834 s | 664,651,776 B | 975,175,680 B |
| NP-1 | 29.3942223 dB | 8.3695459 dB | 29.3942223 dB | 52.4152 s | 53.4828 s | 674,237,952 B | 979,369,984 B |

Both outputs pass raw and ambiguity-aware `good25` on this smoke image.

The observed single-image difference is:

```text
NP-1 minus SITCOM-1 raw PSNR: +2.5564327 dB
NP-1 reconstruction-time ratio relative to SITCOM-1: 0.58375
```

These are smoke observations only and must not be generalized to the 100-image
panel before the paired benchmark is complete.

## Exact replay integrity

First NP attempt reconstruction tensor-content SHA-256:

```text
b6dee35138e453fc8a1c77aa1dc3331d1b70bce8fa75a10d66fc39d5fd836641
```

Recovered NP reconstruction tensor-content SHA-256:

```text
b6dee35138e453fc8a1c77aa1dc3331d1b70bce8fa75a10d66fc39d5fd836641
```

The replay is exact. The undefined post-projection candidate-margin diagnostic
is now represented as JSON `null` and listed explicitly under
`undefined_diagnostics`. This is correct because the frozen post-projection
configuration has one hard candidate.

## Visual review

The ground truth and both reconstructions were inspected. Both methods recover
the correct orientation, subject, pose, microphone, and scene. NP is visibly
cleaner and less grainy than SITCOM on this image, consistent with the higher raw
PSNR. This visual observation is descriptive only and does not replace the
paired panel evaluation.

## B22.2 requirements

The next authorized task is implementation and non-GPU/dry-run validation of the
full-panel execution package. It must freeze and expose separate rows for:

- existing `Fresh1`;
- existing frozen `Fresh2`;
- `SITCOM-1`;
- `SITCOM-4S`, using the frozen executable correction-norm selector;
- `NP-1`;
- `NP-8-RS`, using `global_run_by_selector` over LF/S2 and seeds 100--103.

Ground-truth-assisted oracle rows may be computed only as clearly labeled
diagnostics and must never replace the executable rows.

The implementation must:

1. consume the same 100 locked measurement tensors;
2. never regenerate measurement noise;
3. reuse existing Fresh1/Fresh2 outputs rather than rerunning them;
4. preserve raw PSNR as primary and rot180-aware PSNR as auxiliary;
5. record per-trajectory and per-policy timing, GPU-seconds, wall time, memory,
   source/config/model/tensor identities, and output hashes;
6. support deterministic resumable sharding over the four PAC GPUs;
7. refuse silent overwrite and partial-row omission;
8. package compact summaries and failure artifacts without terminal flooding;
9. perform a CPU-only plan/dry-run before any full GPU launch;
10. keep the full-panel launch blocked until explicit implementation sign-off.

## Gate

```text
B22.1 decision: PASS / SIGNED OFF
B22.2 implementation: AUTHORIZED
B22.2 full-panel execution: NOT YET AUTHORIZED
```
