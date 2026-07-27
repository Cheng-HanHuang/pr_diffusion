# B22.1 smoke attempt 1: NP JSON-serialization failure

## Status

**Checkpoint result:** incomplete due to an artifact-serialization bug after a
successful NP reconstruction.

This is not an NP algorithm failure and does not indicate a measurement,
checkpoint, environment, CUDA, or finite-output incompatibility.

Full-panel execution remains blocked.

## Returned archive

```text
B22_1_smoke_20260727_142201.tar.gz
```

PAC run root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/
B22_1_smoke_20260727_142201
```

Repository head used by attempt 1:

```text
6852c84132159028f1a8104e2e9ed56db71ae7ee
```

Selected smoke image and locked measurement:

```text
image_id: 60044
measurement: ffhq60044_phase_noise005_meas5401.pt
measurement file SHA-256:
98ad24d664df0366e81a8b9f5418ff0cb7c14fb967d92bb40d72cbebfdc344de
measurement tensor-content SHA-256:
877ab671eb3fc53d4f8311dee9a424d2acb7f9e4927eb9c90cf32b8ab06e2f93
```

The selection rule was the lexicographically first locked measurement and did
not use any method outcome.

## Successful steps

### CPU preflight

Passed all frozen source, model, measurement, ground-truth, and path checks.

### SITCOM-1

SITCOM completed and wrote a valid result:

```text
status: OK
seed: 43
raw PSNR: 26.8377895355 dB
rot180 PSNR: 8.3543310165 dB
reconstruction time: 89.7891111728 s
peak allocated GPU memory: 664651776 bytes
peak reserved GPU memory: 975175680 bytes
reconstruction tensor-content SHA-256:
14ed2c4e209d2f00601f73cfd35b87c7db1365a57065c41a3c3ab162cae429d2
```

The method consumed the exact raw locked measurement.

### NP reconstruction

The frozen NP-1 trajectory completed all 999 reconstruction transitions, wrote
finite tensor and PNG artifacts, and computed valid metrics:

```text
config: LF
seed: 100
steps: 1000
soft candidates: 5
hard candidates: 1
projection start: 300
score radius: 0.6
projection radius: 0.2
raw PSNR: 29.3942241669 dB
rot180 PSNR: 8.3695468903 dB
peak allocated GPU memory: 674237952 bytes
peak reserved GPU memory: 979369984 bytes
negative measurement entries clipped in memory: 157327
clipped measurement tensor-content SHA-256:
26d521a5ae28163598d5c9828803eb369ff1f1943265834dca1b4e6df26552ba
reconstruction tensor-content SHA-256:
b6dee35138e453fc8a1c77aa1dc3331d1b70bce8fa75a10d66fc39d5fd836641
```

## Exact failure

After saving the reconstruction tensor and PNG, the NP runner attempted to
write `np1/result.json` with strict `allow_nan=False` JSON compliance. The
selector diagnostics contained:

```text
selector_post_lf_mse_margin_mean = NaN
```

The resulting exception was:

```text
ValueError: Out of range float values are not JSON compliant: nan
```

The field is expected to be undefined. After projection begins, the frozen
configuration uses `hard_candidates=1`; there is no second candidate from which
to define a first-versus-second post-projection margin.

The old writer began writing directly to the final path, so it left a truncated
and invalid `np1/result.json`. Timing fields occurred after the invalid field in
the serialized key order and were therefore not durably recorded. Independent
validation correctly did not run.

## Classification

```text
method failure: no
measurement incompatibility: no
finite-output failure: no
checkpoint/environment failure: no
artifact contract failure: yes
failure stage: post-reconstruction JSON serialization
```

No policy, seed, measurement, preprocessing, or model parameter may be changed
in response.

## Corrective implementation

The B22 branch now:

1. records non-finite scalar diagnostics as standards-compliant JSON `null`;
2. records the paths of undefined diagnostics explicitly;
3. preserves strict JSON—non-standard `NaN` tokens remain forbidden;
4. provides an NP-only recovery launcher;
5. preserves the entire first NP attempt;
6. reruns the same NP configuration and seed;
7. requires an exact reconstruction tensor-content hash match against attempt 1;
8. reuses the already successful SITCOM result;
9. runs the original independent validator;
10. keeps `full_panel_authorized=false` pending execution-lead review.

The replay is required because exact NP timing was not durably written before
the serialization exception. It is not a new experiment or a policy rerun for
selection purposes.

## Recovery command

After pulling the corrected branch, run:

```bash
bash scripts/b22/resume_b22_1_after_np_json_failure.sh \
  0 \
  /egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/B22_1_smoke_20260727_142201
```

Replace GPU `0` only if another physical GPU should be used. Do not change any
method parameter.

## Gate

```text
B22.0 inventory: SIGNED OFF
B22.1 attempt 1: INCOMPLETE — output-contract failure after NP reconstruction
B22.1 NP-only recovery: AUTHORIZED
Full 100-image panel: BLOCKED
```
