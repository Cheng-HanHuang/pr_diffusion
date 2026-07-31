# Replay, determinism, and RNG policy

## Two replay tracks

Bitwise replay is eligible only when repeated unchanged native runs on the pinned PAC stack have
equal output/checkpoint hashes. Tolerance-qualified replay is used when the native GPU path itself
is not bitwise repeatable. No cross-release or cross-hardware bitwise result is required.

B23.0 defines the procedure; B23.1 supplies measurements.

For each parent in B23.1:

1. run at least three unchanged native repeats on one signed, new smoke row;
2. record output hashes, intermediate traces, scalar metrics, raw operation counts, and RNG ledgers;
3. determine bitwise eligibility from the native repeats;
4. if needed, freeze native max/mean absolute error, relative L2 error, trace deltas, raw-PSNR delta,
   and measurement-loss delta plus a declared dtype/scale numerical floor;
5. only then run the wrapper on the same row;
6. require wrapper error inside the frozen envelope and exact operation/RNG reconciliation.

The result uses `schemas/b23/replay_report.schema.json`. A `PASS` is impossible unless operation and
RNG counts both reconcile.

## Deterministic-flag audit

Record current and tested settings for:

- `torch.use_deterministic_algorithms`;
- `torch.backends.cudnn.deterministic`;
- `torch.backends.cudnn.benchmark`;
- `CUBLAS_WORKSPACE_CONFIG`.

Do not force a flag when it changes the frozen parent or makes its native path invalid. Report that
effect and retain the native algorithm.

## Named RNG streams

The versioned derivation is SHA-256 over canonical JSON fields:

```text
base_seed, stream_name, image_id, measurement_id,
parent_id, branch_id, draw_index, version=b23-sha256-v1
```

The first eight digest bytes are interpreted big-endian and masked to a nonnegative 63-bit integer.
The implementation and frozen test vector are in `prdiffusion/b23_protocol.py` and
`tests/b23/test_protocol.py`.

Required streams include measurement noise, native start noise, diffusion transition, NP proposals,
adapter re-noising, optimizer stochasticity, and diagnostic-only randomness. Each has its own derived
seed so adding an unrelated module cannot shift later draws in another stream.

Each draw record includes stream/base/derived seed, generator device, draw shape, dtype, device,
call count, values drawn, branch/proposal ID, serialized state hash, and whether an adapter introduced
new randomness.

Different parents need not consume equal randomness. The unchanged parent and its wrapper must.

## Serialization/resume

At a valid boundary, preserve payload hashes, optimizer state, all generator states and counters,
native/semantic/budget coordinates, source/model/operator identity, trace ID, and ledger ID. Resume
must reproduce the uninterrupted continuation inside the native envelope with exact counts. If the
parent exposes no sound boundary, report `NOT_APPLICABLE`; do not manufacture one.

## Stops

Hidden retry/candidate randomness, counter drift, serialization mismatch, wrapper error outside the
native envelope, NaN, or nontermination is a replay failure. Preserve the failure and stop; do not
change a seed or parent to obtain a pass.
