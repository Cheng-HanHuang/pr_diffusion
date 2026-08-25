# Compute ledger and Fresh1-equivalent-run specification

## Reference

`1.0 FRE` is one frozen Fresh1/DAPS trajectory: FFHQ 256, phase retrieval oversample 2.0,
`sigma_y=0.05`, ann400, diff5, LF/HIO off, one start, one retained terminal candidate, and the exact
source/model/environment/hardware recorded by the PAC freeze.

FRE is not “one invocation” and heterogeneous raw calls are not treated as equal.

## Per-image raw ledger

`schemas/b23/compute_ledger.schema.json` requires:

- denoiser forward, backward, JVP, and VJP calls;
- measurement forward, adjoint, JVP, and VJP calls;
- FFT, projection, and correction calls;
- optimizer iterations in aggregate and by optimizer type;
- state conversions and re-noising calls;
- random proposals and named RNG draw calls/values;
- maximum live, retained, and terminal branches/candidates;
- GPU-active and wall seconds with timer method;
- peak allocated and reserved memory;
- preprocessing, serialization, conversion, and other overhead.

Zero-count fields remain explicit. NP discarded proposals and shared-prefix work are counted. Offline
oracle/best-of-k branches are diagnostic and cannot be charged as deployment compute.

## Atomic calibration

Before any hybrid execution, B23.1 must microbenchmark every operation class that has a nonzero
count. The procedure is:

1. pin the exact B23.0 hardware, driver, CUDA, PyTorch, dtype, shape, batch, model, and operator;
2. warm up without counting warm-up iterations;
3. isolate each atomic operation with CUDA events and synchronization at declared boundaries;
4. interleave operation and Fresh1 reference measurements to control drift;
5. record all samples, median GPU-active cost `w_j`, dispersion, iteration count, and timer method;
6. rerun enough paired blocks to establish stable medians;
7. freeze a machine-readable weight registry and SHA-256 in the pre-hybrid commit.

Operations that cannot be isolated are recorded in `coupled_operation_blocks` with a unique typed
operation name, count, finite strictly positive measured weight, and SHA-256 of its definition. They
are never assigned a guessed decomposition. Every other parent operation must decompose completely
into the fixed raw counters. A nonzero atomic or coupled operation without a finite strictly positive
measured weight makes the ledger invalid.

B23.0 intentionally supplies no numeric `w_j` values.

## Arithmetic

For policy `s`, image `i`, count `c_j`, and measured atomic median `w_j`:

```text
calibrated_work(s,i) = sum_j c_j(s,i) * w_j
work_FRE(s,i) = calibrated_work(s,i) / calibrated_work(Fresh1 reference)
time_FRE(s,i) = GPU_active_seconds(s,i) / paired_median_GPU_active_seconds(Fresh1)
claim_FRE(s,i) = max(work_FRE(s,i), time_FRE(s,i))
```

`prdiffusion.b23_protocol.validate_compute_ledger` recomputes all three FRE values from the frozen
reference raw counts, coupled blocks, weights, paired GPU-active time, hardware inventory identity,
and evidence hashes. It rejects supplied values that differ from the recomputation. It also
reconciles optimizer totals by type, RNG totals by unique named stream, branch/candidate identities,
timing, memory, and overhead. `claim_FRE` is never a cheaper choice between work and time.

Schema `b23.compute-ledger.v2` also fails closed on calibrated timing. If a calibrated ledger has
any nonzero atomic counter or coupled-operation count, its GPU-active seconds and wall seconds must
both be finite and strictly positive, and `timer_method` must identify a real measurement method.
`NOT_RUN`, `UNKNOWN`, `UNAVAILABLE`, `TBD`, `N/A`, and equivalent placeholder spellings are invalid
for executed calibrated work. In particular, a ledger with a nonzero denoiser count,
`gpu_active_seconds=0`, and `timer_method=NOT_RUN` is rejected even if its supplied FRE fields are
otherwise arithmetically self-consistent.

## Gates

- B1: hard per-image work-FRE `<=1.10`, paired median time-FRE `<=1.10`, q90 time-FRE `<=1.20`,
  exactly one terminal candidate.
- B2: hard per-image work-FRE `<=2.10`, paired median time-FRE `<=2.10`, q90 time-FRE `<=2.20`,
  at most two terminal candidates.
- Module-effect comparisons: calibrated work within 5% and paired median GPU-active time within 10%.

If cost matching fails, report a cost-response frontier; do not make a causal module claim. Higher
cost points remain separately labeled and cannot support H1/H2.

## B23.0 validation

The uncalibrated example has full zero-valued raw fields, null FRE values, no atomic weights, and
status `UNCALIBRATED`. Unit tests validate arithmetic on synthetic measured weights without freezing
those test numbers as scientific weights. The uncalibrated, all-zero example may retain
`timer_method=NOT_RUN` because it records no executed work and makes no calibrated claim.
