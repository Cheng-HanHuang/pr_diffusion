# Typed state, module, and adapter API

## Design rule

No field named `state.x` has universal semantics across DAPS, NP, and SITCOM. A state is a typed
native envelope whose payload names and representation contracts are parent-specific. An operation
may cross a parent boundary only through a named, audited adapter.

The CPU-testable reference types are in `prdiffusion/b23_protocol.py`:

- `NativeState[parent]`
- `SemanticCoordinate`
- `RNGStreamState`
- `ModuleSpec`
- `AdapterSpec`

These are protocol types, not GPU wrappers and not evidence that a donor is valid.

## `NativeState[parent]`

Required fields are:

- parent ID and exact source revision/patch identity;
- named tensor payloads and one representation contract per payload;
- native coordinate plus normalized semantic-noise and cumulative-budget coordinates;
- exact model, scheduler, operator, measurement, and preprocessing identities;
- native optimizer state where applicable;
- independent named RNG streams with seed, device, counter, and optional serialized state;
- trace and compute-ledger IDs.

Validation rejects blank identities, unmatched payload/contract keys, duplicate RNG stream names,
non-finite coordinates, or missing model/scheduler/measurement identities.

## `Module[input -> output]`

A module declares its exact input and output parent/state contracts, valid normalized-noise interval,
mathematical semantics, named RNG streams, raw counter names, and whether the boundary may be
serialized. A module cannot relabel an NP proposal/ranking procedure as a single “soft step,” nor
split SITCOM in a way that silently drops LGVD or forward-noising semantics.

Module replay requires:

1. valid native input from an unchanged parent trace;
2. payload, shape, dtype, device, operator, and coordinate checks;
3. exact operation/RNG count reconciliation;
4. output within the frozen native repeatability envelope;
5. resume equivalence at a genuinely valid boundary.

## `Adapter[A -> B]`

Every adapter records:

- source and target parents and native coordinates;
- mathematical conversion and normalization;
- information discarded;
- information newly sampled;
- validity checks and meaningful round-trip check;
- state-conversion/renoising cost;
- named RNG streams consumed;
- whether the conversion is lossy.

An adapter may not reinterpret a clean estimate as a valid noisy state, change the operator or
measurement, reset optimizer/RNG state silently, generate hidden candidates, or call a lossy map
reversible. The reference validator rejects a lossy adapter claiming an exact round trip.

## Shared coordinates

Two coordinates are recorded without pretending raw step indices are comparable:

1. normalized semantic noise/log-SNR, monotone on `[0,1]` and derived from the native coordinate;
2. cumulative fraction of preregistered work-FRE on `[0,1]`.

Noise controls semantic compatibility; budget controls cost matching. Cross-parent switches require
a preregistered semantic boundary and a qualified adapter.

## Serialization and resume

`serialize_native_state` writes canonical JSON metadata and a PyTorch payload file at an explicit
native boundary. Each tensor is hashed from CPU-contiguous bytes with shape and dtype. Loading checks
protocol version and all tensor fingerprints before rebuilding the typed state.

The B23.0 synthetic CPU test demonstrates lossless serialization mechanics only. B23.1 must show
that a native algorithm actually exposes a valid resume boundary. Where it does not, the check is
`NOT_APPLICABLE`, not simulated by inventing state.

## Failure classification

Any state/adapter validity failure, non-reconciled RNG or compute ledger, information-loss ambiguity,
or replay failure makes that cross-parent operation `BASELINE-ONLY` or `REJECTED-PROTOTYPE`.
Failure of one donor does not invalidate the other native baselines.
