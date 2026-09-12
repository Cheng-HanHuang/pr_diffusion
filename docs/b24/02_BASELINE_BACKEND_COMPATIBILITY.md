# B24 baseline backend compatibility

## Do not reuse historical SITCOM-4S as B24 SITCOM-4

The B22 `run_b22_2_sitcom_worker.py` implementation is useful historical infrastructure but its four-candidate population is not the B24 reference protocol. B22 seeds NumPy/Torch/CUDA once before model construction, saves the post-model RNG state, restores that one state for each image, and then runs four trajectories sequentially through the same RNG stream. It records one `master_seed` and a `rng_stream_run_index`.

B24 instead requires **four independent frozen SITCOM-1 trajectories**, each with its own preregistered solver seed derived by `B24_SOLVER_SEED_V1`. Therefore:

- B22 `SITCOM-4S` cannot be relabeled as B24 `SITCOM-4`;
- B22's selector is irrelevant to B24 baseline best-of-four;
- B24 best-of-four is selected only after all four terminals by raw-orientation RGB PSNR;
- a future batched SITCOM implementation must preserve one independent RNG identity per candidate and pass the B24.1 serial-versus-batched gate against four serial SITCOM-1 runs.

## DAPS-4 reference

B24 DAPS-4 is four independent frozen Fresh1 trajectories on the same locked measurement with four preregistered B24 solver seeds. The accepted B23.1 Fresh1 native one-terminal parent is the semantic reference for a single candidate, not permission to reuse B23 measurements or exposed images.

The B23 evidence runner saved raw trajectories for replay auditing. B24 scale screening must not: its DAPS backend must retain terminal candidates, metrics, timing, hashes and required provenance while setting trajectory retention off.

A future batched DAPS implementation must preserve the four independent candidate seed identities and pass the same B24.1 serial-versus-batched equivalence gate.

## B24.1 implementation gate

B24.0 intentionally supplies the deterministic control plane while leaving scientific GPU dispatch disabled. Under separate B24.1 authorization, the executor must:

1. implement serial references as four independent Fresh1/SITCOM-1 candidates;
2. implement the memory-efficient batched form without changing the frozen candidate identities;
3. run only the exposed-image equivalence/memory/throughput smoke;
4. freeze the backend commit and observed memory before B24.2;
5. stop if any candidate correspondence, metric threshold decision, source identity, RNG identity, or tolerance gate disagrees.

Only after B24.1 signoff may the 64-image B24.2 baseline screen run. Method-development logic is not needed for B24.1 or B24.2 and remains downstream of baseline class allocation.
