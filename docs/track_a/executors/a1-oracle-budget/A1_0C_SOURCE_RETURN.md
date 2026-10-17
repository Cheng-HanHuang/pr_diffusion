# A1.0c — Native DAPS start-noise seed override verified

**2026-10-17. Stage:** source-only / no GPU / no image or measurement read. **Disposition:** `DAPS_GET_START_SEED_CONSUMED_CONFIRMED`. In combination with A1.0b this closes the narrowly scoped native *initialization* source audit, **not** the full A1 independent-restart execution/preflight.

## User-returned PAC source excerpt

Pinned native DAPS HEAD observed: `e7a77d094167084faed19b599b96673b7bb11447`, matching A1.0 V2 source-identity receipt. File `/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps/sampler.py`:

- Lines around 176–182: `_b21_seed_all(seed)` converts to int and calls `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`.
- `DAPS.get_start(self,batch_size,model)`, around 430–449, obtains `device=next(model.parameters()).device`, `in_shape=model.get_in_shape()`, reads `B21_START_NOISE_SEED` from the environment, then **before sampling initial noise** calls `_b21_seed_all(int(start_seed))` whenever set. It subsequently executes:

  `x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()`

  Thus `B21_START_NOISE_SEED` is **operationally consumed to seed the actual random initial tensor**, and is not merely metadata. The native initial state is based on model input dimensions and scheduler sigma, not GT pixel values.
- Native `posterior_sample.py:233–235` separately seeds NumPy, torch, CUDA with the Hydra `args.seed` before solver setup. For A1 planned independent processes, pass both `args.seed=native_seed` and `B21_START_NOISE_SEED=native_seed`, plus `B21_SOURCE_SEED=native_seed`/fixed saved measurement path, to avoid silent per-process seed divergences. Use distinct uint32 native seeds and isolate process environment. The source references `B21_SOURCE_SEED` elsewhere and `B21_MEASUREMENT_PATH`; other stochastic calls do not become proven deterministic solely from this excerpt.

## Inference boundaries and remaining work

1. **Verified initial-noise isolation mechanism** for native DAPS, and (A1.0b) `sampler.get_start(ref)` for SITCOM uses only `torch.randn_like(ref)` times frozen sigma (no GT values). Cross-restart *statistical independence* follows from independently seeded pseudo-random streams as experimental convention; do not claim an exact mathematical proof of independence or complete downstream bitwise determinism.
2. A1 prospective SITCOM executable should supply a shape-only reference to `get_start` (same 1×3×256×256 dtype and device) rather than loading `gt` before candidate generation. Source-level removal of pixel dependency already established; implementation not yet validated.
3. Existing read-only historical baseline source `scripts/b24/run_b24_1_method_smoke.py` shows one subprocess per seed for native DAPS and SITCOM with CUDA device isolation and per-candidate timing; A1 will implement its **own** dry-run/launcher under `scripts/track_a/a1_*`, with all historical scripts, outputs and source checkouts untouched.
4. Remaining gates: CPU-only synthetic 16-trajectory unique seed/command manifest preflight, validation of native NP and DAPS/SITCOM observation preprocessing and common float PSNR, new provenance-eligible DEV image cohort and independent measurement seed namespace, live issue #41 [DATA RESERVATION] and [RUN RESERVATION], then an explicitly user-launched bounded GPU pilot.
5. **Do not** treat exclusion of 180 exposed B21/B22/B24 DEV80 IDs as proving disjointness from the excluded protected B24 confirmation cohort. Its metadata and membership must not be accessed.

**Status:** `GO_A1_CPU_RESTART_PLAN_ENGINEERING`, `NO_GPU_PERMISSION`, `NO_FRESH_COHORT`. No log ZIP was created for A1.0c; all details derive from user-returned PAC terminal source text (not a hashed stand-alone artifact).
