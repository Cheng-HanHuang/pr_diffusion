# A1.0b native startup/RNG source inspection — user-returned evidence

**Date:** 2026-10-17. **Executor:** continuing Track A executor. **Result:** `SITCOM_INIT_GT_VALUES_NOT_USED_CONFIRMED`; `DAPS_SEED_CALLS_SEEN / START_NOISE_OVERRIDE_BODY_REVIEW_PENDING`. No GPU, reconstruction, measurement generation, new data, or B24 confirmation access.

## PAC output inspected

User ran source-only `grep` against original `/egr/research-pac/huang248/external/SITCOM_ODE/sampler.py`, project `scripts/b22/run_b22_1_sitcom_smoke.py`, native DAPS `posterior_sample.py` and `sampler.py` and pasted output in the conversation. This source-output has **no standalone PAC SHA receipt**; findings below are grounded in the user-returned literal lines, cross checked with GitHub's corresponding B22/B24 invocation and A1.0 V2 verified native HEAD/diff source identities. Not a new model execution or compiled proof.

### SITCOM: the specific initialization leak suspicion is resolved

Two native sampler methods exist:
- `sampler.py:167-178`, `get_start(self,ref)` returns `torch.randn_like(ref)*self.scheduler.sigma_max`.
- `sampler.py:403-414`, `get_start(self,ref)` returns `torch.randn_like(ref)*self.annealing_scheduler.sigma_max`.

These expressions refer only to `ref` **shape/dtype/device/layout**, not tensor pixel contents. The existing `scripts/b22/run_b22_1_sitcom_smoke.py:139` passes a GT tensor as `ref`; therefore *this initialization expression* does **not** leak GT numerical values into candidate generation. Entry point lines 68–72 seed `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` and request deterministic cuDNN. `sampler.sample(..., measurement, evaluator=None,...)` in project source lines 141–149 is compatible with clean-free **generation** (the later `metric_pair(reconstruction,gt)` is postrun evaluation); no evidence of GT-guided *initial state* here. To enforce a stronger architectural firewall, A1's own prospective SITCOM entrypoint should use an **empty shape-only reference tensor** with the correct frozen 1×3×256×256 float/device rather than loading GT for `get_start`. Actual equivalence and native sampling call must be tested before GPU/prospective claims.

### DAPS: the user output establishes seed hooks, but not their complete semantics

- `posterior_sample.py:233-235` invokes `np.random.seed(args.seed)`, `torch.manual_seed(args.seed)`, `torch.cuda.manual_seed_all(args.seed)`.
- `sampler.py:180-182` separately invokes NumPy/PyTorch/CUDA RNG seeds (specific calling context still must be read).
- `sampler.py:225-226` records `B21_START_NOISE_SEED`, `B21_SOURCE_SEED`, and `B21_MEASUREMENT_PATH` into metadata; `sampler.py:239` describes a synthetic step-0 clean state; `sampler.py:442` reads `B21_START_NOISE_SEED`.
- Project native `scripts/b21/run_b21_11_fresh2_final_benchmark_multigpu.sh:206-221` explicitly passes these environment variables and `seed="$seed"`, one candidate/process. Project `scripts/b24/run_b24_1_method_smoke.py:191-243` (read-only historical code) constructs separate DAPS/SITCOM child processes, uses `native_seed=canonical_seed % 2**32` and isolated per-replicate paths, time/memory records and `CUDA_VISIBLE_DEVICES`. This is a **design precedent**, not certification of any new A1 run.
- **Still unverified:** source body surrounding native DAPS `B21_START_NOISE_SEED` at 442, precisely how sampler seeds are scoped/reset and whether noise is actually sampled using the override rather than merely reported.

## Findings, permissions and next action

1. Mark the *specific* SITCOM `get_start(gt)` GT-value concern resolved **for random initial-state construction**. Do not generalize to every unrelated sampler/evaluation codepath without further inspection.
2. Continue with bounded native DAPS `sampler.py` **exact source-body inspection**, with no GPU. The next source-only command and details are in [A1_0C_DAPS_SEED_SOURCE_HANDOFF.md](A1_0C_DAPS_SEED_SOURCE_HANDOFF.md).
3. Independent DAPS4/SITCOM4 process launchers, shared noisy-measurement identity and representation/evaluation parity still require an A1-owned implementation and synthetic/CPU source preflight. A0/B21/B22/B24 artifacts and Git sources remain immutable.
4. The A1 GPU engineering pilot remains **BLOCKED** by both native-runner implementation and disjoint/provenance-certified actual-image cohort governance. Simple `FFHQ \ {180 exposed}` exclusion is **not** a legitimate certificate of disjointness from the excluded B24 confirmation set; never inspect that locked registry to obtain membership.
