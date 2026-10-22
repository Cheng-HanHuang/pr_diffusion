# Track R R0.0b NumPy metadata drift adjudication — 2026-10-22

**Status for Cheng:** The clean Track R checkout at
`4d1f70309ed91923eee12ebb20b6ad8f551e8eeb` executed
R0.0b at 07:31Z. Dependency reconciliation stopped with
`STOP_R0B_CORE_CLONE_DRIFT numpy 2.4.3 2.4.6`.
No assets, CPU tests, or GPU work followed the refusal.
All source environments remain untouched.

**Important distinction:** the stop came from Python
`importlib.metadata.version("numpy")`, which reads installed
distribution metadata, not `numpy.__version__` of the code
that is actually imported. The earlier R0.0 inventory explicitly
recorded `numpy.__version__ == 2.4.6` for
`prdiff_ffhq`. A second, stale distribution-info record
is plausible but **not yet demonstrated**; the correct response
is to require evidence, not overwrite NumPy.

`r0_0b_reconcile_pip.py` now has one narrow adjudication rule:
if and only if **NumPy alone** disagrees between metadata versions,
run both interpreters with all GPU devices hidden and compare:

1. Both imported `numpy.__version__` equal 2.4.6, matching R0.0.
2. NumPy `__init__.py` SHA256 is identical between source and clone.
3. Native `numpy._core._multiarray_umath` binary SHA256 is identical.
4. Both imported module and native extension paths are confined
   to the respective declared conda environment prefixes.

If all pass, report `R0B_WARN_NUMPY_DISTINFO_DISAGREEMENT`
and proceed with the already authorized six missing dependencies
and isolated `setuptools==80.9.0` correction.
Any real source/clone runtime or binary mismatch, missing probe
or changed torch/torchvision/scipy/triton metadata still STOPs.
There is no package replacement, retry blind-installation or
modification to historical environments.

**Planner escalation:** If runtime evidence contradicts the frozen
R0.0 observation, stop with the recorded reason; do not
reclone or downgrade NumPy until the planner decides.

Original R0.0a CPU FAST gate still VERIFIED; G0, G1 and all GPU
stages still NOT RUN.
