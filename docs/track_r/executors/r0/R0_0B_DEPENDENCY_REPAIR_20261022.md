# R0.0b pip conflict repair — 2026-10-22

**Status for Cheng:** The 2026-10-22T03:43:48Z resume reached a
new isolated environment but stopped before asset installation/tests.
The source interpreter `prdiff_ffhq` passed `pip check` and has **not**
been changed. The clean Git checkout matched `8b21023cee9a6962b9dd8d2ae8f72d7341fb8f52`.
No GPU runs were made and the B24 protected cohort remains untouched.

The full inner installer log explicitly showed six missing dependencies:

- `scikit-learn 1.9.1`: `joblib`, `narwhals`, `threadpoolctl`
- `ftfy 6.3.1`: `wcwidth`
- `pytest 9.1.1`: `iniconfig`, `pluggy`

It also showed a single installed-version incompatibility:
`torch 2.11.0+cu128` requires `setuptools<82`, but the
clone has `setuptools 82.0.1`.

The revised `r0_0b_reconcile_pip.py` now compares cloned core package
versions to the pristine source, installs missing packages with
`pip install --no-deps`, and permits **only** this verified PyTorch
setuptools mismatch to be addressed by installing `setuptools==80.9.0`
inside the Track R cloned environment. This is the original pin
already specified in the planner R0.0b handoff; it supports
`pkg_resources`. All other version conflicts stop without mutation.

R0.0a prior gate is VERIFIED:
`fast_dips_a15_blackwell` reports PyTorch 2.7.1+cu128 with compiled
flags including `sm_120`, and all four native FAST imports passed
from pinned `FAST_DIPS_A15`. B22 Fresh CSV recorded both DAPS arm
PNG paths; a DEV80 task's `baseline/` child can be inspected in the
forthcoming bounded source-layout step.

**Next:** retry R0.0b on new pinned Git HEAD; an existing
`prdiff_track_r` prefix is kept and never deleted. If dependencies
verify, install missing SHA-verified UnivFD/CLIP assets, then execute
the initial R0.1 CPU tests. No G0/G1 or GPU authorization until
scientific validity and replay gates pass.
