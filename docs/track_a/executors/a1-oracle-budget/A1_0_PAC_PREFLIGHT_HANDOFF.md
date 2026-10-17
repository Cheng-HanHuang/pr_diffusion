# A1.0 PAC handoff — first user-run stage is strictly read-only

**Status:** READY FOR USER TO RUN `A1.0` SOURCE + METADATA PREFLIGHT; **NOT** GPU or real-image development. Continuing Track A executor, 2026-10-17.

## Purpose

Use the dedicated A1 branch `codex/prdiff-track-a-a1-oracle-budget-20261016` and **new** reserved PAC checkout only. Run tests before inventory. Compare current Git HEAD/diff signatures of the native DAPS, SITCOM and DiffFPR copies to the known B22/B24 source signatures, SHA-256 two historical FFHQ checkpoint paths, confirm the archived A0 return and the three explicitly **exposed** B21/B22/B24 DEV80 metadata records, check count/overlap of exposed panel IDs, and capture a bounded one-shot `nvidia-smi` observation.

**No sample reconstruction**, no diffusion model import, no FFT, no measurement/image/GT reading, no new image-ID allocation, no B24 confirmation305 registry or membership check, no new random observation and no GPU kernels. Python stdlib only. Preflight does not even list public FFHQ image file names; it merely checks the root directory exists.

The script is `scripts/track_a/a1_stage0_preflight.py`; test runner/archiver is `scripts/track_a/a1_run_stage0_preflight.sh`, using `scripts/track_a/test_a1_protocol.py` and `test_a1_stage0_preflight.py`. The user-run command must **pin** the exact pushed A1 branch HEAD via `A1_EXPECTED_HEAD`, verify clean checkout and branch before execution, and never modify historical checkout sources or outputs.

## Output

Dedicated new A1 root:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/`.

- `A1_0_SYNTHETIC_TESTS.log` — full synthetic tests;
- `A1_0_PREFLIGHT.log` — short source/inventory check; absent if tests fail;
- `a1_0_preflight_20261017/A1_0_PREFLIGHT.json` — bounded source identities/model hashes/exposed-panel counts/hardware glimpse;
- `A1_0_PREFLIGHT_RETURN.zip` — safe compressed above logs/JSON, assembled even after test or source STOP (when tests created log).

The runner returns **nonzero on STOP** but never kills the interactive shell. It refuses to overwrite any previously created run file; ZIP SHA-256 is printed. Please return the ZIP and terminal summary. If only test failures occur, the ZIP still contains the test log. Stage progress is not silently promoted to source/method certification unless required checks PASS.

## Interpretation and next stage

Even if status is `PASS_SOURCE_AND_EXPOSED_METADATA_PREFLIGHT`, **no eligible new prospective cohort has been established**, because the script cannot certify membership against the protected B24 confirmation pool and is forbidden to read its registry. Need a separately *authorized disjoint public FFHQ development pool*, exclusion/reservation governance through issue #41, verified independent native SITCOM4 and DAPS4 restart wrappers, and actual run+GPU reservation before any measurement generation or GPU use.

A1.1 proposed engineering smoke: eight independently reserved DEV images, 16 trajectories/image (4 DAPS, 4 independent SITCOM, 4 NP-LF, 4 NP-S2), total 128 unique trajectories. Fixed NP mixed8 is an offline view of existing eight NP candidates, not a separate inference arm. Estimated candidate reconstruction ~2.74 aggregate GPU-hours from historical method-specific times only; maximum proposed allocated pilot 6 aggregate GPU-hours and 32 GiB process VRAM per GPU, **not** a justified hardware-specific launch yet. Full experimental protocol, hypothesis and publication go/no-go in [A1_STUDY_PREREGISTRATION.md](A1_STUDY_PREREGISTRATION.md).

Source caveats: native DAPS implementation contains frozen tracked changes; current SITCOM full diff differs from B24 due to bytecode records although its B22 Python source patch was reconciled. This preflight checks **B22 prospective intended source** and does not rewrite B24 archival uncertainty.

Upon receipt: executor independently checks the returned ZIP/checksums/tests and issues `A1_0_PREFLIGHT_RETURN.md` or `A1_0_STOP_REPORT.md` on this same branch; posts durable result to [#41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41). No PAC GPU command before this gate.
