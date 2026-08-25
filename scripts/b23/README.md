# B23 scripts

All B23.0 entrypoints are CPU-only protocol, inventory, validation, packaging, or publication tools.
The consolidated PAC entrypoint is `run_b23_0_zero_gpu.sh`. It clears `CUDA_VISIBLE_DEVICES`, uses
the explicit PAC `daps` Python, writes full logs under the B23 output root, creates the required
extracted capsule and `.tar.gz`, and prints only a short summary. Every prerequisite runs from the
repository with an explicit `PYTHONPATH`; each return code is recorded, and later steps are skipped
after the first failure.

`--publish` is intentionally explicit. It is the user-authorized exception that commits both the
transparent extracted evidence and a safety-checked archive smaller than 5 MiB, then performs a
non-force push to `codex/b23-execution`. It refuses a dirty worktree or a changed remote head.
It also refuses publication unless the exact four-row zero-GPU prerequisite ledger is all PASS.

The later `run_b23_1a_b.sh` entrypoint is separately authorized by the 2026-08-24 decision and is
not part of B23.0. It accepts only an exact pushed pre-run head and runs the bounded B23.1A/B graph:
five locked inputs, four ordered parent replays, pre-wrapper tolerance freezes, coupled compute
calibration, four-image smoke, donor classification, and return packaging. After the preserved
seed-range false start, the entrypoint requires `--reuse-inputs` and validates the five existing
locked inputs without regenerating them. It also requires `--recover-fresh1-native0` for the exact
completed Fresh1 native-0 partial run rejected by the incomplete 40,401-vs-40,402 RNG formula.
Recovery is zero-GPU, so only the other 31 authorized trajectories execute. It hard-stops before
B23.2, large panels, B24, or adaptive schedules.
