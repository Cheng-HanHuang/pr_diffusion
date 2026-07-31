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

None of these files authorizes B23.1 or contains an experiment launcher.
