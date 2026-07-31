# B23 scripts

All B23.0 entrypoints are CPU-only protocol, inventory, validation, packaging, or publication tools.
The consolidated PAC entrypoint is `run_b23_0_zero_gpu.sh`. It clears `CUDA_VISIBLE_DEVICES`, uses
the explicit PAC `daps` Python, writes full logs under the B23 output root, creates the required
extracted capsule and `.tar.gz`, and prints only a short summary.

`--publish` is intentionally explicit. It is the user-authorized exception that commits both the
transparent extracted evidence and a safety-checked archive smaller than 5 MiB, then performs a
non-force push to `codex/b23-execution`. It refuses a dirty worktree or a changed remote head.

None of these files authorizes B23.1 or contains an experiment launcher.
