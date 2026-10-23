# R0-N: N1 plotting dependency failure; N0-preserving resume (2026-10-23)

## PAC evidence and exact scientific gate

Attempt `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0n_all_20261023T181730Z`:
- N0 import RC=0; immutable aggregate table commit
  `0e487bf28e9f6ae5ac4f29005b97a1c6a2132208`
  on local detached HEAD; push attempt failed because VSCode
  Git askpass socket at `/run/user/31111/vscode-git-b0da0d831f.sock`
  was unreachable and anonymous write access was denied.
- N1 test suite RC=0; `97 passed`.
- N1 CPU RC=8 at figure rendering **only**:
  `ModuleNotFoundError: No module named 'matplotlib'`.
- Before exception: `R0N_N1_K3_REPLAY_PASS 3240 2040 85 324 791 REPAIR 84/85 HARM 0/2040`.
- Frozen 3240 score-receipt replays reached `R0N_N1_SCORE_RECEIPTS 3000 /3240` and then the plotting call, which necessarily means all 3240 passed.
- This has **no connection to GPU occupancy**, model failures, numerical mismatch, source hashes, GT leakage, or 0.25h N2 budget. The GPU stage was not launched.
- The failed N1 run may contain pre-figure JSON/CSV drafts; preserve them unmodified for forensics, but they are **not** a final N1 PASS because report/figures/completion receipt were not generated.

## Narrow repair

- New `prdiffusion/track_r/r0n_figures_pillow.py`:
  creates meaningful, data-driven PNG/PDF visualizations for
  N1a label anatomy, N1d cost-success relation, and N1c signed-residual
  violin plots via the already-used Pillow and NumPy libraries.
  The violin's smoothing is graphical only; numerical results are unchanged.
- `scripts/track_r/r0n_1_letter_tables_cpu.py` first tries frozen
  Matplotlib plots; when *Matplotlib itself* is absent, it uses the
  Pillow renderer. Logs `R0N_N1_FIGURES_PILLOW_FALLBACK_NO_MATPLOTLIB`.
- Synthetic regression actually opens the generated PNGs and verifies
  PDF magic bytes and file sizes. No new pip or conda install needed.
- Launcher adds static AST and Pillow import/version preflight.
- N2 already received a separate **measured free-VRAM** co-tenancy
  admission gate: >=40 GiB free and runtime canary; 900s total
  GPU ceiling unchanged and no protected data accessed.

## Correct N0-preserving recovery

Current PAC worktree is detached at local N0 commit `0e487bf`.
Latest implementation is pushed to the GitHub executor branch, but
the local N0 commit was **not** pushed. Do **not** reset or delete it.

1. Confirm clean PAC worktree and preserve `0e487bf` under a
   distinct local `backup/r0n-n0-0e487bf-20261023` branch.
2. Fetch `codex/prdiff-track-r-r0n-20261023` to local refs.
   Confirm remote HEAD matches exact advertised implementation SHA.
3. Checkout `--detach` that implementation SHA.
4. Locally `git merge --no-ff` the preserved N0 commit into
   the implementation (both descend from planner merge and
   accepted R0, so N0 adds only archival aggregate files).
   This produces a **new two-parent local merge**, with both the
   latest implementation and SHA-verified original N0 as ancestors.
5. Verify merged ancestry, clean status, original N0 files and
   their `sha256sum -c SHA256SUMS.txt`.
6. Launch `bash scripts/track_r/r0n_all_launch.sh
   <actual-local-merge-head> --resume-n0`.
   This checks the N0 files are still tracked and SHA identical,
   then runs CPU tests and starts new N1 CPU output root. No N0
   import, N0 re-commit, or recreation of any candidate.

The controller will still attempt a normal non-force push, but failure
to authenticate is recorded in `push_errors`, leaving both local
commits and the planner ZIP intact. Final status will bear
`_LOCAL_PUSH_PENDING` if GitHub publication is not possible on PAC.

No increase to optional N2 cap is authorized by this engineering fix.
No R1, no new reconstruction, and no confirmation305 access.
