# Recorded zero-GPU procedure (documentation; not an executable launcher)
cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> -m unittest discover -s tests/b23 -v
cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/validate_b23_0.py --repo <execution-worktree>
cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/render_b23_1_dry_runs.py --repo <execution-worktree> --output <b23-output-root>/B23_1_dry_run_<timestamp>.json
cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/collect_b23_0_pac_evidence.py --repo <execution-worktree> --output-root <b23-output-root>
