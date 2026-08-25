# Recorded zero-GPU procedure (documentation; not an executable launcher)
CUDA_VISIBLE_DEVICES='' <daps-python> -m unittest discover -s tests/b23 -v
CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/validate_b23_0.py --repo <execution-worktree>
CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/collect_b23_0_pac_evidence.py --repo <execution-worktree> --output-root <b23-output-root>
