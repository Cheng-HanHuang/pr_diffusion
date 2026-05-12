# Repo audit notes after FFHQ tuning phase

## Key checks

1. Verify score-mode arguments and logic in NP benchmark scripts.
2. Verify CUDA-safe complex magnitude handling.
3. Verify projection-radius schedule support.
4. Remove/archive accidental backup files (`*.bak*`, `*_patched.py`).
5. Run Python compile checks.
6. Ensure README/docs make FFHQ the main benchmark and CelebA-HQ historical.
7. Decide whether local analysis scripts should be committed.

## Conceptual checks

- Distinguish global best-run vs image-level best-of-k statistics.
- Keep alignment conventions explicit (raw/rot180/resolve).
- Avoid over-claiming NP over SITCOM; present regime-dependent strengths.
