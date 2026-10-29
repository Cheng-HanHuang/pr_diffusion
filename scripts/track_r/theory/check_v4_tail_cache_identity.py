#!/usr/bin/env python3
"""Planner QA (2026-10-29): the cached conditional_old_scores is bit-identical to the original.
All 100 ImageNet 16-run pilots x 4 variants x 4 fits (primary, tau1, 2 bootstrap refits) x 2 seeds x 2 passes:
identical (labels, scores) and identical generator state after each pass.
Usage: python3 -s check_v4_tail_cache_identity.py XP_D1_16RUN_A8_STYLE_FOR_V4.csv"""
import csv, sys, numpy as np
import fit_th1_prefix_v1 as F
ORIG = F.conditional_old_scores
from audit_a8_rank_race import fit_weights
import v4_tail_cache as C
assert F.conditional_old_scores is C.conditional_old_scores_cached and C._ORIGINAL is ORIG
rows = [{"image_id": r["image_id"], "n": 16, "ranked": r["e0_rank_labels"],
         "counts": np.array([int(r[k]) for k in ("n_GOOD", "n_ORBIT_global", "n_ORBIT_partial", "n_CONSISTENT", "n_INCONSISTENT")])}
        for r in csv.DictReader(open(sys.argv[1]))]
rngb = np.random.default_rng(1)
fits = [fit_weights(rows, tau=2.5), fit_weights(rows, tau=1.)]
for _ in range(2):
    idx = rngb.integers(0, len(rows), len(rows)); fits.append(fit_weights([rows[int(j)] for j in idx], tau=2.5))
n = 0
for fi, fit in enumerate(fits):
    for variant in ("tail", "race", "S2", "race_tau1"):
        for seed in (11, 12):
            r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
            for _ in range(2):
                for row in rows:
                    x = ORIG(row["ranked"], variant, fit, r1); y = C.conditional_old_scores_cached(row["ranked"], variant, fit, r2)
                    assert list(x[0]) == list(y[0]) and np.array_equal(x[1], y[1]), (row["image_id"], variant, fi, seed)
                    n += 1
            assert r1.bit_generator.state == r2.bit_generator.state
print("V4_TAIL_CACHE_IDENTITY_PASS", n)
