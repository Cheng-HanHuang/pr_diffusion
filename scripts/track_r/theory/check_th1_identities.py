#!/usr/bin/env python3
"""Seeded CPU checks for Track R TH1 statements, with no research images.

Usage:
  python scripts/track_r/theory/check_th1_identities.py
Only Python stdlib + numpy are required. All samples are SYNTHETIC.
"""
from __future__ import annotations

import itertools
import math
import numpy as np

SEED = 20261026


def comb(a: int, b: int) -> int:
    return math.comb(a, b) if (0 <= b <= a) else 0


def forward(x: np.ndarray) -> np.ndarray:
    """Per-channel padded *magnitude* measurement (synthetic only)."""
    assert x.shape[0] == 3 and x.shape[1:] == (256, 256)
    z = (x + 1.) / 2.
    p = np.pad(z, ((0, 0), (64, 64), (64, 64)))
    return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(p, axes=(-2,-1)),
                                                   axes=(-2,-1), norm="ortho"),
                                  axes=(-2,-1)))


def symmetry_check() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1, 1, size=(3, 256, 256))
    base = forward(x)
    for flips in itertools.product((0, 1), repeat=3):
        y = x.copy()
        for c, flip in enumerate(flips):
            if flip:
                y[c] = np.flip(y[c], axis=(-2, -1))
        assert np.max(np.abs(forward(y) - base)) < 1e-11, flips


def model(K: int, p: float, og: float, op: float, c: float,
          h: float=0, tg: float=1, tp: float=0) -> tuple[float, ...]:
    q = p + og + op
    s = q + c
    b = (1 - h) * p + tg * og + tp * op
    return (1-(1-p)**K, 1-(1-q)**K,
            0. if not s else (1-(1-s)**K)*p/s,
            0. if not s else (1-(1-s)**K)*b/s)


def gap_check() -> None:
    rng = np.random.default_rng(SEED)
    for i in range(1000):
        p, og, op, c, _u = rng.dirichlet(np.full(5, .3))
        h, tg, tp = rng.random(3)
        old_raw, old_orbit = -1., -1.
        for K in range(1, 71):
            o_raw, o_orb, e0, e1 = model(K, p, og, op, c, h, tg, tp)
            g_raw, g_orb = o_raw-e0, o_orb-e1
            assert g_raw + 1e-12 >= old_raw, (i,K,"raw")
            assert g_orb + 1e-12 >= old_orbit, (i,K,"orbit")
            old_raw, old_orbit = g_raw, g_orb
    # A counterexample to the *uncorrected* limiting gap if q=0:
    assert model(1024, 0., 0., 0., .7)[1] - model(1024, 0., 0., 0., .7)[3] == 0.


def exact_u_argmin(scores: np.ndarray, values: np.ndarray, K: int) -> float:
    n = len(scores)
    assert n >= K >= 1
    ordered = sorted(range(n), key=lambda i: (scores[i], i))
    return sum(comb(n-r-1, K-1)*values[j]
               for r,j in enumerate(ordered))/comb(n,K)


def exact_u_argmax(scores: np.ndarray, values: np.ndarray, K: int) -> float:
    n = len(scores)
    assert n >= K >= 1
    ordered = sorted(range(n), key=lambda i: (scores[i], -i))
    # The above key makes *lower-index* runs win ties in argmax:
    # ascending tie order = descending original index.
    return sum(comb(j, K-1)*values[i]
               for j,i in enumerate(ordered))/comb(n,K)


def exact_u_pass(good: np.ndarray, K: int) -> float:
    n = len(good)
    return 1 - comb(n-int(np.count_nonzero(good)), K)/comb(n,K)


def brute_u(scores: np.ndarray, vals: np.ndarray, K: int, which: str) -> float:
    n = len(scores)
    out = []
    for subset in itertools.combinations(range(n), K):
        if which == "min":
            idx = min(subset, key=lambda i: (scores[i], i))
        elif which == "max":
            idx = max(subset, key=lambda i: (scores[i], -i))
        else:
            raise ValueError(which)
        out.append(vals[idx])
    return float(np.mean(out))


def u_checks() -> None:
    rng = np.random.default_rng(SEED)
    for n in range(1, 9):
        for trial in range(25):
            scores = rng.integers(0, 4, n).astype(float)  # force ties
            vals = rng.normal(size=n)
            binary = (vals > 0)
            for K in range(1, n+1):
                a = exact_u_argmin(scores, vals, K)
                b = exact_u_argmax(scores, vals, K)
                assert abs(a-brute_u(scores, vals, K, "min")) < 1e-12
                assert abs(b-brute_u(scores, vals, K, "max")) < 1e-12
                v = exact_u_pass(binary, K)
                w = np.mean([any(binary[list(I)])
                            for I in itertools.combinations(range(n), K)])
                assert abs(v-w) < 1e-12


def auroc_counterexample() -> None:
    # AUROC GOOD versus CONSISTENT >0.5 does NOT guarantee
    # selecting argmin among many improves on random accept-first.
    # GOOD: 10 (70%), 100 (30%); CONSISTENT: 0 (25%), 20 (75%).
    # P(GOOD residual < CONSISTENT residual) = .7*.75 = .525.
    auc = .7 * .75
    assert auc > .5
    rng = np.random.default_rng(SEED)
    K, trials = 50, 30000
    labels = rng.integers(0, 2, size=(trials, K))  # equal class prior
    r = rng.random(size=(trials, K))
    scores = np.where(labels == 1,
                      np.where(r < .7, 10., 100.),
                      np.where(r < .25, 0., 20.))
    wins = labels[np.arange(trials), np.argmin(scores, axis=1)] == 1
    assert wins.mean() < .1, wins.mean()
    # Accept-first chance is 1/2.


def main() -> None:
    symmetry_check()
    gap_check()
    u_checks()
    auroc_counterexample()
    print("TH1_CPU_SYNTHETIC_CHECKS_PASS")
    print("seed:", SEED)
    print("checks: exact Fourier G invariance, K-gap monotonicity,"
          " exact subset order statistics with ties, AUROC counterexample")


if __name__ == "__main__":
    main()
