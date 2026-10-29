"""Bit-identical cache for the tail-variant conditional prefix law (planner QA patch, 2026-10-29).

fit_th1_prefix_v1.conditional_old_scores recomputes, on every draw, a sum over all
2^(#CONSISTENT runs) heavy-tail statuses of the observed ranks. With 16-run ImageNet
pilots that is up to 65,536 Plackett-Luce likelihoods per image per draw (about 6 h of
production). The status probabilities depend only on (ranked labels, weights, eps), never
on the random generator, so they are computed once per (ranked, weights, eps) and reused.
The generator is consumed exactly as in the original (one rng.choice over the same
normalized vector, then the same exponential draws), so outputs are bit-identical; the
selected status is rebuilt from its index in itertools.product order. Importing this module
installs the replacement into fit_th1_prefix_v1, where sample_panel looks it up.
"""
import itertools
import numpy as np
import fit_th1_prefix_v1 as _PV1

LABELS = _PV1.LABELS
_CACHE = {"key": None, "table": {}}


def _expand(ranked, choices, status, weights):
    status_by_index = dict(zip(choices, status))
    cl = [(5 if c == "I" else (4 if status_by_index[j] else 3) if c == "C" else LABELS.index(c))
          for j, c in enumerate(ranked)]
    rr = [float(weights[k]) for k in cl]
    if any(c == "I" and any(d != "I" for d in ranked[j + 1:]) for j, c in enumerate(ranked)):
        rr = [.001 if c == "I" else rate for c, rate in zip(ranked, rr)]
    return cl, rr


def _tail_probs(ranked, weights, eps):
    key = (tuple(float(x) for x in weights), float(eps))
    if _CACHE["key"] != key:          # keep only the current fit's table (bounded memory)
        _CACHE["key"] = key
        _CACHE["table"] = {}
    tab = _CACHE["table"]
    hit = tab.get(ranked)
    if hit is None:
        classes = [LABELS.index(c) for c in ranked]
        choices = [i for i, c in enumerate(classes) if c == 3]
        probs = []
        for status in itertools.product([False, True], repeat=len(choices)):
            cl, rr = _expand(ranked, choices, status, weights)
            prior = (eps ** sum(status)) * ((1 - eps) ** (len(status) - sum(status)))
            probs.append(prior * _PV1.weighted_permutation_prob(cl, rr))
        ps = np.asarray(probs, dtype=float)
        if ps.sum() <= 0:
            raise ValueError("zero heavy-tail conditional likelihood")
        ps /= ps.sum()
        hit = (choices, ps)
        tab[ranked] = hit
    return hit


def conditional_old_scores_cached(ranked, variant, fit, rng):
    weights, eps = _PV1.score_weights(fit, variant)
    classes = [LABELS.index(c) for c in ranked]
    n = len(classes)
    if variant == "tail":
        choices, ps = _tail_probs(ranked, weights, eps)
        selected = int(rng.choice(len(ps), p=ps))
        m = len(choices)
        status = tuple(bool((selected >> (m - 1 - i)) & 1) for i in range(m))
        actual, rr = _expand(ranked, choices, status, weights)
    else:
        actual = classes.copy()
        rr = [float(weights[k]) for k in actual]
        if any(c == "I" and any(d != "I" for d in ranked[j + 1:]) for j, c in enumerate(ranked)):
            rr = [.001 if c == "I" else rate for c, rate in zip(ranked, rr)]
    scores = np.full(n, np.inf, dtype=float)
    t = 0.
    for j in range(n):
        total = sum(rr[j:])
        if total <= 0:
            break
        if rr[j] <= 0:
            raise ValueError("observed rank impossible under reference race")
        t += float(rng.exponential(1 / total))
        scores[j] = t
    return actual, scores


_ORIGINAL = _PV1.conditional_old_scores
_PV1.conditional_old_scores = conditional_old_scores_cached
