#!/usr/bin/env python3
"""Frozen A0 selectors. Pure metadata functions; NO file/data/model/operator imports.

A0.1 execution is forbidden until A0.0 source/measurement/representation gates PASS.
PSNR is an offline GT metric and is NEVER an executable runtime selector.
"""
from __future__ import annotations
from collections import Counter
from math import comb, isfinite
from random import Random
from statistics import mean, median
from typing import Any

GOOD = 25.0
BAD = 20.0

class AuditStop(ValueError):
    """An unavailable or scientifically incompatible historical cell."""

def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditStop(f"{label}: boolean is not a number")
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditStop(f"{label}: missing/non-numeric") from exc
    if not isfinite(n):
        raise AuditStop(f"{label}: non-finite")
    return n

def required_candidates(cohort: str, method: str, candidates: list[dict]) -> list[dict]:
    """Check identity, order, representation and full frozen K; no dropping cases."""
    layout = {
        ("B22", "DAPS"): [("rep", i) for i in range(2)],
        ("B22", "SITCOM"): [("run", i) for i in range(4)],
        ("B22", "NP"): [("lf", i) for i in (100, 101, 102, 103)]
                            + [("s2_preproj_lam001", i) for i in (100, 101, 102, 103)],
        ("B24", "DAPS"): [("rep", i) for i in range(4)],
        ("B24", "SITCOM"): [("rep", i) for i in range(4)],
        ("B24", "NP"): [("lf", i) for i in range(4)],
    }
    key = (cohort, method)
    if key not in layout:
        raise AuditStop(f"unsupported cohort/method: {key}")
    expected = layout[key]
    if len(candidates) != len(expected):
        raise AuditStop(f"{key}: expected {len(expected)} candidate records; got {len(candidates)}")
    ids = [(c.get("group"), c.get("index")) for c in candidates]
    if ids != expected or len(set(ids)) != len(ids):
        raise AuditStop(f"{key}: ordered seed/config/rep identity mismatch: {ids!r}")
    images = {c.get("image_id") for c in candidates}
    measurements = {c.get("measurement_sha256") for c in candidates}
    models = {c.get("model_sha256") for c in candidates}
    origins = {c.get("source_id") for c in candidates}
    reps = {c.get("representation") for c in candidates}
    if any(len(s) != 1 or not next(iter(s)) for s in
           (images, measurements, models, origins, reps)):
        raise AuditStop(f"{key}: image/measurement/model/source/metric mismatch")
    if len(str(next(iter(measurements)))) != 64 or len(str(next(iter(models)))) != 64:
        raise AuditStop(f"{key}: invalid SHA-256")
    if not str(next(iter(origins))).strip():
        raise AuditStop(f"{key}: source identity missing")
    for c in candidates:
        finite(c.get("psnr"), "psnr")
        if "wall_s" in c and c["wall_s"] is not None:
            if finite(c["wall_s"], "wall_s") < 0:
                raise AuditStop("negative wall time")
    return candidates

def best(candidates: list[dict], field: str, maximize: bool) -> dict:
    if not candidates:
        raise AuditStop("empty candidate prefix")
    values = [finite(c.get(field), field) for c in candidates]
    index = (max if maximize else min)(range(len(values)),
                                        key=lambda j: ((values[j] if maximize else -values[j]), -j))
    return candidates[index]

def native(cohort: str, method: str, candidates: list[dict], k: int) -> dict:
    """Return historically defined clean-free winner or raise for missing cells."""
    if not 1 <= k <= len(candidates):
        raise AuditStop(f"invalid K {k}")
    if k == 1:
        return candidates[0]
    if method == "DAPS":
        if k != 2:
            raise AuditStop("no historical native DAPS K>2 selector")
        a, b = candidates[:2]
        loss0, loss1 = finite(a.get("daps_signed_loss"), "DAPS loss0"), finite(b.get("daps_signed_loss"), "DAPS loss1")
        return b if loss1 < loss0 - 0.7 else a
    if method == "SITCOM":
        if cohort != "B22":
            raise AuditStop("B24 SITCOM step160 correction norm not archived/verified")
        if k != 4:
            raise AuditStop("SITCOM K2/K3 prefix not a deployed historical native policy")
        return best(candidates[:k], "step160_correction_norm", False)
    if method == "NP":
        if cohort == "B22" and k > 4 and k != 8:
            raise AuditStop("NP heterogeneous K5-K7 not in frozen comparisons")
        return best(candidates[:k], "postproj_lf_mse", False)
    raise AuditStop(f"unknown method: {method}")

def oracle(candidates: list[dict], k: int) -> dict:
    """Offline PSNR ceiling, stable earliest-candidate ties. Never deployable."""
    if not 1 <= k <= len(candidates):
        raise AuditStop("oracle invalid K")
    return best(candidates[:k], "psnr", True)

def ladders(cohort: str, method: str, candidates: list[dict]) -> list[dict]:
    c = required_candidates(cohort, method, candidates)
    ks = {"DAPS": [1, 2] if cohort == "B22" else [1, 2, 4],
          "SITCOM": [1, 2, 3, 4],
          "NP": [1, 2, 3, 4, 8] if cohort == "B22" else [1, 2, 3, 4]}[method]
    out = []
    for k in ks:
        winner = oracle(c, k)
        out.append(_policy(c, method, "GT_ORACLE", k, winner, winner, True))
        try:
            chosen = native(cohort, method, c, k)
            out.append(_policy(c, method, "NATIVE", k, chosen, winner, False))
        except AuditStop as exc:
            out.append({"cohort": cohort, "image_id": c[0]["image_id"], "method": method,
                        "policy": "NATIVE", "K": k, "status": "UNAVAILABLE", "why": str(exc)})
    if cohort == "B22" and method == "NP":
        for offset, group in ((0, "LF"), (4, "S2")):
            for k in (1, 2, 3, 4):
                sub = c[offset:offset+k]
                w = best(sub, "postproj_lf_mse", False)
                o = best(sub, "psnr", True)
                out.append(_policy(c, method, "NATIVE_"+group, k, w, o, False))
                out.append(_policy(c, method, "GT_ORACLE_"+group, k, o, o, True))
    return out

def _policy(c: list[dict], method: str, policy: str, k: int,
            chosen: dict, optimum: dict, gt: bool) -> dict:
    p, q = finite(chosen["psnr"], "selected PSNR"), finite(optimum["psnr"], "oracle PSNR")
    if q + 1e-9 < p:
        raise AuditStop("oracle unexpectedly below selected")
    return {
        "cohort": chosen["cohort"], "image_id": chosen["image_id"],
        "measurement_sha256": chosen["measurement_sha256"],
        "source_id": chosen["source_id"], "representation": chosen["representation"],
        "method": method, "policy": policy, "K": k,
        "status": "PASS", "gt_leak_tag": "OFFLINE_ORACLE" if gt else "CLEAN_FREE",
        "selected_id": chosen["candidate_id"], "oracle_id": optimum["candidate_id"],
        "selected_psnr": p, "oracle_psnr": q, "regret_db": max(0.0, q-p),
        "good25": int(p >= GOOD), "bad20": int(p < BAD),
    }

def quantile(values: list[float], q: float) -> float:
    if not values:
        raise AuditStop("empty quantile")
    s = sorted(values)
    x = (len(s)-1)*q
    lo = int(x)
    return s[lo] * (1-(x-lo)) + s[min(lo+1,len(s)-1)]*(x-lo)

def summarize(rows: list[dict]) -> dict:
    if not rows or any(r.get("status") != "PASS" for r in rows):
        raise AuditStop("summary requires complete PASS rows")
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise AuditStop("duplicate image rows")
    x = [finite(r["selected_psnr"], "psnr") for r in rows]
    gaps = [finite(r["regret_db"], "regret") for r in rows]
    return {"n": len(x), "mean":mean(x), "median":median(x), "q05":quantile(x,.05),
            "min":min(x), "good25":sum(v>=GOOD for v in x),
            "bad20":sum(v<BAD for v in x), "mean_regret":mean(gaps),
            "q95_regret":quantile(gaps,.95), "exact_oracle_matches":sum(g<=1e-9 for g in gaps)}

def paired(a: list[dict], b: list[dict], *, seed: int = 5401, reps: int = 10000,
           allow_different_k: bool = False) -> dict:
    """Paired bootstrap/McNemar. Cross-family unequal-K comparison is cost-UNMATCHED."""
    if len(a) != len(b) or len(a) < 2:
        raise AuditStop("paired comparison needs same nontrivial full panel")
    aa = {r["image_id"]:r for r in a}
    bb = {r["image_id"]:r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):
        raise AuditStop("missing/duplicate paired image IDs")
    x, rescues, harms = [], 0, 0
    for image_id in sorted(aa):
        p,q = aa[image_id],bb[image_id]
        if (p["cohort"],p["representation"],p["measurement_sha256"]) != (
            q["cohort"],q["representation"],q["measurement_sha256"]):
            raise AuditStop(f"paired provenance/representation mismatch: {image_id}")
        if p["K"] != q["K"] and not allow_different_k:
            raise AuditStop(f"paired K mismatch without explicit cost-unmatched label: {image_id}")
        v,w = finite(p["selected_psnr"], "a"),finite(q["selected_psnr"], "b")
        x.append(v-w)
        rescues += v>=GOOD and w<GOOD
        harms += v<GOOD and w>=GOOD
    n = rescues + harms
    pval = min(1.0, 2*sum(comb(n,j) for j in range(min(rescues,harms)+1))/2**n) if n else 1.0
    rng = Random(seed)
    bootstrap = sorted(mean([x[rng.randrange(len(x))] for _ in x]) for _ in range(reps))
    return {"n":len(x), "compute_matched": not allow_different_k, "mean_delta":mean(x),
            "ci95_low":quantile(bootstrap,.025), "ci95_high":quantile(bootstrap,.975),
            "rescues":rescues, "harms":harms, "mcnemar_exact_p":pval}
