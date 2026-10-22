"""CPU-only retrospective R0 labels, AUROC, paired bootstrap and policies.

GT quantities are restricted to offline evaluation; workers must not import this
module. Policies E0–E3 use only measurement/prior/UnivFD values.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import numpy as np


def label_candidate(raw_db, orbit_db, signed_residual, tau, good=25.0):
    if raw_db >= good:
        return "GOOD"
    best = max(orbit_db, key=lambda s: orbit_db[s])
    if best != "000" and orbit_db[best] >= good:
        return "ORBIT_ERROR"
    if tau is not None and signed_residual <= tau:
        return "NONORBIT_CONSISTENT"
    return "NONORBIT_INCONSISTENT"


def tau_from_good(rows, source, family):
    values = [float(r["rho_signed_raw"]) for r in rows
              if r["source"]==source and r["family"]==family
              and r["psnr_raw_db"]>=25]
    return float(np.quantile(values,0.95)) if values else None


def auc_binary(labels, scores):
    """Higher score means more likely failure; ties get average ranks."""
    y = np.asarray(labels,dtype=np.int64)
    s = np.asarray(scores,dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("invalid AUROC inputs")
    n1 = int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0:
        return None
    order = np.argsort(s,kind="stable")
    ranks = np.empty(len(y),dtype=float)
    i=0
    while i<len(y):
        j=i+1
        while j<len(y) and s[order[j]]==s[order[i]]:
            j+=1
        ranks[order[i:j]] = (i+1+j)/2
        i=j
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n0*n1))


def seed_bootstrap(analysis_id):
    payload=("R0BOOT|"+analysis_id).encode("utf-8")
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")
    return np.random.default_rng(seed)


def bootstrap_cluster_rate(rows, predicate, analysis_id, replicates=10000):
    """One sample per image key with replacement; retain all runs in a cluster."""
    groups = {}
    for r in rows:
        key=(r["source"],r["family"],str(r["image_id"]))
        groups.setdefault(key,[]).append(r)
    clusters=list(groups.values())
    if not clusters:
        return {"n_images":0,"n_runs":0,"rate":None,"ci95":None}
    numer = np.array([sum(bool(predicate(r)) for r in v) for v in clusters],dtype=float)
    denom = np.array([len(v) for v in clusters],dtype=float)
    rng=seed_bootstrap(analysis_id)
    i=rng.integers(0,len(clusters),size=(replicates,len(clusters)))
    reps=numer[i].sum(axis=1)/denom[i].sum(axis=1)
    return {"n_images":len(clusters),"n_runs":int(denom.sum()),
            "rate":float(numer.sum()/denom.sum()),
            "ci95":list(map(float,np.quantile(reps,[0.025,0.975])))}


def _argmin(rows, key):
    if not rows:
        raise ValueError("empty candidate list")
    if any(key not in r or r[key] is None or not math.isfinite(float(r[key])) for r in rows):
        raise ValueError("missing candidate score " + key)
    return min(enumerate(rows),key=lambda iv:(float(iv[1][key]),iv[0]))[1]


def select_policy(rows, name, tau):
    """GT-free E0–E3; stable recorded-order tie breaking.

    Rows are in recorded run order; each carries prior-repaired and raw scores.
    Return (chosen row, repaired?), never inspect PSNR during selection.
    """
    if not rows:
        raise ValueError("empty policy input")
    ordered=sorted(rows,key=lambda r:r["run_index"])
    if name=="E0":
        return _argmin(ordered,"rho_signed_raw"),False
    if name=="E1":
        return _argmin(ordered,"rho_signed_raw"),True
    if name not in ("E2","E3"):
        raise ValueError("unknown executable policy")
    if tau is None:
        raise ValueError("undefined source-family feasibility tau")
    feasible=[r for r in ordered if r["rho_signed_repaired"] <= tau]
    if not feasible:
        return _argmin(ordered,"rho_signed_repaired"),True
    score="S_repaired" if name=="E2" else "p_fake_repaired"
    return _argmin(feasible,score),True


def evaluate_policy(rows, name, tau):
    if name=="O_raw":
        return max(float(r["psnr_raw_db"]) for r in rows)
    if name=="O_orbit":
        return max(float(v) for r in rows for v in r["psnr_orbit_db"].values())
    chosen, repaired = select_policy(rows,name,tau)
    return float(chosen["psnr_repaired_db"] if repaired else chosen["psnr_raw_db"])


def tally_labels(rows):
    return dict(Counter(r["label"] for r in rows))


def summarize_psnrs(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite PSNR table")
    return {"n":len(x),"mean":float(x.mean()),"median":float(np.median(x)),
            "good25":float((x>=25).mean()),"bad20":float((x<20).mean()),
            "bad15":float((x<15).mean()),
            "q05":float(np.quantile(x,0.05)),"min":float(x.min())}
