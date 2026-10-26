"""Exact order-statistic estimators for XP; no outcome access at import."""
from __future__ import annotations
from math import comb
from collections import defaultdict
import random,hashlib

def choose(n,k):
    return comb(n,k) if 0 <= k <= n else 0

def argmin_weights(scores,k,run_indices=None):
    """Weights for each original run, selector ties broken by run index."""
    n=len(scores)
    if not 1<=k<=n:raise ValueError("invalid K")
    ix=list(range(n)) if run_indices is None else list(run_indices)
    if len(set(ix))!=n:raise ValueError("duplicate indices")
    order=sorted(range(n),key=lambda j:(scores[j],ix[j]))
    denom=choose(n,k)
    w=[0.]*n
    for rank,j in enumerate(order):
        w[j]=choose(n-rank-1,k-1)/denom
    assert abs(sum(w)-1)<1e-10
    return w

def argmax_weights(scores,k,run_indices=None):
    n=len(scores)
    if not 1<=k<=n:raise ValueError("invalid K")
    ix=list(range(n)) if run_indices is None else list(run_indices)
    order=sorted(range(n),key=lambda j:(scores[j],ix[j]))
    denom=choose(n,k);w=[0.]*n
    for rank,j in enumerate(order):
        w[j]=choose(rank,k-1)/denom
    assert abs(sum(w)-1)<1e-10
    return w

def success_oracle(n,g,k):
    if not 0<=g<=n or not 1<=k<=n:raise ValueError
    return 1-choose(n-g,k)/choose(n,k)

def expected_pick(values,weights):
    if len(values)!=len(weights):raise ValueError
    return sum(a*b for a,b in zip(values,weights))

def summarize_selector(rows,k,score_key,value_key,pass_value=None):
    """rows one image's run records; exact K-subset estimator."""
    scores=[float(r[score_key]) for r in rows]
    values=[float(r[value_key]) for r in rows]
    weights=argmin_weights(scores,k,[int(r["run_index"]) for r in rows])
    expectation=expected_pick(values,weights)
    success=None if pass_value is None else sum(w*(v>=pass_value) for w,v in zip(weights,values))
    return {"mean":expectation,"good":success}

def bootstrap_cluster(records,estimate,stage,replicates=10000):
    """Image cluster resampling; observations grouped by image id."""
    byid=defaultdict(list)
    for r in records:byid[str(r["image_id"])].append(r)
    ids=sorted(byid)
    if not ids:raise ValueError("empty")
    seed=int.from_bytes(hashlib.sha256(("XPBOOT|"+stage).encode()).digest()[:8],"big")
    rng=random.Random(seed)
    point=estimate([r for i in ids for r in byid[i]])
    values=[]
    for _ in range(replicates):
        sample=[rng.choice(ids) for __ in ids]
        values.append(estimate([r for i in sample for r in byid[i]]))
    values.sort()
    return {"n_images":len(ids),"estimate":point,
       "lo95":values[int(.025*(replicates-1))],
       "hi95":values[int(.975*(replicates-1))],
       "replicates":replicates}

def freeze_gate(registry_text,family,dataset,run_indices):
    """False means outcomes cannot be published, even if jobs completed."""
    if family=="FAST" and dataset=="imagenet" and any(i>=4 for i in run_indices):
        return ("v2" in registry_text) or ("48H_FALLBACK_EXPLICITLY_CLEARED" in registry_text)
    if dataset=="ffhq" and family in ("FAST","SITCOM","DAPS"):
        boundary=2 if family=="DAPS" else 4
        if any(i>=boundary for i in run_indices):return "v1" in registry_text
    return True
