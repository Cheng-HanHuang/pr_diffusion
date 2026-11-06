#!/usr/bin/env python3
"""Seeded CPU-only exact subset and symmetry-model checks. No image or XP access."""
import argparse
import itertools
import math
import numpy as np

def choose(n,k):
    return math.comb(n,k) if 0 <= k <= n else 0

def selected_weights(n,k,kind):
    den=choose(n,k)
    if kind=="min":
        return np.array([choose(n-r-1,k-1)/den for r in range(n)])
    return np.array([choose(r,k-1)/den for r in range(n)])

def brute(scores, values, k, kind):
    ans=[]
    for subset in itertools.combinations(range(len(scores)),k):
        winner=(min if kind=="min" else max)(subset,key=lambda j:(scores[j],j if kind=="min" else -j))
        ans.append(values[winner])
    return np.mean(ans)

def check(seed):
    rng=np.random.default_rng(seed)
    for n in range(2,10):
        scores=rng.permutation(n)
        values=rng.normal(size=n)
        for k in range(1,n+1):
            for kind in ("min","max"):
                ranks=np.argsort(scores)
                if kind=="max": ranks=np.argsort(scores)
                analytic=np.dot(selected_weights(n,k,kind),values[ranks])
                empirical=brute(scores,values,k,kind)
                assert np.isclose(analytic,empirical,atol=1e-12),(n,k,kind)
    for _ in range(1000):
        a=rng.dirichlet(np.ones(5))
        p,og,op,c,u=a
        q=p+og+op;s=q+c
        for k in range(1,32):
            gap=lambda j: 1-(1-q)**j-(p+og)/s*(1-(1-s)**j)
            assert gap(k+1)+1e-12>=gap(k)
    print("PASS: exact subset weights and monotone-gap identities; seed",seed)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=20261106)
    check(parser.parse_args().seed)
