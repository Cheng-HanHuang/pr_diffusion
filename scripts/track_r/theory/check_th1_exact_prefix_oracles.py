#!/usr/bin/env python3
"""CPU-only numerical regression for exact A6 fixed-prefix oracle expectations.

Synthetic rows and beta-Dirichlet marginals only; no XP returns, FFHQ
images, or A8b. Run with this file's directory in sys.path.
"""
import math
import numpy as np
from fit_th1_prefix_v1 import exact_prefix_oracles,pass_at_k

def run():
    rng=np.random.default_rng(20261027)
    cases=[
      ((1,0,0,0,1),2,8,(1,2,4,6,8)),
      ((0,0,0,2,0),2,8,(1,2,4,6,8)),
      ((0,1,0,0,3),4,16,(1,2,8,12,16)),
      ((1,0,1,0,2),4,64,(4,8,16,32,64)),
      ((0,0,0,0,4),4,4,(1,2,4))
    ]
    alpha=np.array((1.2,.7,.15,2.1,1.8),dtype=float)
    for c,n0,nmax,K in cases:
        row={"counts":c,"n":n0}
        exact=exact_prefix_oracles(row,alpha,K,nmax)
        assert np.all((exact>=0)&(exact<=1))
        for i,(gg,q) in enumerate(exact):
            assert q+1e-12>=gg
            if i:
                assert exact[i,0]>=exact[i-1,0]-1e-12
                assert exact[i,1]>=exact[i-1,1]-1e-12
        if n0==nmax:
            assert abs(exact[-1,0]-(int(c[0]>0)))<1e-12
            assert abs(exact[-1,1]-(int(sum(c[:3])>0)))<1e-12
        # Conditional full-panel draws, no old-rank simulation necessary
        # because oracle scores depend only on labels.
        old=np.concatenate([np.full(z,j,dtype=int) for j,z in enumerate(c)])
        ntrial=100000 if nmax<20 else 25000
        vals=np.zeros((len(K),2))
        for _ in range(ntrial):
            theta=rng.dirichlet(np.asarray(c)+alpha)
            future=rng.choice(5,size=nmax-n0,p=theta)
            labels=np.concatenate((old,future))
            ng=int(np.count_nonzero(labels==0))
            nq=int(np.count_nonzero(labels<=2))
            vals+=np.array([[pass_at_k(nmax,ng,k),pass_at_k(nmax,nq,k)]
                            for k in K])
        observed=vals/ntrial
        assert np.max(np.abs(exact-observed))<.006, (c,nmax,exact,observed)
    print("TH1_EXACT_PREFIX_ORACLE_PASS cases",len(cases))

if __name__=="__main__":
    run()
