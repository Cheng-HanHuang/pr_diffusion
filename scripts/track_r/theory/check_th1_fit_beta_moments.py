#!/usr/bin/env python3
"""Regression tests for posterior-predictive TH1 fitter; synthetic CPU only.

Run from repository root:
python scripts/track_r/theory/check_th1_fit_beta_moments.py
No A8, XP, PAC, images, or ground-truth data are loaded.
"""
from __future__ import annotations
import numpy as np
from fit_th1_dirichlet import beta_zero, curves_conditional, curves_given_theta

SEED=20261026

def main()->None:
    # Noninteger positive Dirichlet pseudo-counts stress the posterior moments.
    B=np.array([[1.,.5,.25,.25,8.],
                [3.,1.,1.,1.,.5],
                [.5,.5,.5,.5,.5]],dtype=float)
    n=B.shape[0]
    C=np.zeros_like(B)
    t=B.sum(axis=1)
    p,og,op,c,u=B.T
    rng=np.random.default_rng(SEED)
    assert np.allclose(beta_zero(t-u,t,1),u/t)

    # Identity: with one run, selection equals the single run's class mass.
    one=curves_conditional(C,B,1)
    assert np.allclose(one[:,0],p/t)
    assert np.allclose(one[:,1],(p+og)/t)
    assert np.allclose(one[:,2],p/t)
    assert np.allclose(one[:,3],(p+og+op)/t)

    # K=2 has exact low-degree Dirichlet moments.
    two=curves_conditional(C,B,2)
    E_pu=p*u/(t*(t+1))
    E_g_u=(p+og)*u/(t*(t+1))
    assert np.allclose(two[:,0],p/t+E_pu)
    assert np.allclose(two[:,1],(p+og)/t+E_g_u)

    # Posterior Monte Carlo gives an independent check for K as large as 64.
    samples=120000
    raw=rng.gamma(shape=B[None,:,:],size=(samples,n,5))
    theta=raw/raw.sum(axis=2,keepdims=True)
    for K in (1,2,4,8,16,32,64):
        predicted=curves_conditional(C,B,K)
        observed=np.array([curves_given_theta(theta[i],K).mean(axis=0)
                           for i in range(2000)])
        # Batch-wise MC of 2000 draws is less expensive; 3 images each.
        mean=observed.mean(axis=0)
        stderr=observed.std(axis=0,ddof=1)/np.sqrt(len(observed))
        assert np.all(np.abs(mean-predicted.mean(axis=0)) <
                      6*stderr+1e-3),(K,mean,predicted.mean(axis=0))

    # Show the original bug: replacing E[u^K] with E[(1-u)^K]
    # fails the K=1 identity except at accidental parameter equality.
    wrong=(1-beta_zero(u,t,1))*p/(t-u)
    assert not np.allclose(wrong,p/t)
    print('TH1_POSTERIOR_BETA_MOMENTS_PASS seed',SEED)

if __name__=='__main__':
    main()
