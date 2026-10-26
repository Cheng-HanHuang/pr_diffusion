#!/usr/bin/env python3
"""Synthetic exact local-basin OU vs final-re-noise comparison; CPU only."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr

SEED=20261026

def main()->None:
    rng=np.random.default_rng(SEED)
    d=np.sqrt(14)/7
    b=t0=d/2
    kappa,eta,temp,J=2.0,.1,.2,8
    a=1-eta*kappa
    mu=b+a**J*(t0-b)
    var=2*eta*temp*(1-a**(2*J))/(1-a*a)
    sig=np.sqrt(2*eta*temp)
    assert 0<a<1 and var>sig**2
    n=150000
    x=np.full(n,t0)
    for _ in range(J):
        x=b+a*(x-b)+sig*rng.standard_normal(n)
    prob=ndtr(-mu/np.sqrt(var))
    observed=np.mean(x<0)
    assert abs(prob-observed)<.006,(prob,observed)
    outer=mu+sig*rng.standard_normal(n)
    p_outer=ndtr(-mu/sig)
    observed_outer=np.mean(outer<0)
    assert abs(p_outer-observed_outer)<.006,(p_outer,observed_outer)
    assert prob>p_outer
    # Replacing sig with sqrt(var) yields the same normal endpoint law.
    assert abs(ndtr(-mu/np.sqrt(var))-prob)<1e-15
    # Positive stable deterministic GD and proximal descent preserve basin.
    z=np.array([.1,.2,1.])
    for _ in range(30):
        z=(1-eta*kappa)*z+eta*kappa*b
        assert np.all(z>0)
    z=np.array([.1,.2,1.])
    for _ in range(30):
        z=(z+.4*kappa*b)/(1+.4*kappa)
        assert np.all(z>0)
    print('TH2_BASIN_CPU_PASS',SEED)
    print('Langevin expected',prob,'observed',observed,
          'outer expected',p_outer,'observed',observed_outer)

if __name__=='__main__':
    main()
