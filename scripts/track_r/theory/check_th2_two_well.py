#!/usr/bin/env python3
"""Exact two-well matched-budget contrasts (synthetic CPU only)."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
SEED=20261026
PHI=lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)

def reset_exact(J,c):
    p=ndtr(-c*math.sqrt(J))
    return 0.5*(1+(1-2*p)**J)

def sign_reset_test():
    J=4
    rng=np.random.default_rng(SEED)
    N=300000
    for c in (.3,1.0):
        b=1.; sigma=1/c
        state=np.full(N,b)
        for _ in range(J):
            state=b*np.where(state>=0,1.,-1.)+sigma/math.sqrt(J)*rng.standard_normal(N)
        actual=(state>=0).mean();exact=reset_exact(J,c);single=ndtr(c)
        stderr=math.sqrt(exact*(1-exact)/N)
        assert abs(actual-exact)<=5*stderr+1e-3,(c,exact,actual)
        if c==.3: assert exact<single-0.05
        else: assert exact>single+0.05
        print('RESET',J,'c',c,'inner',round(exact,8),'MC',round(actual,8),'outer',round(single,8))

def two_step_probability(a,c):
    affine=ndtr(-c/math.sqrt(1+a*a))
    delta=quad(lambda z:PHI(z)*(ndtr(-(2*a-1)*c-a*z)-ndtr(-c-a*z)),
               -math.inf,-c,epsabs=1e-12)[0]
    return affine+delta,affine,delta

def two_step_test():
    a=.2;c=1.
    exact,affine,delta=two_step_probability(a,c)
    assert delta>0
    p=ndtr(-c);h=2*(1-a)*c
    mu=c-h*p
    variance=1+a*a+h*h*p*(1-p)+2*h*a*PHI(c)
    gaussian_match=ndtr(-mu/math.sqrt(variance))
    assert abs(exact-gaussian_match)>0.015
    rng=np.random.default_rng(SEED+1)
    N=500000
    z1=rng.standard_normal(N);z2=rng.standard_normal(N)
    x1=c+z1
    x2=a*x1+(1-a)*c*np.where(x1>=0,1.,-1.)+z2
    y2=c+a*z1+z2
    assert np.all(x2<=y2+1e-14)
    actual=(x2<0).mean()
    stderr=math.sqrt(exact*(1-exact)/N)
    assert abs(actual-exact)<=5*stderr+1e-3,(actual,exact)
    assert (x2<0).mean()>(y2<0).mean()+0.07
    print('TWO_STEP','nonlinear',round(exact,8),'MC',round(actual,8),
          'affine',round(affine,8),'moment matched gaussian',round(gaussian_match,8))

def main():
    sign_reset_test();two_step_test()
    print('TH2_TWO_WELL_CPU_PASS',SEED)
if __name__=='__main__':main()
