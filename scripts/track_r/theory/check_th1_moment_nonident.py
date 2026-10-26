#!/usr/bin/env python3
"""Synthetic truncated-moment obstruction (no real data or XP access)."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

def check(n: int,Kgrid:tuple[int,...])->None:
    deg=n+1
    def P(x):return float(eval_legendre(deg,2*x-1))
    assert abs(quad(P,0,1,epsabs=1e-13)[0])<1e-12
    for k in range(n+1):
        assert abs(quad(lambda x:x**k*P(x),0,1,epsabs=1e-13)[0])<1e-12
    for K in Kgrid:
        integr=quad(lambda p:(1-p)**K*P(p),0,1,epsabs=1e-13)[0]
        analytic=((-1)**deg*
                  math.prod(range(K-deg+1,K+1))/
                  math.prod(range(K+1,K+deg+2)) if K>=deg else 0.)
        assert abs(integr-analytic)<1e-12
        print(f'n={n} K={K} two-prediction spread={2*abs(integr):.10f}')

def main()->None:
    check(4,(1,2,4,8,16,32,64))
    check(2,(1,2,4,8,16,32))
    print('TH1_TRUNCATED_MOMENT_CHECK_PASS')

if __name__=='__main__':
    main()
