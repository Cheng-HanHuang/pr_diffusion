#!/usr/bin/env python3
"""Exact-moment-equivalent populations with incompatible large-K coverage."""
import numpy as np

def check(n, x, w, targets):
    assert np.all(w > 0) and np.all((0 <= x) & (x <= 1))
    for j in range(n+1):
        moment = np.sum(w*x**j)
        assert abs(moment-1/(j+1)) < 1e-12, (n,j,moment)
    # Every possible binomial count among n runs is indistinguishable.
    from math import comb
    for g in range(n+1):
        cont=1/(n+1)
        discr=comb(n,g)*np.sum(w*(x**g)*(1-x)**(n-g))
        assert abs(discr-cont)<1e-12,(n,g,discr,cont)
    for K in targets:
        uniform=K/(K+1)
        discrete=1-np.sum(w*(1-x)**K)
        print('n',n,'K',K,'uniform_oracle',round(uniform,9),
              'discrete_oracle',round(discrete,9),
              'gap_percentage_points',round(100*(uniform-discrete),5))
    print('mass of never-successful images',w[0])

z=np.sqrt(6)
check(4, np.array([0.,(6-z)/10,(6+z)/10]),
         np.array([1/9,(16+z)/36,(16-z)/36]), (4,8,16,32,64,256))
check(2, np.array([0.,2/3]),np.array([1/4,3/4]),(2,4,8,16,64))
print('TH1_RADAU_OBSTRUCTION_PASS')
