#!/usr/bin/env python3
"""Purely synthetic score-race/rare-decoy/TV-coupling tests (no XP or images).

Requires numpy and scipy. Reproducible seed=20261026.
Use this to verify TH1_SELECTION_TAILS_20261026.md; not a predictor fit.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad

SEED = 20261026
P, R, H = 0.700, 0.005, 0.295  # good / extreme wrong / ordinary wrong
assert abs(P+R+H-1) < 1e-14

def exact_good(K: int) -> float:
    return (1-R)**K-H**K

def F_good(t: float) -> float:
    return min(1., max(0., (t-.2)/.2))

def f_good(t: float) -> float:
    return 5. if .2 < t < .4 else 0.

def F_wrong(t: float) -> float:
    return ((R/(R+H))*min(1.,max(0.,t/.1))
            +(H/(R+H))*min(1.,max(0.,(t-.8)/.2)))

def score_race(K: int) -> float:
    return quad(lambda t: K*P*(1-P*F_good(t)-(R+H)*F_wrong(t))**(K-1)
                *f_good(t),.2,.4,epsabs=1e-12)[0]

def test_rare_tail() -> None:
    # Direct order-statistic integral equals the three-stratum event formula.
    for K in (1,2,3,4,5,8,16,32,64):
        assert abs(exact_good(K)-score_race(K))<1e-10
    grid=np.arange(1,301,dtype=int)
    assert grid[np.argmax([exact_good(int(k)) for k in grid])]==5
    assert abs(H/(R+H)-.9833333333333333)<1e-12
    assert exact_good(64)<exact_good(5)
    rng=np.random.default_rng(SEED)
    for K in (4,16,64):
        n=200000; wins=0
        for _ in range(0,n,5000):
            cls=rng.choice(3,size=(5000,K),p=[P,R,H])
            scores=rng.random(size=(5000,K))*.2+np.choose(cls,[.2,0.,.8])
            wins+=int(np.count_nonzero(
                cls[np.arange(5000),np.argmin(scores,axis=1)]==0))
        obs=wins/n;expected=exact_good(K)
        se=np.sqrt(expected*(1-expected)/n)
        assert abs(obs-expected)<5*se+1e-3,(K,obs,expected)
        print('K',K,'exact',round(expected,6),'MC',round(obs,6))

def test_tv_coupling() -> None:
    # Reference all-pass score F0=Uniform[0,1]. Good has F0, wrong has
    # shifted Uniform[eps,1+eps]. Their total-variation distance is eps.
    # Directly test the score-race bias against the coupling bound.
    pg,pw,eps=.7,.3,.02
    def Fg(t): return max(0.,min(1.,t))
    def Fw(t): return max(0.,min(1.,t-eps))
    def actual(K):
        return quad(lambda t: K*pg*(1-pg*Fg(t)-pw*Fw(t))**(K-1),
                    0.,1.,epsabs=1e-12,points=[eps])[0]
    for K in (1,2,4,8,16,64):
        bias=abs(actual(K)-pg)
        bound=1-(1-pw*eps)**K
        assert bias<=bound+1e-12,(K,bias,bound)
        print('TV bound',K,'bias',round(bias,7),'bound',round(bound,7))

def main() -> None:
    test_rare_tail()
    test_tv_coupling()
    print('TH1_SCORE_TAIL_CHECKS_PASS seed',SEED)

if __name__=='__main__':
    main()
