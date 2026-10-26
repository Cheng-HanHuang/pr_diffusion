#!/usr/bin/env python3
"""CPU-only finite-space and Gaussian checks of Track R math; synthetic only."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr
SEED=20261026

def tv(p,q):return float(np.sum(np.abs(p-q))/2)
def kl(p,q):return float(np.sum(p*np.log(p/q)))
def check_quotient():
    # Orbits are the rows, their two orientation representatives are columns.
    prior=np.array([[.45,.05],[.20,.30]])
    like=np.array([.4,.8])  # likelihood is constant within each orbit
    posterior=prior*like[:,None]; posterior/=posterior.sum()
    assert np.allclose(posterior/posterior.sum(axis=1,keepdims=True),prior/prior.sum(axis=1,keepdims=True))
    solver=np.array([[.25,.25],[.35,.15]])  # a nonposterior solver law
    lhs=kl(solver,posterior)
    marginal=kl(solver.sum(axis=1),posterior.sum(axis=1))
    inner=sum(solver[i].sum()*kl(solver[i]/solver[i].sum(),posterior[i]/posterior[i].sum())for i in range(2))
    assert abs(lhs-marginal-inner)<1e-13
    flipped=posterior[:,::-1]
    defect=tv(solver,solver[:,::-1]);delta=tv(posterior,flipped)
    assert 2*tv(solver,posterior)+defect+1e-14>=delta
    # Symmetric solver cannot match an asymmetric posterior too closely.
    sym=np.array([[.25,.25],[.25,.25]])
    assert tv(sym,posterior)+1e-14>=delta/2
    print('KL quotient split:',round(lhs,6),round(marginal,6),round(inner,6))
    print('posterior-orientation TV',round(delta,6),'solver-defect TV',round(defect,6))

def check_gaussian():
    m=np.array([.6,0.0]); gm=m[::-1]
    s=1.0
    # Exact TV between equal-covariance isotropic Gaussians.
    delta=np.linalg.norm(m-gm)
    exact=2*ndtr(delta/(2*s))-1
    chain_bound=delta/(2*s)
    assert exact<=chain_bound+1e-13
    print('Gaussian TV exact',round(exact,6),'Pinsker bound',round(chain_bound,6))

def fft(z,m=8):return np.abs(np.fft.fft(z,n=m,norm='ortho'))
def check_zero_flip():
    z=np.array([1.,5.,6.])/7
    v=np.array([2.,7.,3.])/7
    for m in (3,5,8,16): assert np.max(abs(fft(z,m)-fft(v,m)))<1e-14
    assert not np.array_equal(z,v[::-1])
    rng=np.random.default_rng(SEED)
    b=v.copy();b[-1]+=.08
    a=fft(z); c=fft(b); d=c-a
    sigma=.05
    pred=ndtr(np.linalg.norm(d)/(2*sigma))
    n=120000
    noise=sigma*rng.standard_normal((n,len(a)))
    observed=np.mean(np.sum(noise**2,axis=1)<np.sum((noise-d)**2,axis=1))
    assert abs(pred-observed)<.01
    assert np.max(abs(np.sum(noise**2,axis=1)-np.sum((noise+a-fft(v))**2,axis=1)))<1e-11
    print('zero-flip maxerr',np.max(abs(fft(z)-fft(v))),'noise rank predicted',round(pred,6),'MC',round(observed,6))

def check_selection():
    rng=np.random.default_rng(SEED)
    for i in range(10000):
        p,og,op,c,u=rng.dirichlet(np.ones(5)*.3)
        h,tg,tp=rng.uniform(size=3)
        s=1-u;q=p+og+op;a=(1-h)*p+tg*og+tp*op
        for K in [1,2,4,8,16,32,64]:
            gap_orbit_increment=q*(1-q)**K-a*(1-s)**K
            gap_raw_increment=p*((1-p)**K-(1-s)**K)
            assert gap_orbit_increment>=-1e-13 and gap_raw_increment>=-1e-13
            gener=u**K
            miss=(1-u**K)*(s-a)/s
            e1=(1-u**K)*a/s
            assert abs(1-e1-gener-miss)<1e-12
    print('selection: 10k random probability instances passed')

if __name__=='__main__':
    check_quotient();check_gaussian();check_zero_flip();check_selection()
    print('TH1_THEORY_BRIDGE_CPU_PASS seed',SEED)
