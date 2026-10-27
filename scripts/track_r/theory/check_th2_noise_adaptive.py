#!/usr/bin/env python3
"""Purely synthetic check of Gaussian fixed-candidate discrimination and
noise-adaptive residual optimism. No benchmark images / XP files / PAC."""
import math
import numpy as np
from scipy.special import ndtr

SEED = 20261026

def fixed_pair():
    sigma=.05; m=3*384*384
    delta=5*sigma
    residual_root=math.sqrt(sigma*sigma+delta*delta/m)
    test_correct=ndtr(delta/(2*sigma))
    assert abs(residual_root-0.0500014122)<1e-8
    assert 0.993<test_correct<0.995
    assert abs(delta**2/(2*sigma**2)-12.5)<1e-12
    return residual_root,test_correct

def linear_model():
    rng=np.random.default_rng(SEED)
    m,d=64,22;sigma=.05
    J=rng.standard_normal((m,d))
    U,S,_=np.linalg.svd(J,full_matrices=False)
    P=U@U.T
    lam=3.
    H=(U*(S*S/(S*S+lam)))@U.T
    N=20000
    Z=rng.standard_normal((m,N))
    Znew=rng.standard_normal((m,N))
    raw=sigma*sigma*np.mean(np.sum(Z*Z,axis=0))
    lsres=sigma*sigma*np.mean(np.sum(((np.eye(m)-P)@Z)**2,axis=0))
    ridge=sigma*sigma*np.mean(np.sum(((np.eye(m)-H)@Z)**2,axis=0))
    signal=sigma*sigma*np.mean(np.sum((H@Z)**2,axis=0))
    newres=sigma*sigma*np.mean(np.sum((Znew-H@Z)**2,axis=0))
    assert abs(raw-sigma*sigma*m)<.015
    assert abs(lsres-sigma*sigma*(m-d))<.015
    assert abs(ridge-sigma*sigma*np.trace((np.eye(m)-H)@(np.eye(m)-H)))<.015
    assert abs(signal-sigma*sigma*np.trace(H@H))<.015
    assert abs(newres-sigma*sigma*(m+np.trace(H@H)))<.015
    assert abs(newres-ridge-2*sigma*sigma*np.trace(H))<.018
    assert np.linalg.norm(P@Z+(np.eye(m)-P)@Z-Z)<1e-10
    print('RANK',d,'M',m,'RIDGE_DF',np.trace(H),'train_error',ridge,'replicate_error',newres)

def main():
    r,p=fixed_pair()
    linear_model()
    print('FIXED_PAIR_RMS',r,'PAIRWISE_CORRECT_PROB',p)
    print('TH2_NOISE_ADAPTIVE_CHECK_PASS seed',SEED)

if __name__=='__main__':main()
