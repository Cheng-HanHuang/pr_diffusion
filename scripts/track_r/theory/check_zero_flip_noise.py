#!/usr/bin/env python3
"""Synthetic 1D Fourier zero-flip + noise-ranking identities; CPU only.

No benchmark image or XP data are read. All randomness uses a fixed seed.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr

SEED=20261026

def magnitude(z: np.ndarray, m: int=8) -> np.ndarray:
    assert z.ndim==1 and len(z)<=m
    return np.abs(np.fft.fft(np.pad(z,(0,m-len(z))),norm='ortho'))

def pairwise_accuracy(a: np.ndarray,b: np.ndarray,sigma: float) -> float:
    delta=np.linalg.norm(b-a)
    return float(ndtr(delta/(2*sigma))) if delta>0 else .5

def main() -> None:
    z=np.array([1.,5.,6.])/7.
    v=np.array([2.,7.,3.])/7.
    assert np.all((0<=z)&(z<=1)) and np.all((0<=v)&(v<=1))
    assert not np.allclose(z,v) and not np.allclose(z,v[::-1])
    assert np.allclose(np.convolve([1,2],[1,3]),7*z)
    assert np.allclose(np.convolve([2,1],[1,3]),7*v)
    for nfft in (3,5,8,16,64):
        assert np.max(abs(magnitude(z,nfft)-magnitude(v,nfft))) < 1e-14
    rng=np.random.default_rng(SEED)
    sigma=.05
    nrep=100000
    for perturb in (.004,.02,.08,.2):
        vp=v.copy()
        vp[-1]+=perturb
        a=magnitude(z,8)
        b=magnitude(vp,8)
        delta=np.linalg.norm(b-a)
        noise=sigma*rng.standard_normal((nrep,8))
        score_a=np.sum(noise**2,axis=1)
        score_b=np.sum((noise+(a-b))**2,axis=1)
        empirical=np.mean(score_a<score_b)
        theoretical=pairwise_accuracy(a,b,sigma)
        se=math.sqrt(theoretical*(1-theoretical)/nrep)
        assert abs(empirical-theoretical)<=5*se+1e-3,(perturb,empirical,theoretical)
        print(f'eps={perturb:.3f} delta={delta:.6f} sigma={sigma:.3f} '
              f'prediction={theoretical:.6f} MC={empirical:.6f} SE={se:.5f}')
    # For an exact zero-flip pair, squared residuals are equal for every noisy y.
    a=magnitude(z,8); b=magnitude(v,8)
    noise=sigma*rng.standard_normal((100,8))
    assert np.max(abs(np.sum((noise+a-a)**2,axis=1)-
                          np.sum((noise+a-b)**2,axis=1)))<1e-13
    # Multiple correlated decoys: K-way success is not a product of pairwise
    # probabilities, unless extra independence geometry is given.
    v1=v.copy(); v1[-1]+=.08
    v2=v.copy(); v2[0]+=.12
    ds=np.stack((magnitude(v1)-a,magnitude(v2)-a))
    noise=sigma*rng.standard_normal((nrep,8))
    indicators=(2*noise@ds.T<np.sum(ds**2,axis=1))
    all_wins=float(np.mean(np.all(indicators,axis=1)))
    individual=np.mean(indicators,axis=0)
    covariance=float(ds[0]@ds[1])
    print('multi_decoy_all_win',round(all_wins,6),
          'individually',np.round(individual,6).tolist(),
          'pairwise_product',round(float(np.prod(individual)),6),
          'd1_dot_d2',round(covariance,8))
    assert abs(all_wins-float(np.prod(individual)))>0.02
    print('TH2_ZERO_FLIP_NOISE_CHECK_PASS seed',SEED)

if __name__=='__main__':
    main()
