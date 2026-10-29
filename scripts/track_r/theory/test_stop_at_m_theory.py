#!/usr/bin/env python3
"""Synthetic validation of exact stop-at-M moments, selection formulas,
       and an equal-compute heterogeneity counterexample.
       No Track R XP data or GT images read.
"""
import math, itertools, random


def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0

def pr_binom(k,j,s):
    return choose(k,j)*s**j*(1-s)**(k-j)

def mean_runs(N,M,s):
    if not 0<=s<=1 or not 1<=M<=N:raise ValueError('parameters')
    return sum(sum(pr_binom(t,j,s) for j in range(min(M-1,t)+1)) for t in range(N))

def mean_runs_m2_closed(N,s):
    if s==0:return N
    r=1-s
    return (2-N*r**(N-1)+(N-2)*r**N)/s

def stop_success(N,M,s,Hplus,Hminus):
    return sum(pr_binom(N,j,s)*(Hplus[M] if j>=M else Hplus[j] if j>0 else Hminus[N]) for j in range(N+1))

def fixed_success(k,s,Hplus,Hminus):
    return sum(pr_binom(k,j,s)*(Hplus[j] if j else Hminus[k]) for j in range(k+1))

def interp_fixed(real_k,s,Hplus,Hminus):
    lo=math.floor(real_k);hi=math.ceil(real_k);t=real_k-lo
    return (1-t)*fixed_success(lo,s,Hplus,Hminus)+t*fixed_success(hi,s,Hplus,Hminus)

def pl_two_class_pass_success(j,q,w):
    if j==0:return 0.
    return sum(pr_binom(j,g,q)*g/(g+(j-g)*w) for g in range(1,j+1))

def exhaustive_indicator_mark(N,M,s,good_given_pass):
    """Enumerate iid PASS-GOOD, PASS-BAD, FAIL-BAD run strings under
       exchangeable conditional residual ranks; each pass winner GOOD with
       probability good_given_pass, independently of how many passes."""
    total=0.; mean_t=0.; fc=0.
    for flags in itertools.product((0,1),repeat=N):
        weight=math.prod(s if x else 1-s for x in flags)
        t=N; successes=0
        for i,x in enumerate(flags,1):
            successes+=x
            if successes>=M:t=i;break
        total+=weight*(good_given_pass if successes>0 else 0)
        mean_t+=weight*t
        fc+=weight*(successes==0)
    return total,mean_t,fc

def test():
    tests=0
    for N in range(2,10):
        for M in range(1,N+1):
            for s in (0.,.03,.2,.41,.8,.99,1.):
                b=mean_runs(N,M,s)
                assert M-1e-10<=b<=N+1e-10
                if M==2:
                    assert abs(b-mean_runs_m2_closed(N,s))<1e-9,(N,s,b)
                    for q in (.2,.8):
                        if N<=8:
                            exact=exhaustive_indicator_mark(N,2,s,q)
                            hp=[0]+[q]*N;hm=[0]*(N+1)
                            val=stop_success(N,2,s,hp,hm)
                            assert abs(exact[0]-val)<1e-10
                            assert abs(exact[1]-b)<1e-10
                            assert abs(exact[2]-(1-s)**N)<1e-10
                tests+=1
    # Heterogeneity alone does NOT imply an advantage: one image is
    # essentially unsolvable and another profits from more than two passes.
    N=16;M=2;bad_s=.02;good_s=.9
    K=(mean_runs(N,M,bad_s)+mean_runs(N,M,good_s))/2
    zero=[0.]* (N+1)
    hp=[pl_two_class_pass_success(j,.5,.1) for j in range(N+1)]
    stopped=(stop_success(N,M,bad_s,zero,zero)+stop_success(N,M,good_s,hp,zero))/2
    fixed=(interp_fixed(K,bad_s,zero,zero)+interp_fixed(K,good_s,hp,zero))/2
    assert K>8 and stopped<fixed-.08
    print('PASS',tests,'tail-sum, M2 closed form, brute-force stopping and flag tests')
    print('COUNTEREXAMPLE: s=(.02,.90), N=16 M=2 mean K=%.9f stop=%.9f fixed=%.9f gain=%+.9f'%(K,stopped,fixed,stopped-fixed))
    # Binary C-share limit: all G,C pass, I fails. Class equal score.
    p=.3;N=16;M=2
    for alpha in (.0,.25,.5,.75,.95,.999):
        c=(1-p)*alpha;u=(1-p)*(1-alpha);s=1-u
        q=p/s
        HP=[0]+[q]*N;HM=[0]*(N+1)
        K=mean_runs(N,M,s)
        gain=stop_success(N,M,s,HP,HM)-interp_fixed(K,s,HP,HM)
        assert gain>=-1e-12
        print('C-SHARE alpha=%.3f s=%.4f equal_runs=%.4f gain=%+.7f'%(alpha,s,K,gain))
    assert mean_runs(16,2,1)==2
if __name__=='__main__':test()
