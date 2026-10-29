#!/usr/bin/env python3
"""Independent exact finite-panel stop-at-M QA for the frozen protocol.

No external datasets or Track R XP outcomes are read. The reference
protocol's 4,000 random orders are NOT altered; this checks their expected
success, compute, and FLAG_C rates under finite-population randomization.
"""
import itertools,math,random


def comb(n,k):
    return math.comb(n,k) if 0<=k<=n else 0


def hypergeo(n,q,t,j):
    return comb(q,j)*comb(n-q,t-j)/comb(n,t)


def weight(q,j,r):
    return comb(q-r-1,j-1)/comb(q,j)


def winner_on_uniform_subset(candidates,j):
    """Candidates are (rho,corrected_success) pairs, ordered by rho."""
    if j<1 or j>len(candidates):raise ValueError('bad subset size')
    q=len(candidates)
    return sum(weight(q,j,r)*x[1] for r,x in enumerate(candidates))


def exact_panel(values,tau,M,N):
    """Exact expected (success, runs, FLAG_C) for random orders.
    values: fixed full panel (rho, corrected_success) pairs.
    N <= n, M <= N. Ties follow original index order, like stable sort.
    """
    n=len(values)
    if not (1<=M<=N<=n):raise ValueError('bad panel')
    passing=sorted((x for i,x in enumerate(values) if x[0]<=tau),key=lambda x:x[0])
    failing=sorted((x for i,x in enumerate(values) if x[0]>tau),key=lambda x:x[0])
    q=len(passing)
    # First M pass marks are uniformly distributed among all q passing
    # candidates, conditional on at least M passes in the N-prefix.
    success=0.
    for j in range(N+1):
        p=hypergeo(n,q,N,j)
        if p==0:continue
        if j>=M:success+=p*winner_on_uniform_subset(passing,M)
        elif j>0:success+=p*winner_on_uniform_subset(passing,j)
        else:success+=p*winner_on_uniform_subset(failing,N)
    mean_runs=sum(sum(hypergeo(n,q,t,j)for j in range(min(M-1,t)+1))for t in range(N))
    flag=hypergeo(n,q,N,0)
    if N==n:
        direct_runs=M*(n+1)/(q+1)if q>=M else n
        assert abs(mean_runs-direct_runs)<1e-10,(n,q,M,mean_runs,direct_runs)
        assert abs(flag-(1 if q==0 else 0))<1e-10
    return (success,mean_runs,flag)


def brute_panel(values,tau,M,N):
    n=len(values);tot=0;S=T=F=0.
    for perm in itertools.permutations(range(n)):
        taken=[];count=0
        for ii in perm[:N]:
            taken.append(ii);count+=values[ii][0]<=tau
            if count>=M:break
        best=min(taken,key=lambda ii:(values[ii][0],ii))
        S+=values[best][1];T+=len(taken)
        F+=not any(values[ii][0]<=tau for ii in taken)
        tot+=1
    return (S/tot,T/tot,F/tot)


def run():
    rng=random.Random(20261029);cases=0
    for n in (3,4,5,6,7):
        for M in (1,2,3):
            if M>n:continue
            for N in set([M,n,max(M,n-1)]):
                for q in set([0,1,2,n-1,n]):
                    if q>n:continue
                    scores=[(.10+.01*i,float(rng.randint(0,1)))for i in range(q)]+[(.60+.01*i,float(rng.randint(0,1)))for i in range(n-q)]
                    rng.shuffle(scores)
                    expected=exact_panel(scores,.5,M,N)
                    observed=brute_panel(scores,.5,M,N)
                    if max(abs(a-b)for a,b in zip(expected,observed))>1e-10:
                        raise AssertionError((n,M,N,q,expected,observed))
                    cases+=1
    print('TH_PROTOCOL_FINITE_PANEL_EXACT_PASS',cases,'synthetic exhaustive random-order cases')
    # Example: two pass candidates in full 16-run panel, stop M=2,
    # expected last pass position is 2*(16+1)/(2+1)=11 1/3.
    values=[(.1,1),(.2,0)]+[(1.+i*.01,0)for i in range(14)]
    s,t,f=exact_panel(values,.5,2,16)
    assert abs(t-34/3)<1e-10 and f==0
    print('N=16 L=2 M=2: exact mean stopping runs',t,'success',s,'FLAG_C',f)
if __name__=='__main__':run()
