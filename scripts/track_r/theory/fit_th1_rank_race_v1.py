#!/usr/bin/env python3
"""Track R TH1 v1 prospectively frozen class-race forecaster.

Only uses the authorized R0N2_PER_IMAGE_COUNTS.csv (K<=4).
No XP returns, no later run indices, no PAC dependencies and no images.
The ranks, counts, and checksum are verified by audit_a8_rank_race.py.

--output produces a DRAFT by default; --freeze writes v1 and requires
--data-commit and --code-commit. NEVER open K>4 outcomes before the
planner registers the frozen file checksum and commit.

Three predeclared variants use identical per-image Dirichlet posteriors:
  race: class-conditional exponential winning scores, weights fit on A8 ranks;
  S2: exchangeable residual ranks among passing classes;
  tail: two-component rare-extreme-C race fixed by 5/205 pair calibration.

Posterior intervals are conditional on this model family. They include
parameter/image-cluster bootstrap and independent finite-run panel
U-statistic noise, not distribution-free high-K identification bands.

Requires numpy scipy; no sklearn or GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

from audit_a8_rank_race import read_a8, crosscheck, fit_weights, FAMILIES
from fit_th1_dirichlet import fit as fit_alpha, curves_conditional

SEED=20261027
KGRID={"DAPS":(4,6,8),"SITCOM":(8,12,16,24,32),
       "FAST":(8,12,16,24,32,48,64)}
POLICIES=("E0","E1_CLIP","O_raw","O_orbit")
VARIANTS=("race","S2","tail")
CLASS="GOPCI"
COHORTS=("B22","DEV80","POOLED")
N_BY_COHORT={"B22":100,"DEV80":80}

def counts_matrix(rows):
    return np.asarray([r["counts"] for r in rows],dtype=float)

def posterior_samples(C,alpha,rng,samples):
    shapes=C[None,:,:]+alpha[None,None,:]
    z=rng.gamma(shape=shapes,size=(samples,len(C),C.shape[1]))
    return (z/z.sum(axis=2,keepdims=True)).transpose(1,0,2)

def race_probabilities(theta, weights, Kgrid, quad):
    """Mean over posterior theta draws, for each K; result shape (len(Kgrid),4)."""
    n,draws,d=theta.shape
    flat=theta.reshape(-1,d)
    nodes,qweights=quad
    arr=np.zeros((len(Kgrid),4),dtype=float)
    if len(weights)!=d:raise ValueError("weights/latent dimension mismatch")
    for j in range(d):
        if weights[j]<=0:continue
        exponents=weights/weights[j]
        # Same t_j score integration with z=exp(-w_j*t_j).
        shapes=nodes[None,:]**exponents[:,None]
        H=flat@shapes
        hprefix=flat[:,j]*weights[j]
        # Derivative in score transformed integral is K theta_j,
        # not K theta_j w_j; the Jacobian cancels the w_j.
        hprefix=flat[:,j]
        for ii,K in enumerate(Kgrid):
            win=K*hprefix*(H**(K-1)@qweights)
            if j==0:arr[ii,0]+=float(win.mean());arr[ii,1]+=float(win.mean())
            if j in (1,2):
                if j==1:arr[ii,1]+=float(win.mean())
    # The saved forecasts concern GOOD after global-only CLIP correction.
    # E0: class G wins; E1: class G or global O wins.
    # Oracles use theta-marginal power moments (below).
    thetaG=flat[:,0];thetaq=flat[:,0]+flat[:,1]+flat[:,2]
    for ii,K in enumerate(Kgrid):
        arr[ii,2]=float(np.mean(1-(1-thetaG)**K))
        arr[ii,3]=float(np.mean(1-(1-thetaq)**K))
    return arr

def variant_weights(fit_params,variant):
    wg=1.
    wo=fit_params["w_O"]
    wc=fit_params["w_C"]
    if variant=="S2":
        return np.array([1.,1.,1.,1.,0.]),None
    if variant=="race":
        return np.array([wg,wo,wo,wc,0.]),None
    if variant=="tail":
        # Approximate hard lower residual tail: super-fast exponential
        # C extreme scores, very slow ordinary C, compared with G.
        baseline=1e-6
        extreme=1000.
        target=5/205
        eps=(target-baseline/(1+baseline)) / (
            extreme/(1+extreme)-baseline/(1+baseline))
        eps=float(np.clip(eps,0.,1.))
        return np.array([wg,wo,wo,baseline,extreme,0.]),eps
    raise ValueError(variant)

def evaluate(theta,fit_params,variant,Kgrid,quad):
    weights,eps=variant_weights(fit_params,variant)
    if variant=="tail":
        # New class order G,O,P,C_ordinary,C_extreme,I.
        augmented=np.concatenate((theta[...,:3],
            theta[...,3:4]*(1-eps),
            theta[...,3:4]*eps,theta[...,4:5]),axis=2)
        theta=augmented
    return race_probabilities(theta,weights,Kgrid,quad)

def posterior_quantities(C,alpha,Kgrid):
    """Exact beta-moment reference for all four policies, conditional C."""
    return np.asarray([curves_conditional(C,alpha,K).mean(axis=0)
                       for K in Kgrid])

def image_bootstrap(rows,rng):
    return [rows[i] for i in rng.integers(0,len(rows),len(rows))]

def n_panel_noise(C,alpha,fit_params,variant,Kgrid,rng,nrep=160):
    """Simulate finite run-budget K-subset exact U statistics per fixed image.

    Draw one latent theta for each image per simulated panel, then nmax
    iid run class labels and exponential scores. Evaluate exact K-subset
    rank weights and pass@K *within* that finite panel, pooled across
    the fixed images. The model posterior is conditional on original
    counts; ranks are used through family-level race parameters.
    This does not pretend to condition on exact A8 physical residuals,
    which are unavailable before A8b.
    """
    nimg=len(C);nmax=max(Kgrid)
    weights,eps=variant_weights(fit_params,variant)
    out=np.empty((nrep,len(Kgrid),4))
    for b in range(nrep):
        theta=posterior_samples(C,alpha,rng,1)[:,0,:]
        if variant=="tail":
            theta=np.concatenate((theta[:,:3],theta[:,3:4]*(1-eps),
                         theta[:,3:4]*eps,theta[:,4:5]),axis=1)
        cumul=np.cumsum(theta,axis=1)
        u=rng.random((nimg,nmax))
        label=(u[:,:,None]>cumul[:,None,:]).sum(axis=2)
        # I is unselectable if any passing label exists; otherwise any I is bad.
        noise=rng.random((nimg,nmax))
        rate=weights[label]
        scores=np.full((nimg,nmax),1e12,dtype=float)
        mask=rate>0
        scores[mask]=-np.log(np.maximum(noise[mask],1e-300))/rate[mask]
        # Ties among all-I labels are harmless for E0/E1, since all fail.
        order=np.argsort(scores,axis=1)
        ranked=np.take_along_axis(label,order,axis=1)
        orig_G=(label==0).sum(axis=1)
        orig_Q=(label<=2).sum(axis=1)
        for ki,K in enumerate(Kgrid):
            # Exact U-stat weights: Pr(fixed residual rank r wins among
            # a uniform K-subset of nmax runs).
            rankw=np.array([
              math.comb(nmax-r-1,K-1)/math.comb(nmax,K)
              if nmax-r-1>=K-1 else 0.
              for r in range(nmax)])
            out[b,ki,0]=np.mean(np.sum((ranked==0)*rankw[None,:],axis=1))
            out[b,ki,1]=np.mean(np.sum((ranked<=1)*rankw[None,:],axis=1))
            out[b,ki,2]=np.mean([
              1-(math.comb(nmax-int(g),K)/math.comb(nmax,K)
                 if nmax-int(g)>=K else 0.)
              for g in orig_G])
            out[b,ki,3]=np.mean([
              1-(math.comb(nmax-int(g),K)/math.comb(nmax,K)
                 if nmax-int(g)>=K else 0.)
              for g in orig_Q])
    return out

def predict_frozen(rows,rng,posterior_mc,bootstrap,nquad,panels):
    N,W=leggauss(nquad);quad=((N+1)/2,W/2)
    data={(f,c):[row for row in rows if row["family"]==f
                   and row["cohort"]==c]
          for f in FAMILIES for c in ("B22","DEV80")}
    out=[];fits=[];crossover=[];shapes=[]
    for family in FAMILIES:
        fitparams=fit_weights([x for x in rows if x["family"]==family])
        strata={}
        for cohort in ("B22","DEV80"):
            C=counts_matrix(data[family,cohort])
            alpha=fit_alpha(C)
            theta=posterior_samples(C,alpha,rng,posterior_mc)
            vals={}
            for variant in VARIANTS:
                vals[variant]=evaluate(theta,fitparams,variant,KGRID[family],quad)
            # Force oracle + S2 reference to exact Dirichlet moments,
            # eliminating MC error in the analytical parts.
            reference=posterior_quantities(C,alpha,KGRID[family])
            vals["S2"]=reference
            for v in VARIANTS:
                vals[v][:,2:]=reference[:,2:]
            strata[cohort]={"C":C,"alpha":alpha,"points":vals}
            fits.append({"family":family,"cohort":cohort,
                         "concentration":alpha.tolist(),
                         "class_totals":C.sum(axis=0).astype(int).tolist(),
                         "rank_weights":fitparams})
        # Preserve fixed 100/80 cohort weights in pooled predictions.
        for variant in VARIANTS:
            for cohort in COHORTS:
                if cohort=="POOLED":
                    point=(strata["B22"]["points"][variant]*(100/180)+
                           strata["DEV80"]["points"][variant]*(80/180))
                    n=180
                else:
                    point=strata[cohort]["points"][variant]
                    n=len(strata[cohort]["C"])
                for ki,K in enumerate(KGRID[family]):
                    if family=="DAPS" and K==4 and cohort!="B22":
                        # This is a baseline observed K in DEV80, not
                        # a novel prospective DAPS/DEV80 prediction.
                        continue
                    for j,policy in enumerate(POLICIES):
                        out.append({"family":family,"dataset":"FFHQ",
                                    "cohort":cohort,"model_variant":variant,
                                    "policy":policy,"metric":"good25",
                                    "K":K,"n_images":n,
                                    "point":float(point[ki,j])})
        # Later bootstrap and predictive panel simulation below attach
        # interval fields consistently to all variant/policy records.
        for v in VARIANTS:
            shapes.append({"family":family,"model_variant":v,
               "statement":("monotone O_orbit-E1 gap"
                 if v=="S2" else "NO monotone-gap constraint")})
        # For audit generation definition: no correct-orbit candidate.
        # Selection loss is O_orbit-E1, and their sum is 1-E1.
    # Conservative model-conditional intervals: resample fixed image
    # class posterior and future panels in bootstrap work. We prefer
    # generating them reproducibly rather than inventing Gaussian CIs.
    #
    # Pooling/fixed-cohort bootstrap and joint fitted race uncertainty
    # will be added BEFORE --freeze; do not mark any draft as frozen
    # while 'lo95'/'hi95' are absent.
    return out,fits,shapes,crossover

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a8",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--posterior-mc",type=int,default=256)
    ap.add_argument("--nquad",type=int,default=128)
    ap.add_argument("--bootstrap",type=int,default=100)
    ap.add_argument("--panels",type=int,default=160)
    ap.add_argument("--data-commit")
    ap.add_argument("--code-commit")
    ap.add_argument("--freeze",action="store_true")
    args=ap.parse_args()
    if args.freeze:
        raise RuntimeError("v1 freeze blocked until parameter/bootstrap"
                           " uncertainty, panel U-stat intervals, and A8b"
                           " conditional gate are implemented and validated")
    rows,sha=read_a8(args.a8)
    crosscheck(rows)
    rng=np.random.default_rng(SEED)
    vals,fits,shapes,crossover=predict_frozen(
        rows,rng,args.posterior_mc,args.bootstrap,args.nquad,args.panels)
    obj={"version":"DRAFT_UNREGISTERED","model":
         "Image-conditional five-class Dirichlet-multinomial + regularized "
         "rank-calibrated exponential race, with S2 and rare-C sensitivity",
         "source_sha256":sha,"seed":SEED,
         "fit_data":["authorized A8 K<=4 ranks and class counts"],
         "predictions":vals,"fit_diagnostics":fits,
         "shape_predictions":shapes,"crossover_K":crossover,
         "uncertainty":"PENDING: no bands in unregistered draft"}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("TH1_RANK_RACE_DRAFT_ONLY",out,len(vals))

if __name__=="__main__":
    main()
