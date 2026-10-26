#!/usr/bin/env python3
"""Seeded CPU-only TH1 prediction fitter; no image reads or XP access.

Input: a normalized A8 counts CSV with columns family,cohort,image_id,
good,orbit_global,orbit_partial,consistent,inconsistent. One row per image.
DAPS B22 must have exactly 2 calibration runs; all other rows 4.
Acceptable cohorts: S5_180 (FAST), B22, DEV80 (SITCOM/DAPS).
Run --help for arguments. Check actual A8 schema and normalize it
explicitly before using this script; NEVER infer missing class labels.
Fits a Dirichlet-multinomial empirical Bayes population per cohort and
derives the posterior predictive K curves using exact beta moments.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

SEED=20261026
CLASSES=("good","orbit_global","orbit_partial","consistent","inconsistent")
KGRID={"FAST":(8,12,16,24,32,48,64),"SITCOM":(8,12,16,24,32),
       "DAPS":(4,6,8)}
POLICIES=("E0","E1_CLIP","O_raw","O_orbit")
FAMS={"FAST":"FAST","FAST-DIPS":"FAST","SITCOM":"SITCOM","DAPS":"DAPS"}
COHORTS={"S5":"S5_180","S5_180":"S5_180","S1_b22":"B22","B22":"B22",
         "S2_b24_dev80":"DEV80","DEV80":"DEV80"}


def load(path):
    groups=defaultdict(list); identities=set()
    with open(path,newline="",encoding="utf8") as f:
        for no,row in enumerate(csv.DictReader(f),2):
            fam=FAMS.get(row.get("family","").upper())
            cohort=COHORTS.get(row.get("cohort",row.get("source","")))
            uid=row.get("image_id",row.get("uid","")).strip()
            if fam is None or cohort is None or not uid:
                raise ValueError("row %s: unknown family, cohort, or image id"%no)
            if (fam=="FAST") != (cohort=="S5_180"):
                raise ValueError("row %s: inconsistent FAST/S5 cohort"%no)
            key=(fam,cohort,uid)
            if key in identities:
                raise ValueError("row %s: duplicate image"%no)
            identities.add(key)
            try:
                C=np.array([int(row[c]) for c in CLASSES],dtype=int)
            except (KeyError,ValueError) as e:
                raise ValueError("row %s: normalize all five count columns first"%no) from e
            n=2 if key[:2]==("DAPS","B22") else 4
            if np.any(C<0) or C.sum()!=n:
                raise ValueError("row %s: unexpected calibration run count"%no)
            groups[(fam,cohort)].append(C)
    required={("FAST","S5_180"),("SITCOM","B22"),("SITCOM","DEV80"),
              ("DAPS","B22"),("DAPS","DEV80")}
    if set(groups)!=required: raise ValueError("Expected groups %s; got %s"%(required,set(groups)))
    if len(groups["FAST","S5_180"])!=180 or any(len(groups[f,"B22"])!=100 or
       len(groups[f,"DEV80"])!=80 for f in ("SITCOM","DAPS")):
        raise ValueError("Unexpected cohort sizes; no incomplete predictions")
    return {key:np.stack(c) for key,c in groups.items()}


def loss(logalpha,C):
    a=np.exp(logalpha); total=a.sum(); n=C.sum(axis=1)
    return float(-np.sum(gammaln(total)-gammaln(total+n)+
                         np.sum(gammaln(C+a)-gammaln(a),axis=1)))


def fit(C):
    p=(C.sum(axis=0)+.25)/(C.sum()+1.25)
    sols=[]
    for k in (.5,3.,20.):
        r=minimize(loss,np.log(k*p),args=(C,),method="L-BFGS-B",
                   bounds=[(-11.,8.)]*5,options={"maxiter":600})
        if np.isfinite(r.fun): sols.append(r)
    if not sols: raise RuntimeError("fit failed")
    return np.exp(min(sols,key=lambda r:r.fun).x)


def beta_zero(a,t,K):
    # E[(1-theta)^K] for theta~Beta(a,t-a).
    v=np.ones_like(a,dtype=float)
    for j in range(K): v *= (t-a+j)/(t+j)
    return v


def curves_conditional(C,a,K):
    B=C+a; t=B.sum(axis=1); p,og,op,c,u=B.T
    surv=1-beta_zero(u,t,K)
    return np.stack((surv*p/(t-u),surv*(p+og)/(t-u),
                     1-beta_zero(p,t,K),1-beta_zero(p+og+op,t,K)),axis=1)


def curves_given_theta(theta,K):
    p,og,op,c,u=theta.T; s=1-u
    factor=np.where(s>0,(1-u**K)/np.maximum(s,1e-300),float(K))
    return np.stack((factor*p,factor*(p+og),1-(1-p)**K,
                     1-(1-p-og-op)**K),axis=1)


def cross_given_theta(theta):
    p,og,op,c,u=theta.T; s=1-u
    for K in range(1,65):
        gen=np.mean(u**K)
        miss=np.mean((1-u**K)*np.divide(op+c,s,out=np.zeros_like(s),where=s>0))
        if miss>=gen:return K
    return None


def predict(C,rng,reps,Kgrid):
    a=fit(C)
    points={K:curves_conditional(C,a,K).mean(axis=0) for K in Kgrid}
    boot={K:[] for K in Kgrid}; crosses=[]; n=len(C)
    for _ in range(reps):
        ix=rng.integers(0,n,n)
        B=C[ix]; ab=fit(B)
        latent=rng.gamma(B+ab)
        latent/=latent.sum(axis=1,keepdims=True)
        for K in Kgrid: boot[K].append(curves_given_theta(latent,K).mean(axis=0))
        crosses.append(cross_given_theta(latent))
    bands={K:np.quantile(np.asarray(boot[K]),(.025,.975),axis=0) for K in Kgrid}
    crosses_finite=[v for v in crosses if v is not None]
    Kstar=cross_given_theta((C+a)/(C+a).sum(axis=1,keepdims=True))
    # Null upper if bootstrap includes no crossing by K=64.
    crosslo=(float(np.quantile(crosses_finite,.025)) if crosses_finite else None)
    crosshi=(float(np.quantile(crosses_finite,.975)) if len(crosses_finite)==len(crosses)
             else None)
    return a,points,bands,(Kstar,crosslo,crosshi)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--counts",required=True)
    ap.add_argument("--fit-commit",required=True)
    ap.add_argument("--code-commit",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--bootstrap",type=int,default=200)
    args=ap.parse_args()
    if args.bootstrap<40:raise ValueError("bootstrap must be >=40")
    groups=load(args.counts)
    rng=np.random.default_rng(SEED)
    preds=[];shape=[];cross=[];fits=[]
    for fam,Kgrid in KGRID.items():
        cohorts=("S5_180",) if fam=="FAST" else ("B22","DEV80","POOLED")
        for cohort in cohorts:
            C=(np.concatenate([groups[fam,"B22"],groups[fam,"DEV80"]])
               if cohort=="POOLED" else groups[fam,cohort])
            a,points,bands,(ks,kl,kh)=predict(C,rng,args.bootstrap,Kgrid)
            fits.append({"family":fam,"cohort":cohort,"images":len(C),
                         "alpha":a.tolist(),"class_totals":C.sum(axis=0).tolist()})
            for K in Kgrid:
                lo,hi=bands[K]
                for j,policy in enumerate(POLICIES):
                    preds.append({"family":fam,"dataset":"FFHQ","cohort":cohort,
                                  "policy":policy,"metric":"good25","K":K,
                                  "point":float(points[K][j]),
                                  "lo95":float(lo[j]),"hi95":float(hi[j])})
            for name in ("O_orbit - E1_CLIP non-decreasing in K",
                         "O_raw - E0 non-decreasing in K"):
                shape.append({"family":fam,"cohort":cohort,"statement":name})
            cross.append({"family":fam,"cohort":cohort,"K_star":ks,
                          "lo95":kl,"hi95":kh})
    obj={"version":"v1",
         "model":"Five-type image-heterogeneous Dirichlet-multinomial empirical Bayes; ideal global-only orientation repair (tau_g=1,tau_p=h=0)",
         "fit_data":["A8 strings K<=4"],"fit_commit":args.fit_commit,
         "code_commit":args.code_commit,
         "uncertainty":"Seeded image-cluster bootstrap with refitting empirical Bayes concentrations and drawing latent per-image probabilities; extrapolation sensitivity to structural zeros and S1-S3 is not covered by bands.",
         "predictions":preds,"shape_predictions":shape,"crossover_K":cross,
         "fit_diagnostics":fits,"seed":SEED}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print("TH1_FIT_WRITE",out,"predictions",len(preds))


if __name__=="__main__":
    main()
