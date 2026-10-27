#!/usr/bin/env python3
"""Conditional-prefix Track R TH1 forecast simulator (K<=4 A8 inputs only).

Preserves the KNOWN ranks and classes of historical runs 0..n0-1,
draws only the unobserved future runs, and computes exact K-subset
U-statistics for each simulated complete nmax panel.

This is a guarded PRE-FREEZE research implementation. Source file must
pass audit_a8_rank_race.read_a8's authorized A8 SHA-256 guard.
No code path reads XP returns or runs >=4 (>=2 for DAPS/B22).

The preliminary fresh-K simulator fit_th1_rank_race_v1.py is NOT the
right primary estimand for the observed historical-plus-new-run protocol.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np

from audit_a8_rank_race import FAMILIES, read_a8, crosscheck, fit_weights
from fit_th1_dirichlet import fit as fit_alpha

SEED=20261027
N_BY_COHORT={"B22":100,"DEV80":80}
PANELS={"DAPS":{"primary":(8,(4,6,8))},
        "SITCOM":{"primary":(16,(8,12,16)),
                  "secondary":(32,(8,12,16,24,32))},
        "FAST":{"primary":(64,(8,12,16,24,32,48,64)),
                "secondary":(16,(8,12,16))}}
VARIANTS=("race","S2","tail","race_tau1")
POLICIES=("E0","E1_CLIP","O_raw","O_orbit")
LABELS="GOPCI"

def score_weights(fit,variant):
    if variant=="S2":return np.array([1.,1.,1.,1.,0.]),None
    if variant in ("race","race_tau1"):
        return np.array([1.,fit["w_O"],fit["w_O"],fit["w_C"],0.]),None
    if variant=="tail":
        baseline,extreme=1e-6,1000.
        target=5/205
        eps=(target-baseline/(1+baseline))/(
            extreme/(1+extreme)-baseline/(1+baseline))
        assert 0<eps<1
        # ordinary and extreme are subtypes of original C, expanded index 4.
        return np.array([1.,fit["w_O"],fit["w_O"],
                         baseline,extreme,0.]),eps
    raise ValueError(variant)

def weighted_permutation_prob(labels, rates):
    """Exact PL likelihood of the observed ascending rank order."""
    ans=1.
    for j in range(len(labels)):
        rest=sum(rates[j:])
        if rest<=0:
            # Only zero-rate candidates remain; their fixed index tie
            # permutation is uninformative for actual policy success.
            break
        if rates[j]<=0:return 0.
        ans*=rates[j]/rest
    return ans

def conditional_old_scores(ranked,variant,fit,rng):
    """Simulate old scores given their observed within-old-pool ordering."""
    weights,eps=score_weights(fit,variant)
    classes=[LABELS.index(c) for c in ranked]
    n=len(classes)
    if variant=="tail":
        choices=[i for i,c in enumerate(classes) if c==3]
        probs=[];types=[]
        for status in itertools.product([False,True],repeat=len(choices)):
            # Expanded classes: G=0, O=1, P=2, C_base=3,
            # C_extreme=4, I=5. Map directly from observed labels.
            status_by_index=dict(zip(choices,status))
            cl=[(5 if c=="I" else
                 (4 if status_by_index[j] else 3)
                 if c=="C" else LABELS.index(c))
                for j,c in enumerate(ranked)]
            rr=[float(weights[k]) for k in cl]
            if any(c=="I" and any(d!="I" for d in ranked[j+1:])
                   for j,c in enumerate(ranked)):
                rr=[.001 if c=="I" else rate
                    for c,rate in zip(ranked,rr)]
            prior=(eps**sum(status)) * ((1-eps)**(len(status)-sum(status)))
            probs.append(prior*weighted_permutation_prob(cl,rr))
            types.append((cl,rr))
        ps=np.asarray(probs,dtype=float)
        if ps.sum()<=0:raise ValueError("zero heavy-tail conditional likelihood")
        ps/=ps.sum()
        selected=int(rng.choice(len(types),p=ps))
        actual,rr=types[selected]
    else:
        actual=classes.copy()
        rr=[float(weights[k]) for k in actual]
        # Exactly one known FAST image has residual rank I before G.
        # Represent that pre-existing exception by a *finite* tiny
        # rate for ORIGINAL I entries, leaving future I at zero.
        if any(c=="I" and any(d!="I" for d in ranked[j+1:])
               for j,c in enumerate(ranked)):
            rr=[.001 if c=="I" else rate
                for c,rate in zip(ranked,rr)]
    scores=np.full(n,np.inf,dtype=float)
    t=0.
    for j in range(n):
        total=sum(rr[j:])
        if total<=0:break
        if rr[j]<=0:
            raise ValueError("observed rank impossible under reference race")
        t+=float(rng.exponential(1/total))
        scores[j]=t
    return actual,scores

def sample_panel(row,alpha,fit,variant,nmax,rng):
    """One complete hypothetical nmax-run panel conditioned on observed prefix."""
    C=np.asarray(row["counts"],dtype=float)
    theta=rng.dirichlet(C+alpha)
    n0=row["n"]
    weights,eps=score_weights(fit,variant)
    old_labels,old_scores=conditional_old_scores(row["ranked"],variant,fit,rng)
    future_base=rng.choice(5,size=nmax-n0,p=theta)
    if variant=="tail":
        # Source class index 4 is I, while expanded index 4 is C_extreme.
        future=[]
        for c in future_base:
            if c==4:future.append(5)
            elif c==3:future.append(4 if rng.random()<eps else 3)
            else:future.append(int(c))
        future=np.asarray(future,dtype=int)
    else:
        future=np.asarray(future_base,dtype=int)
    future_scores=np.full(len(future),np.inf,dtype=float)
    for j,cl in enumerate(future):
        if weights[cl]>0:future_scores[j]=rng.exponential(1/weights[cl])
    labels=np.concatenate([old_labels,future])
    scores=np.concatenate([old_scores,future_scores])
    assert len(labels)==nmax
    return np.asarray(labels,dtype=int),scores


def exact_prefix_oracles(row,alpha,Kgrid,nmax):
    """Exact conditional EXPECTED K-subset oracles, given observed A8 prefix.

    Integrate the probability that a uniform K-subset of the completed
    nmax panel contains no successful member: choose j old non-success
    runs and K-j future draws. The future draws are iid conditional on
    theta, and the Dirichlet marginal supplies beta non-success moments.
    """
    C=np.asarray(row["counts"],dtype=float)
    posterior=C+np.asarray(alpha,dtype=float)
    total=float(posterior.sum())
    n0=int(row["n"]);m=nmax-n0
    if m<0:raise ValueError("panel smaller than observed prefix")
    output=np.empty((len(Kgrid),2))
    for col,target in enumerate(((0,), (0,1,2))):
        g0=int(sum(C[j] for j in target))
        non=float(sum(posterior[j] for j in range(5) if j not in target))
        moments=[1.]
        for h in range(1,max(Kgrid)+1):
            moments.append(moments[-1]*(non+h-1)/(total+h-1))
        for ki,k in enumerate(Kgrid):
            zero=0.
            for j in range(max(0,k-m),min(k,n0-g0)+1):
                h=k-j
                zero+=math.comb(n0-g0,j)*math.comb(m,h)/math.comb(nmax,k)*moments[h]
            output[ki,col]=1-zero
    return output

def rank_weights(n,k):
    den=math.comb(n,k)
    return np.asarray([
       math.comb(n-r-1,k-1)/den if n-r>=k else 0.
       for r in range(n)],dtype=float)

def pass_at_k(n,g,k):
    return 1-(math.comb(n-g,k)/math.comb(n,k) if n-g>=k else 0.)

def exact_subset_values(labels,scores,Kgrid,cache):
    n=len(labels)
    ordering=np.argsort(scores,kind="stable")
    rank=labels[ordering]
    # C_extreme has expanded index4, I expanded5; both are failures.
    good=int(np.count_nonzero(labels==0))
    orbit=int(np.count_nonzero(labels<=2))
    ans=np.zeros((len(Kgrid),4))
    for ki,k in enumerate(Kgrid):
        w=cache[n,k]
        ans[ki,0]=np.dot(w,(rank==0).astype(float))
        ans[ki,1]=np.dot(w,(rank<=1).astype(float))
        ans[ki,2]=pass_at_k(n,good,k)
        ans[ki,3]=pass_at_k(n,orbit,k)
    return ans

def simulate(rows,alpha,fit,variant,Kgrid,nmax,rng,repetitions):
    if nmax<max(Kgrid):raise ValueError("panel smaller than requested K")
    cache={(nmax,k):rank_weights(nmax,k) for k in Kgrid}
    panels=np.zeros((repetitions,len(Kgrid),4))
    for b in range(repetitions):
        acc=np.zeros((len(Kgrid),4))
        for row in rows:
            labs,scores=sample_panel(row,alpha,fit,variant,nmax,rng)
            acc+=exact_subset_values(labs,scores,Kgrid,cache)
        panels[b]=acc/len(rows)
    return panels

def run(rows,rng,point_reps=250,bootstrap=60,boot_reps=3):
    output=[];diagnostics=[];anatomy=[];crossovers=[]
    for family in FAMILIES:
        fr=[x for x in rows if x["family"]==family]
        fitted=fit_weights(fr)
        cohort_data={}
        for cohort in ("B22","DEV80"):
            cr=[x for x in fr if x["cohort"]==cohort]
            C=np.array([r["counts"] for r in cr],dtype=float)
            alpha=fit_alpha(C)
            cohort_data[cohort]={"rows":cr,"C":C,"alpha":alpha}
            diagnostics.append({"family":family,"cohort":cohort,
                  "alpha":alpha.tolist(),"class_totals":C.sum(axis=0).tolist(),
                  "rank_weights":fitted})
        fitted_tau1=fit_weights(fr,tau=1.0)
        for panel_name,(nmax,Kgrid) in PANELS[family].items():
            sim={}
            # Closed-form conditional oracle means share one class model
            # across all selection rules; no Monte Carlo disagreement.
            exact_oracles={}
            for cohort in ("B22","DEV80"):
                c=cohort_data[cohort]
                exact_oracles[cohort]=np.mean([
                    exact_prefix_oracles(row,c["alpha"],Kgrid,nmax)
                    for row in c["rows"]],axis=0)
            for cohort in ("B22","DEV80"):
                c=cohort_data[cohort]
                for v in VARIANTS:
                    sim[cohort,v]=simulate(c["rows"],c["alpha"],
                                  (fitted_tau1 if v=="race_tau1" else fitted),
                                  v,Kgrid,nmax,rng,point_reps)
            # Intervals from each image's posterior, finite future-run panel,
            # and bootstrap-refitted hyperparameters. Bootstrap parameters
            # are fitted on resampled allowed image rows, but predictions
            # are always evaluated on the ORIGINAL image IDs.
            fitboot={}
            for cohort in ("B22","DEV80"):
                for v in VARIANTS:fitboot[cohort,v]=[]
            for _ in range(bootstrap):
                bdata={}
                allboots=[]
                for cohort in ("B22","DEV80"):
                    c=cohort_data[cohort]
                    ix=rng.integers(0,len(c["rows"]),len(c["rows"]))
                    sampled=[c["rows"][int(j)] for j in ix]
                    allboots.extend(sampled)
                    ab=fit_alpha(np.array([r["counts"] for r in sampled],float))
                    bdata[cohort]=ab
                wb=fit_weights(allboots)
                wb_tau1=fit_weights(allboots,tau=1.0)
                for cohort in ("B22","DEV80"):
                    c=cohort_data[cohort]
                    for v in VARIANTS:
                        preds=simulate(c["rows"],bdata[cohort],
                                       (wb_tau1 if v=="race_tau1" else wb),
                                       v,Kgrid,nmax,rng,boot_reps)
                        fitboot[cohort,v].append(preds.mean(axis=0))
            for cohort in ("B22","DEV80","POOLED"):
                for v in VARIANTS:
                    if cohort=="POOLED":
                        a=(100/180)*sim["B22",v]+(80/180)*sim["DEV80",v]
                        bb=(100/180)*np.asarray(fitboot["B22",v])+(80/180)*np.asarray(fitboot["DEV80",v])
                        size=180
                    else:
                        a=sim[cohort,v]
                        bb=np.asarray(fitboot[cohort,v])
                        size=N_BY_COHORT[cohort]
                    point=a.mean(axis=0)
                    simulation_mean=point.copy()
                    if cohort=="POOLED":
                        oracle=(100/180)*exact_oracles["B22"]+(80/180)*exact_oracles["DEV80"]
                    else:
                        oracle=exact_oracles[cohort]
                    point[:,2:]=oracle
                    # A6 planner control variate: shift selected-class predictions
                    # by their own panel's exact-vs-MC oracle discrepancy.
                    # This preserves 0 <= E0 <= O_raw and E1 <= O_orbit
                    # without mixing Monte Carlo selector with analytic oracle.
                    delta_raw=oracle[:,0]-simulation_mean[:,2]
                    delta_orb=oracle[:,1]-simulation_mean[:,3]
                    point[:,0]=simulation_mean[:,0]+delta_raw
                    point[:,1]=simulation_mean[:,1]+delta_orb
                    # Center the bootstrap to isolate its parameter
                    # fluctuations; panel draws already reflect the future
                    # finite-n U-statistic noise conditional on observed A8.
                    sel=rng.integers(0,len(bb),len(a))
                    draws=a+(bb[sel]-bb.mean(axis=0)[None,:,:])
                    draws[:,:,2:]+=(oracle-simulation_mean[:,2:])[None,:,:]
                    draws[:,:,0]+=delta_raw[None,:]
                    draws[:,:,1]+=delta_orb[None,:]
                    lo,hi=np.quantile(draws,(.025,.975),axis=0)
                    failure_draws=1-draws[:,:,3]
                    selection_draws=draws[:,:,3]-draws[:,:,1]
                    generation=1-point[:,3]
                    selection=point[:,3]-point[:,1]
                    crossing=next((int(k) for ki,k in enumerate(Kgrid)
                                   if selection[ki]>generation[ki]),None)
                    crossovers.append({"family":family,"panel":panel_name,
                        "panel_n_max":nmax,"cohort":cohort,"model_variant":v,
                        "K_star_on_grid":crossing,"K_grid":list(Kgrid),
                        "definition":"selection loss O_orbit-E1 strictly exceeds no-orbit-candidate generation failure"})
                    gflo,gfhi=np.quantile(failure_draws,(.025,.975),axis=0)
                    sllo,slhi=np.quantile(selection_draws,(.025,.975),axis=0)
                    for ai,k in enumerate(Kgrid):
                        anatomy.append({"family":family,"panel":panel_name,
                            "panel_n_max":nmax,"cohort":cohort,"model_variant":v,
                            "K":k,"generation_failure":float(generation[ai]),
                            "selection_loss":float(selection[ai]),
                            "generation_lo95":float(np.clip(gflo[ai],0,1)),
                            "generation_hi95":float(np.clip(gfhi[ai],0,1)),
                            "selection_lo95":float(np.clip(sllo[ai],0,1)),
                            "selection_hi95":float(np.clip(slhi[ai],0,1))})
                    for ki,K in enumerate(Kgrid):
                        if family=="DAPS" and K==4 and cohort!="B22":
                            continue
                        for pi,policy in enumerate(POLICIES):
                            val=float(point[ki,pi])
                            output.append({"family":family,"dataset":"FFHQ",
                                "cohort":cohort,"model_variant":v,
                                "policy":policy,"metric":"good25",
                                "K":K,"n_images":size,
                                "panel":panel_name,"panel_n_max":nmax,
                                "point":val,
                                "lo95":float(np.clip(min(lo[ki,pi],val),0,1)),
                                "hi95":float(np.clip(max(hi[ki,pi],val),0,1))})
    return output,diagnostics,anatomy,crossovers

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a8",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--point-reps",type=int,default=250)
    ap.add_argument("--bootstrap",type=int,default=60)
    ap.add_argument("--boot-reps",type=int,default=3)
    ap.add_argument("--freeze",action="store_true")
    args=ap.parse_args()
    if args.freeze:
        raise RuntimeError("Freeze still blocked: wait for A8b 2026-10-28 12 ET gate, full QA and registry handoff.")
    rows,sha=read_a8(args.a8)
    crosscheck(rows)
    rng=np.random.default_rng(SEED)
    predictions,fits,anatomy,crossovers=run(rows,rng,args.point_reps,args.bootstrap,args.boot_reps)
    out={"version":"DRAFT_PREFIX_CONDITIONAL_UNREGISTERED",
      "source_sha256":sha,"seed":SEED,
      "estimand":"K-subset U-statistics from a fixed nmax-run pool with observed"
                 " A8 run-label/rank prefix held fixed and only future runs drawn",
      "fit_data":["per-image A8 K<=4 labels and rank permutations"],
      "model":"Dirichlet image posteriors + observed-rank-conditioned PL"
              " exponential spacings + S2 reference + rare-C tail sensitivity",
      "uncertainty":"Conditional panel noise + cluster bootstrap of model hyperparameters;"
                    " model-conditional, not a partial identification interval",
      "predictions":predictions,"fit_diagnostics":fits,
      "anatomy_curves":anatomy,"crossover_K":crossovers}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    out["interval_type"]="predictive_realized_panel"
    out["variant_comparison_rule"]={
       "primary":"For each family primary panel E0/E1_CLIP, count K grid points within [lo95,hi95]; tie-break by mean standardized residual",
       "standardized_residual":"abs(observed-point)/max((hi95-lo95)/3.92,0.005)",
       "secondary_panels":"Evaluate separately and do not pool with primary",
       "sensitivity":"race_tau1 is sensitivity, not an alternative retrospective primary model",
       "tail":"pooled 5/205 C-over-G all arise SITCOM; DAPS/FAST arms stress-only"}
    out["panels"]=PANELS
    out["prior_sensitivity"]={"race_logweight_tau":2.5,
       "race_tau1_logweight_tau":1.0,
       "tail_C_over_G_composite_fraction":5/205,
       "tail_warning":"DAPS and FAST have zero C-over-G events in their A8 ranks"}
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("TH1_CONDITIONAL_PREFIX_DRAFT_PASS",len(predictions),path)

if __name__=="__main__":main()
