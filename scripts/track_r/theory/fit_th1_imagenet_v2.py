#!/usr/bin/env python3
"""Track R ImageNet FAST Good20 v2 conditional-prefix forecaster (A6.6).

Permitted input ONLY:
  docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv
  or its identical authorized uploaded copy.
Reads 100 images, four known native FAST run labels/ranks per image.
The target is the SAME images extended to a 16-run panel, K=8,12,16.

E1_CLIP is NOT predicted without ImageNet-specific G1 decisions.
No xp/returns or phase-2 outcomes are read. All production runs stay gated
from freeze until explicit SHA and UTC commit.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

from audit_a8_rank_race import fit_weights
from fit_th1_dirichlet import fit as fit_alpha
from fit_th1_prefix_v1 import (
    simulate,exact_prefix_oracles,SEED,VARIANTS,POLICIES)

IMAGENET_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
KGRID=(8,12,16)
NMAX=16
MODEL_VARIANTS=("race","S2","tail","race_tau1")
FORECAST_POLICIES=("E0","O_raw","O_orbit")
RUN_LABELS="GOPCI"

def load_imagenet(csvpath):
    payload=Path(csvpath).read_bytes()
    sha=hashlib.sha256(payload).hexdigest()
    if sha!=IMAGENET_SHA:
        raise ValueError("ImageNet v2 input SHA-256 mismatch")
    with open(csvpath,newline="",encoding="utf8") as handle:
        raw=list(csv.DictReader(handle))
    if len(raw)!=100:raise ValueError("expected 100 ImageNet phase-1 images")
    seen=set();rows=[]
    for d in raw:
        if d["source"]!="XP_D1_P1" or d["family"]!="FAST":
            raise ValueError("unexpected source/family")
        uid=d["image_id"]
        if uid in seen:raise ValueError("duplicate image")
        seen.add(uid)
        n=int(d["n_runs"])
        if n!=4:raise ValueError("outside authorized prefix")
        labels=d["run_labels"];ranked=d["e0_rank_labels"]
        order=[int(x)for x in d["e0_rank_order"].split(",")]
        rho=[float(x)for x in d["rho_signed_by_run"].split(";")]
        if (len(labels)!=n or len(ranked)!=n or
            sorted(order)!=list(range(n)) or
            "".join(labels[i]for i in order)!=ranked or
            len(rho)!=n or
            any(rho[order[i]]>rho[order[i+1]]+1e-11
                for i in range(n-1))):
            raise ValueError("rank and residual order mismatch "+uid)
        counts=np.array([int(d[z])for z in
            ("n_GOOD","n_ORBIT_global","n_ORBIT_partial",
             "n_CONSISTENT","n_INCONSISTENT")])
        if tuple(counts)!=tuple(labels.count(x)for x in RUN_LABELS):
            raise ValueError("count mismatch "+uid)
        rows.append({"image_id":uid,"family":"FAST","cohort":"IMAGENET",
                     "n":n,"counts":counts,"ranked":ranked,
                     "original":labels,"source":"XP_D1_P1"})
    total=np.sum([r["counts"]for r in rows],axis=0)
    if tuple(total)!=(100,67,3,188,42):
        raise ValueError("ImageNet Good20 expected class counts mismatch")
    return rows,sha

def run(rows,seed,point_reps,bootstrap,boot_reps):
    rng=np.random.default_rng(seed)
    original=np.stack([r["counts"]for r in rows])
    alpha=fit_alpha(original)
    weight=fit_weights(rows,tau=2.5)
    weight_tau1=fit_weights(rows,tau=1.)
    exact_oracle=np.mean([
        exact_prefix_oracles(r,alpha,KGRID,NMAX) for r in rows
    ],axis=0)
    output=[];anatomy=[]
    for variant in MODEL_VARIANTS:
        par=weight_tau1 if variant=="race_tau1" else weight
        sims=simulate(rows,alpha,par,variant,KGRID,NMAX,rng,point_reps)
        raw=sims.mean(axis=0);pts=raw.copy()
        pts[:,0]+=exact_oracle[:,0]-raw[:,2]
        pts[:,1]+=exact_oracle[:,1]-raw[:,3]
        pts[:,2:]=exact_oracle
        # Bootstrap A6 image-cluster parameters; predictions always
        # target the exact ORIGINAL 100 identified ImageNet images.
        fitboot=[]
        for _ in range(bootstrap):
            ix=rng.integers(0,len(rows),len(rows))
            sample=[rows[int(j)]for j in ix]
            alpha_b=fit_alpha(np.array([r["counts"]for r in sample]))
            w_b=fit_weights(sample,tau=1.0 if variant=="race_tau1" else 2.5)
            fitboot.append(simulate(rows,alpha_b,w_b,variant,KGRID,NMAX,
                                    rng,boot_reps).mean(axis=0))
        B=np.asarray(fitboot)
        ix=rng.integers(0,len(B),len(sims))
        # Control variate each realized panel around the exact
        # model-conditional oracle expectations.
        panel=sims+(B[ix]-B.mean(axis=0)[None,:,:])
        panel[:,:,0]+=(exact_oracle[:,0]-raw[:,2])[None,:]
        panel[:,:,1]+=(exact_oracle[:,1]-raw[:,3])[None,:]
        panel[:,:,2:]+=(exact_oracle-raw[:,2:])[None,:,:]
        lo,hi=np.quantile(panel,(.025,.975),axis=0)
        for ki,K in enumerate(KGRID):
            for policy,col in (("E0",0),("O_raw",2),("O_orbit",3)):
                val=float(pts[ki,col])
                output.append({"family":"FAST","dataset":"ImageNet",
                    "cohort":"D1_fixed100","model_variant":variant,
                    "policy":policy,"metric":"good20",
                    "panel":"primary","panel_n_max":NMAX,
                    "K":K,"n_images":100,
                    "point":val,
                    "lo95":float(np.clip(min(val,lo[ki,col]),0.,1.)),
                    "hi95":float(np.clip(max(val,hi[ki,col]),0.,1.))})
            generation=1-pts[ki,3]
            anatomy.append({"family":"FAST","panel":"primary",
                "model_variant":variant,"K":K,
                "generation_failure":float(generation),
                "E1_CLIP_selection_loss":None,
                "selection_loss_abstention":"ImageNet G1 decisions not provided"})
    # No E1CLIP forecast: ImageNet has 70 orbit runs and its
    # orientation repair is not guaranteed by the FFHQ observations.
    return output,anatomy,alpha,weight,weight_tau1

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--point-reps",type=int,default=350)
    p.add_argument("--bootstrap",type=int,default=60)
    p.add_argument("--boot-reps",type=int,default=3)
    p.add_argument("--freeze",action="store_true")
    args=p.parse_args()
    if args.freeze:
        raise RuntimeError("Freeze must be done only after production QA and"
          " provenance/commit SHA and UTC registration on theory branch")
    rows,sha=load_imagenet(args.input)
    predictions,anatomy,alpha,w,ws=run(
        rows,SEED,args.point_reps,args.bootstrap,args.boot_reps)
    obj={"version":"DRAFT_V2_UNREGISTERED",
         "dataset":"ImageNet","family":"FAST","metric":"good20",
         "n_images":100,"panel_n_max":16,"K_grid":list(KGRID),
         "source_sha256":sha,"seed":SEED,
         "interval_type":"predictive_realized_panel",
         "model":"Image-conditioned Dirichlet + observed-rank-conditional"
                 " exponential-score class race; fixed 4-run prefix;"
                 " model-conditional realized 16-run U-statistic panel",
         "rank_weights_primary":w,
         "rank_weights_sensitivity":ws,
         "alpha":alpha,"predictions":predictions,
         "anatomy_curves":anatomy,
         "E1_CLIP":{"status":"ABSTAIN","point":None,
                    "reason":"CLIP G1 decisions pending on ImageNet;"
                             " FFHQ CLIP accuracy not transferred"},
         "model_comparison_rule":{
           "primary":"E0 predictive interval coverage at K8/12/16;"
                     " tie-break by mean standardized residual",
           "standardized_residual":
             "abs(observed-point)/max((hi95-lo95)/3.92,0.005)",
           "sensitivity":"race_tau1 and pooled extreme-C tail stress;"
                         " do not retrofit primary race"},
         "provenance":{"training_input":"ImageNet D1 P1 indices 0-3 ONLY",
            "phase2_outcomes_seen":False,
            "ffhq_k_gt4_seen":False}}
    # Normalize NumPy arrays for portable JSON.
    obj["rank_weights_primary"]={
        k:float(w[k])for k in ("w_O","w_C")}
    obj["rank_weights_sensitivity"]={
        k:float(ws[k])for k in ("w_O","w_C")}
    obj["alpha"]=alpha.tolist()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("TH1_IMAGENET_V2_DRAFT_PASS",len(predictions),len(anatomy),out)
if __name__=="__main__":main()
