#!/usr/bin/env python3
"""B26.2 finite inner-likelihood estimator experiment (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

MASTER=26091501
ALPHAS=[1.0,0.72,0.36,0.12,0.0]
SIGMA=0.08
K=5
NTRAJ=512
BATCH=128
METHODS=("hard_exact_intermediate","categorical_exact_intermediate","categorical_denoised_point","categorical_lhat_1","categorical_lhat_4","categorical_lhat_16")


def seed(name:str)->int:
    return int.from_bytes(hashlib.sha256(f"B26|{name}|{MASTER}".encode()).digest()[:8],"big")

def rng(name:str): return np.random.Generator(np.random.PCG64(seed(name)))

def writej(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)

def logsumexp(x,axis=-1):
    m=np.max(x,axis=axis,keepdims=True)
    out=m+np.log(np.sum(np.exp(x-m),axis=axis,keepdims=True))
    return np.squeeze(out,axis=axis)

def softmax(logw):
    return np.exp(logw-logsumexp(logw,axis=-1)[...,None])

def categorical_rows(r,w,k=1):
    u=r.random((w.shape[0],k)); c=np.cumsum(w,axis=1)
    return (u[:,:,None]>c[:,None,:]).sum(axis=2)

def templates_for(name):
    a=np.zeros((6,6)); a[1,1],a[2,4],a[4,2],a[3,3]=1,.72,.43,.21
    ra=np.flip(a,(0,1)).copy()
    b=np.zeros((6,6)); b[1,1],b[1,4],b[4,1],b[3,4]=1,.65,.35,.22
    c=np.zeros((6,6)); c[1,2],c[2,2],c[3,2],c[4,4]=.9,.55,.35,.8
    if name=="distinguishable_equal": return np.stack([a,b,c]),np.array([1/3]*3)
    if name=="ambiguity_unequal": return np.stack([a,ra,b]),np.array([.6,.2,.2])
    near=ra.copy(); near[1,4]+=.055; near[4,1]+=.025
    if name=="near_ambiguity_equal": return np.stack([a,near]),np.array([.5,.5])
    if name=="near_ambiguity_unequal": return np.stack([a,near]),np.array([.7,.3])
    raise ValueError(name)

def forward(x):
    if x.ndim==3:return np.abs(np.fft.fft2(x,norm="ortho")).reshape(x.shape[0],-1)
    return np.abs(np.fft.fft2(x,norm="ortho")).reshape(-1)

def template_loglik(y,meas): return -0.5*np.sum((meas-y[None,:])**2,axis=1)/(SIGMA*SIGMA)

def weights_z(z,flat,pi,alpha):
    n=z.shape[0]
    if alpha==0:return np.broadcast_to(pi,(n,len(pi))).copy()
    if alpha==1:
        dist=np.sum((z[:,None,:]-flat[None,:,:])**2,axis=2); idx=np.argmin(dist,axis=1)
        w=np.zeros((n,len(pi))); w[np.arange(n),idx]=1.; return w
    var=1-alpha; means=math.sqrt(alpha)*flat
    logw=np.log(pi)[None,:]-.5*np.sum((z[:,None,:]-means[None,:,:])**2,axis=2)/var
    return softmax(logw)

def exact_log_intermediate(z,flat,pi,alpha,loglik):
    w=weights_z(z,flat,pi,alpha)
    lw=np.where(w>0,np.log(np.maximum(w,1e-300)),-np.inf)+loglik[None,:]
    return logsumexp(lw,axis=1),w

def denoised_loglik(z,templates,pi,alpha,y):
    flat=templates.reshape(len(templates),-1); w=weights_z(z,flat,pi,alpha)
    xhat=(w@flat).reshape((-1,6,6)); pred=forward(xhat)
    return -.5*np.sum((pred-y[None,:])**2,axis=1)/(SIGMA*SIGMA)

def bridge_candidates(z,flat,pi,at,as_,rprop):
    n=z.shape[0]; wt=weights_z(z,flat,pi,at)
    tidx=categorical_rows(rprop,wt,K)
    cand=np.empty((n,K,flat.shape[1]))
    if as_==1:
        for j in range(K): cand[:,j]=flat[tidx[:,j]]
        return cand
    ratio=math.sqrt(at/as_) if at>0 else 0.; coef=ratio*(1-as_)/(1-at)
    var=(1-as_)-ratio*ratio*(1-as_)**2/(1-at)
    if var < -1e-12: raise RuntimeError(f"negative bridge variance {var}")
    noise=rprop.normal(size=cand.shape)
    for j in range(K):
        xt=flat[tidx[:,j]]
        mean=math.sqrt(as_)*xt+coef*(z-math.sqrt(at)*xt)
        cand[:,j]=mean+math.sqrt(max(var,0.))*noise[:,j]
    return cand

def uniform_hard_choice(logv,rtie,tol=1e-12):
    out=np.empty(logv.shape[0],dtype=int)
    mx=np.max(logv,axis=1)
    for i in range(logv.shape[0]):
        ids=np.flatnonzero(np.abs(logv[i]-mx[i])<=tol)
        out[i]=ids[int(rtie.integers(len(ids)))]
    return out

def select_categorical(logv,rsel):
    return categorical_rows(rsel,softmax(logv),1)[:,0]

def ess_and_max(logv):
    w=softmax(logv); return 1./np.sum(w*w,axis=1),np.max(w,axis=1)

def lhat_log(z,flat,pi,alpha,loglik,M,rinner):
    w=weights_z(z,flat,pi,alpha); idx=categorical_rows(rinner,w,M)
    vals=loglik[idx]
    return logsumexp(vals,axis=1)-math.log(M)

def mse_template(a,b): return float(np.mean((a-b)**2))

def simulate(family,obs_idx,templates,pi,y,loglik,posterior,truth_idx,method):
    flat=templates.reshape(len(templates),-1); d=flat.shape[1]
    # Reset these streams identically for every method: common-random-number coupling.
    rprop=rng(f"OUTER|{family}|obs={obs_idx}")
    rsel=rng(f"SELECT|{family}|obs={obs_idx}")
    rtie=rng(f"TIE|{family}|obs={obs_idx}")
    # Same inner base stream for M=1,4,16; each method restarts it, making draws nested.
    rinner=rng(f"INNER|{family}|obs={obs_idx}")
    counts=np.zeros(len(pi),dtype=np.int64); mse_sum=0.; done=0
    outer_like=inner_like=0; ess_sum=maxw_sum=weight_cells=0.; lerr=[]
    start=time.perf_counter()
    while done<NTRAJ:
        n=min(BATCH,NTRAJ-done); done+=n
        z=rprop.normal(size=(n,d))
        for t in range(len(ALPHAS)-1,0,-1):
            s=t-1; at=float(ALPHAS[t]); as_=float(ALPHAS[s])
            cand=bridge_candidates(z,flat,pi,at,as_,rprop)
            fc=cand.reshape(n*K,d)
            lexact,_=exact_log_intermediate(fc,flat,pi,as_,loglik); lexact=lexact.reshape(n,K)
            outer_like += n*K
            if method=="hard_exact_intermediate":
                lv=lexact; choose=uniform_hard_choice(lv,rtie)
            elif method=="categorical_exact_intermediate":
                lv=lexact; choose=select_categorical(lv,rsel)
            elif method=="categorical_denoised_point":
                lv=denoised_loglik(fc,templates,pi,as_,y).reshape(n,K); outer_like += n*K
                choose=select_categorical(lv,rsel)
            elif method.startswith("categorical_lhat_"):
                M=int(method.rsplit("_",1)[1])
                lv=lhat_log(fc,flat,pi,as_,loglik,M,rinner).reshape(n,K)
                inner_like += n*K*M
                choose=select_categorical(lv,rsel)
                lerr.extend((lv-lexact).reshape(-1).tolist())
            else: raise ValueError(method)
            e,mx=ess_and_max(lv); ess_sum+=float(e.sum()); maxw_sum+=float(mx.sum()); weight_cells+=n
            z=cand[np.arange(n),choose]
        dist=np.sum((z[:,None,:]-flat[None,:,:])**2,axis=2); idx=np.argmin(dist,axis=1)
        counts+=np.bincount(idx,minlength=len(pi))
        for ii in idx: mse_sum += mse_template(templates[int(ii)],templates[truth_idx])
    probs=counts/NTRAJ
    omissions=[int(i) for i,p in enumerate(posterior) if p>=.05 and probs[i]<.005]
    exact_sampling_mse=sum(float(posterior[i])*mse_template(templates[i],templates[truth_idx]) for i in range(len(pi)))
    postmean=np.tensordot(posterior,templates,axes=(0,0))
    posterior_mean_truth_mse=mse_template(postmean,templates[truth_idx])
    bayes_risk=sum(float(posterior[i])*mse_template(templates[i],postmean) for i in range(len(pi)))
    return {
        "family":family,"observation_index":obs_idx,"method":method,"n":NTRAJ,"terminal_probabilities":probs.tolist(),
        "reference_posterior":posterior.tolist(),"tv_error":.5*float(np.abs(probs-posterior).sum()),"mode_omissions":omissions,
        "reconstruction_mse_to_sampled_truth":mse_sum/NTRAJ,"reference_posterior_sampling_mse_to_truth":exact_sampling_mse,
        "posterior_mean_decision_mse_to_truth":posterior_mean_truth_mse,"posterior_bayes_risk_squared_error":bayes_risk,
        "mean_normalized_weight_ess":ess_sum/weight_cells,"mean_max_normalized_weight":maxw_sum/weight_cells,
        "outer_intermediate_likelihood_evaluations":outer_like,"inner_template_likelihood_evaluations":inner_like,
        "finite_inner_loglik_error_mean":float(np.mean(lerr)) if lerr else None,
        "finite_inner_loglik_error_rmse":float(np.sqrt(np.mean(np.square(lerr)))) if lerr else None,
        "cpu_wall_seconds":time.perf_counter()-start
    }

def summarize(rows):
    out=[]
    for fam in sorted({r["family"] for r in rows}):
        for method in METHODS:
            g=[r for r in rows if r["family"]==fam and r["method"]==method]
            if len(g)!=32: raise RuntimeError(f"aggregate count {fam}/{method}={len(g)}")
            out.append({
                "family":fam,"method":method,"observations":len(g),
                "tv_mean":float(np.mean([r["tv_error"] for r in g])),"tv_median":float(np.median([r["tv_error"] for r in g])),
                "truth_mse_mean":float(np.mean([r["reconstruction_mse_to_sampled_truth"] for r in g])),
                "reference_sampling_mse_mean":float(np.mean([r["reference_posterior_sampling_mse_to_truth"] for r in g])),
                "posterior_mean_truth_mse_mean":float(np.mean([r["posterior_mean_decision_mse_to_truth"] for r in g])),
                "bayes_risk_mean":float(np.mean([r["posterior_bayes_risk_squared_error"] for r in g])),
                "mean_ess":float(np.mean([r["mean_normalized_weight_ess"] for r in g])),
                "mean_max_weight":float(np.mean([r["mean_max_normalized_weight"] for r in g])),
                "mode_omission_observations":sum(bool(r["mode_omissions"]) for r in g),
                "outer_likelihood_evaluations":sum(r["outer_intermediate_likelihood_evaluations"] for r in g),
                "inner_likelihood_evaluations":sum(r["inner_template_likelihood_evaluations"] for r in g),
                "cpu_wall_seconds":sum(r["cpu_wall_seconds"] for r in g),
            })
    # Error-decomposition deltas relative to categorical exact outer finite-proposal comparator.
    dec=[]
    for fam in sorted({r["family"] for r in rows}):
        exact={r["observation_index"]:r for r in rows if r["family"]==fam and r["method"]=="categorical_exact_intermediate"}
        for method in ("categorical_denoised_point","categorical_lhat_1","categorical_lhat_4","categorical_lhat_16"):
            g=[r for r in rows if r["family"]==fam and r["method"]==method]
            dec.append({"family":fam,"method":method,
                "mean_tv_increment_over_exact_outer":float(np.mean([r["tv_error"]-exact[r["observation_index"]]["tv_error"] for r in g])),
                "mean_truth_mse_increment_over_exact_outer":float(np.mean([r["reconstruction_mse_to_sampled_truth"]-exact[r["observation_index"]]["reconstruction_mse_to_sampled_truth"] for r in g]))})
    return out,dec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("B26.2 requires CUDA_VISIBLE_DEVICES='' exactly")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    families=("distinguishable_equal","ambiguity_unequal","near_ambiguity_equal","near_ambiguity_unequal")
    observations=[]; rows=[]; global_start=time.perf_counter()
    for fam in families:
        templates,pi=templates_for(fam); meas=forward(templates)
        if fam=="ambiguity_unequal":
            rel=np.linalg.norm(meas[0]-meas[1])/max(np.linalg.norm(meas[0]),1e-300)
            if rel>1e-12: raise RuntimeError(f"exact symmetry numerical check failed {rel}")
        for oi in range(32):
            rtruth=rng(f"TRUTH|{fam}|obs={oi}"); rnoise=rng(f"MEAS_NOISE|{fam}|obs={oi}")
            truth=int(categorical_rows(rtruth,pi[None,:],1)[0,0])
            y=meas[truth]+SIGMA*rnoise.normal(size=meas.shape[1])
            ll=template_loglik(y,meas)
            forced=False; pre_diff=None
            if fam=="ambiguity_unequal":
                pre_diff=float(abs(ll[0]-ll[1])); shared=float(0.5*(ll[0]+ll[1])); ll[0]=shared; ll[1]=shared; forced=True
            post=softmax(np.log(pi)+ll)
            observations.append({"family":fam,"observation_index":oi,"truth_index":truth,"measurement":y.tolist(),"template_loglik":ll.tolist(),"reference_posterior":post.tolist(),"exact_pair_forced_equal":forced,"pre_force_loglik_abs_diff":pre_diff})
            for method in METHODS:
                rows.append(simulate(fam,oi,templates,pi,y,ll,post,truth,method))
    agg,dec=summarize(rows)
    writej(args.output/"OBSERVATIONS.json",{"schema_version":"b26.cpu-observations.v1","count":len(observations),"rows":observations})
    writej(args.output/"METHOD_ROWS.json",{"schema_version":"b26.cpu-method-rows.v1","rows":rows})
    writej(args.output/"CPU_SUMMARY.json",{"schema_version":"b26.cpu-summary.v1","status":"PASS","families":4,"observations":128,"methods":list(METHODS),"trajectories_per_method_observation":NTRAJ,"total_complete_reverse_trajectories":len(rows)*NTRAJ,"aggregates":agg,"error_decomposition":dec,"wall_seconds":time.perf_counter()-global_start,"gpu_work_performed":False,"pretrained_model_inference_performed":False,"synthetic_measurements_generated":128,"confirmation_payloads_accessed":False})
    print(json.dumps({"status":"PASS","rows":len(rows),"trajectories":len(rows)*NTRAJ,"wall_seconds":time.perf_counter()-global_start},sort_keys=True))

if __name__=="__main__": main()
