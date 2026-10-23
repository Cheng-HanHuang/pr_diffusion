"""Frozen Track R R0.7 scientific statistics (CPU only, no file access).

Primary definitions mirror configs/track_r/r0_frozen_v1.json; the
caller supplies evaluated per-candidate rows. Never access GT in
GPU-side code.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import math

import numpy as np

ORBIT=tuple(f"{i:03b}" for i in range(8))
GOOD=25.
BAD=20.
CATASTROPHIC=15.
BOOT_N=10000

def choose(scores):
    if set(scores)!=set(ORBIT):
        raise ValueError("eight fixed orbit scores expected")
    best=min(ORBIT,key=lambda key:float(scores[key]))
    return "000" if float(scores["000"])-float(scores[best])<=1e-7 else best

def orbit_psnr_channel_mse(x8,g8):
    """Exactly reconstruct 8 PSNRs via per-channel raw/flipped MSEs."""
    import torch
    from prdiffusion.track_r.physics import psnr_rgb8
    if (x8.shape!=(1,3,256,256) or g8.shape!=x8.shape or
        x8.dtype!=torch.uint8 or g8.dtype!=torch.uint8):
        raise ValueError("canonical uint8 source and GT required")
    x=x8[0].float()
    gt=g8[0].float()
    raw=(x-gt).square().flatten(1).mean(1).numpy().astype(np.float64)
    rotated=(torch.flip(x,(-2,-1))-gt).square().flatten(1).mean(1).numpy().astype(np.float64)
    mse={}
    for key in ORBIT:
        channels=[rotated[k] if key[k]=="1" else raw[k] for k in range(3)]
        mse[key]=float(max(sum(channels)/(3*255**2),1e-12))
    psnr={k:float(-10*math.log10(mse[k])) for k in ORBIT}
    return psnr

def label_candidate(rec,tau):
    raw=float(rec["psnr_raw"])
    best=float(rec["psnr_orbit_best"])
    if raw>=GOOD:
        return "GOOD"
    if best>=GOOD:
        return "ORBIT_ERROR"
    if tau is None:
        return "NONORBIT_UNKNOWN_TAU"
    return ("NONORBIT_CONSISTENT" if float(rec["rho_signed_raw"])<=tau
            else "NONORBIT_INCONSISTENT")

def qlinear(values,quantile):
    if not values:return None
    return float(np.quantile(np.asarray(values,float),quantile,method="linear"))

def tau_from_good(rows):
    grouped=defaultdict(list)
    existing=set()
    for r in rows:
        k=(r["source"],r["family"])
        existing.add(k)
        if r["psnr_raw"]>=GOOD:
            grouped[k].append(float(r["rho_signed_raw"]))
    return {k:qlinear(grouped[k],0.95) for k in sorted(existing)}

def auc_roc(labels,scores):
    """Mann-Whitney AUROC, scores higher means more failure, tie-average."""
    if len(labels)!=len(scores):raise ValueError("different lengths")
    if any(x not in (False,True,0,1) for x in labels):
        raise ValueError("binary labels required")
    x=np.asarray(scores,dtype=float)
    y=np.asarray(labels,dtype=bool)
    if len(x)==0 or not bool(np.isfinite(x).all()):
        return None
    pos=int(y.sum());neg=len(y)-pos
    if pos==0 or neg==0:return None
    order=np.argsort(x,kind="stable")
    sorted_scores=x[order]
    _,start,counts=np.unique(sorted_scores,return_index=True,return_counts=True)
    ranks=np.empty(len(x),dtype=float)
    for beginning,n in zip(start,counts):
        ranks[order[beginning:beginning+n]]=beginning+(n+1)/2
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))

def seeded_rng(analysis_id):
    n=int.from_bytes(hashlib.sha256(
        ("R0BOOT|"+analysis_id).encode()).digest()[:8],"little")
    return np.random.Generator(np.random.PCG64(n))

def clustered_ratio_ci(rows,numerator,denominator,analysis_id,reps=BOOT_N):
    """Image-ID cluster bootstrap; duplicate IDs across sources stay paired."""
    byimg=defaultdict(lambda:[0.,0.])
    for r in rows:
        item=byimg[str(r["image_id"])]
        item[0]+=float(numerator(r))
        item[1]+=float(denominator(r))
    if not byimg:return {"estimate":None,"ci95":None,"n_images":0,"n_valid_boot":0}
    counts=np.asarray(list(byimg.values()),dtype=float)
    num=float(counts[:,0].sum());den=float(counts[:,1].sum())
    if den<=0:
        return {"estimate":None,"ci95":None,"n_images":len(counts),"n_valid_boot":0}
    rng=seeded_rng(analysis_id)
    samples=[]
    # Fixed 10,000 bootstrap replicates, preserving image as cluster.
    while len(samples)<reps:
        size=min(1000,reps-len(samples))
        ids=rng.integers(0,len(counts),size=(size,len(counts)))
        sums=counts[ids].sum(axis=1)
        vals=np.full(size,np.nan)
        np.divide(sums[:,0],sums[:,1],out=vals,where=sums[:,1]>0)
        samples.extend(vals.tolist())
    values=np.asarray(samples,float)
    valid=values[np.isfinite(values)]
    return {"estimate":num/den,
        "ci95":[float(v) for v in np.quantile(valid,[0.025,0.975])]
            if len(valid) else None,
        "n_images":len(counts),"n_valid_boot":len(valid)}

def clustered_paired_ci(a,b,analysis_id,reps=BOOT_N):
    """Metric deltas a_i-b_i, paired image-unit bootstrap."""
    if set(a)!=set(b):
        raise ValueError("image-paired policies need identical image keys")
    if not a:
        return {"estimate":None,"ci95":None,"n_images":0}
    ids=sorted(a)
    delta=np.asarray([float(a[k])-float(b[k]) for k in ids])
    estimate=float(delta.mean())
    rng=seeded_rng(analysis_id)
    draws=[]
    for start in range(0,reps,1000):
        size=min(1000,reps-start)
        index=rng.integers(0,len(ids),(size,len(ids)))
        draws.append(delta[index].mean(axis=1))
    values=np.concatenate(draws)
    return {"estimate":estimate,
            "ci95":[float(v) for v in np.quantile(values,[0.025,0.975])],
            "n_images":len(ids)}

def exact_mcnemar(a,b,threshold=GOOD):
    """Two-sided exact McNemar p-value from discordant Good25 outcomes."""
    if set(a)!=set(b):raise ValueError("unpaired images")
    n01=sum(float(a[k])<threshold<=float(b[k]) for k in a)
    n10=sum(float(b[k])<threshold<=float(a[k]) for k in a)
    discordant=n01+n10
    if not discordant:return {"n01":0,"n10":0,"p_exact":1.}
    # Exact two-sided binomial p with a 0.5 null without scipy dependency.
    k=min(n01,n10)
    p=min(1.,2*sum(math.comb(discordant,i) for i in range(k+1))/(2**discordant))
    return {"n01":n01,"n10":n10,"p_exact":float(p)}

def summarize_psnr(values):
    if not values:return None
    a=np.asarray(values,float)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),
        "good25":float(np.mean(a>=GOOD)),
        "bad20":float(np.mean(a<BAD)),
        "bad15":float(np.mean(a<CATASTROPHIC)),
        "q05":float(np.quantile(a,.05,method="linear")),
        "min":float(a.min())}

def select_e(group,k,tau):
    """The frozen E0/E1/E2/E3 and both GT ceilings for one image and K."""
    if tau is None:return None
    candidates=sorted(group,key=lambda r:(r["run_index"],r["uid"]))
    if len(candidates)<k:return None
    selected=candidates[:k]
    if len({r["run_index"] for r in selected})!=k:
        raise RuntimeError("STOP_R07_DUPLICATE_RUN_ORDER")
    e0=min(selected,key=lambda r:(r["rho_signed_raw"],r["run_index"]))
    e2_feasible=[r for r in selected if r["rho_signed_repaired"]<=tau]
    # Frozen empty-feasible fallback: smallest signed measurement residual.
    e2=min(e2_feasible,key=lambda r:(r["S_repaired"],r["run_index"])) if e2_feasible else min(
        selected,key=lambda r:(r["rho_signed_repaired"],r["run_index"]))
    e3=min(e2_feasible,key=lambda r:(r["p_fake_repaired"],r["run_index"])) if e2_feasible else min(
        selected,key=lambda r:(r["rho_signed_repaired"],r["run_index"]))
    result={
        "E0":float(e0["psnr_raw"]),
        "E1":float(e0["psnr_repaired"]),
        "E2":float(e2["psnr_repaired"]),
        "E3":float(e3["psnr_repaired"]),
        "O_raw":max(float(r["psnr_raw"]) for r in selected),
        "O_orbit":max(float(r["psnr_orbit_best"]) for r in selected)}
    return {"psnr":result,"uids":{
        "E0":e0["uid"],"E1":e0["uid"],
        "E2":e2["uid"],"E3":e3["uid"]},
        "empty_feasible":not bool(e2_feasible),
        "n_feasible":len(e2_feasible)}

def rankdata(values):
    x=np.asarray(values,float)
    idx=np.argsort(x,kind="stable")
    out=np.empty(len(x),float)
    sorted_x=x[idx]
    _,starts,counts=np.unique(sorted_x,return_index=True,return_counts=True)
    for i,n in zip(starts,counts):
        out[idx[i:i+n]]=i+(n+1)/2
    return out

def spearman(x,y):
    if len(x)!=len(y) or len(x)<3:return None
    u=rankdata(x);v=rankdata(y)
    if float(u.std())<1e-12 or float(v.std())<1e-12:return None
    return float(np.corrcoef(u,v)[0,1])

def high_frequency_energy(x8):
    """Canonical RGB8, 2-D cycles/pixel radius >0.25; includes DC."""
    from prdiffusion.track_r.physics import from_rgb8
    if x8.shape!=(1,3,256,256):
        raise ValueError("RGB8 shape mismatch")
    x=x8[0].numpy().astype(np.float32)/255.
    energy=np.abs(np.fft.fft2(x,axes=(-2,-1)))**2
    f=np.fft.fftfreq(256)
    radius=np.sqrt(f[:,None]**2+f[None,:]**2)
    denom=float(energy.sum())
    return float(energy[:,radius>0.25].sum()/denom) if denom>0 else 0.
