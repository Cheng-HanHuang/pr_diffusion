"""Pure A23 XP-C native orbit score for the remaining GT-only tests.

Imports frozen R0/A20 DDPM noise semantics; never tunes a score/threshold.
GPU workers call this with pinned native tweedie(z,sigma). Unit tests can
supply a CPU mock model, without importing CUDA or accessing GT files.
"""
from __future__ import annotations
import math
import torch
from prdiffusion.track_r.physics import ORBIT_PATTERNS,apply_orbit
from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed

METHODS=("S_mid_all8","2_forward")
PATTERNS={"S_mid_all8":ORBIT_PATTERNS,"2_forward":("000","111")}
TIMES={"S_mid_all8":(300,400,500,600),"2_forward":(300,)}
DRAWS={"S_mid_all8":(0,1,2,3),"2_forward":(0,)}
EVALS={"S_mid_all8":128,"2_forward":2}
TIE=1e-7

def native_scores(x,uid,model,method,alphas=None,batch_size=None):
    if method not in METHODS:
        raise ValueError("method not in A23 prereg")
    if (x.dtype!=torch.float32 or x.ndim!=4 or tuple(x.shape)!=(1,3,256,256)
        or not bool(torch.isfinite(x).all()) or not uid):
        raise ValueError("invalid native S_mid input")
    alpha=frozen_alpha_schedule() if alphas is None else alphas
    pts=PATTERNS[method]
    if batch_size is None: batch_size=len(pts)
    if batch_size not in (1,2,8) or batch_size>len(pts):
        raise ValueError('invalid single-t prior batch size')
    sums={p:0.0 for p in pts}; count=0; calls=0
    with torch.inference_mode():
        for t in TIMES[method]:
            at=float(alpha[t])
            if not 0<at<1:raise ValueError("bad DDPM alpha")
            sigma=math.sqrt((1.-at)/at)
            for j in DRAWS[method]:
                gen=torch.Generator(device="cpu").manual_seed(prior_seed(uid,t,j))
                noise=torch.randn((1,3,256,256),generator=gen,
                                  dtype=torch.float32).to(x.device)
                # Paired noise: R_p epsilon for R_p x; one t in each call.
                z=torch.cat([apply_orbit(x,p)+sigma*apply_orbit(noise,p)
                             for p in pts],dim=0)
                targets=torch.cat([apply_orbit(noise,p) for p in pts])
                values=[]
                # A21 equivalence canary compares the same t, j and paired
                # noise with small B1/B2 versus B8 native forward grouping.
                for k in range(0,len(pts),batch_size):
                    part=z[k:k+batch_size]
                    target=targets[k:k+batch_size]
                    estimate=model.tweedie(part,sigma)
                    if not isinstance(estimate,torch.Tensor) or estimate.shape!=part.shape:
                        raise RuntimeError("native tweedie shape mismatch")
                    values.extend((((part-estimate)/sigma-target).square()
                         .flatten(1).mean(1).float().cpu().tolist()))
                    calls+=1
                if len(values)!=len(pts) or not all(math.isfinite(v) for v in values):
                    raise RuntimeError("nonfinite native orbit score")
                for p,v in zip(pts,values):sums[p]+=float(v)
                count+=1
    scores={p:sums[p]/count for p in pts}
    best=min(pts,key=lambda p:(scores[p],p))
    if scores["000"]-scores[best]<=TIE:
        best="000"
    return dict(method=method,uid=uid,scores=scores,
        best_orbit=best,upright_kept=best=="000",
        S000=scores["000"],margin_to_best=scores["000"]-min(scores.values()),
        denoiser_examples=EVALS[method],native_calls=calls,
        fixed_t_draw_grid_points=count,batch_size=batch_size,
        single_t_per_call=True,paired_R_noise=True,GT_only=True)
