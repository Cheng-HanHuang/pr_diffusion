#!/usr/bin/env python3
"""A16 experimental NP kernel. Frozen-native baseline must replay before new arms.

Source-isolated: never changes historic B22 NP source. Measures exploratory
mechanisms ONLY on pre-exposed images. All physics calls use the actual saved
oversampled Fourier amplitude; correction does not evaluate the diffusion UNet.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

@dataclass(frozen=True)
class Arm:
    name: str
    family: str = "LF"
    steps: int = 1000
    proposal: str = "native"  # native, sparse, adaptive, beam2, budget2
    correction: str = "none"  # none, fast, gradient
    placement: str = "none"   # none, post, pre_score, both
    replace_gs: bool = False
    target: str = "clipped"   # clipped or signed; score treatment separately
    ranking: str = "native"  # native, corrected_lf, residual, moments
    stride: int = 40
    branch_hold: int = 6
    correction_start: float = 0.30
    correction_end: float = 0.96
    rho: float = 200.
    epsilon: float = .05
    correction_steps: int = 5

    def validate(self):
        assert self.family in ("LF", "S2")
        assert self.steps in (150, 250, 500, 1000)
        assert self.proposal in ("native","sparse","adaptive","beam2","budget2")
        assert self.correction in ("none","fast","gradient")
        assert self.placement in ("none","post","pre_score","both")
        assert self.target in ("clipped","signed")
        assert self.ranking in ("native","corrected_lf","residual","moments")
        assert 0 < self.epsilon and 0 < self.rho and 1 <= self.correction_steps <= 5
        assert 0 < self.stride and 1 <= self.branch_hold <= 12
        assert 0 <= self.correction_start < self.correction_end <= 1
        if self.placement != "none":
            assert self.correction != "none"
        else:
            assert self.correction == "none"
        if self.ranking == "corrected_lf":
            assert self.placement in ("pre_score", "both")
        if self.ranking == "residual":
            assert self.target in ("clipped","signed")


def k_for_step(arm: Arm, i: int, nominal_proj_start: int) -> int:
    """No stochastic/GT dependence; return true candidate UNet-evaluation count."""
    frac = i / max(1, arm.steps-1)
    if arm.proposal == "native":
        return 5 if i < nominal_proj_start else 1
    if arm.proposal == "budget2":
        return 2 if i < nominal_proj_start else 1
    if arm.proposal == "sparse":
        return 5 if i < nominal_proj_start and i % arm.stride == 0 else 1
    if arm.proposal == "adaptive":
        # Coarse-to-fine: sparse 3-child search in high/mid noise,
        # 2 children at intermediate steps, single continuation late.
        if frac < .12 or frac >= .75: return 1
        if i % arm.stride == 0:
            return 3 if frac < .48 else 2
        return 1
    if arm.proposal == "beam2":
        if .15 <= frac <= .70 and i % arm.stride == 0:
            return 2
        return 1
    raise AssertionError(arm.proposal)


def should_correct(arm: Arm, i: int) -> bool:
    if arm.correction == "none": return False
    frac = i/max(1,arm.steps-1)
    return arm.correction_start <= frac <= arm.correction_end and i % arm.stride == 0


def _flat_dot(a, b):
    return (a*b).flatten(1).sum(1).view(-1,1,1,1)


def _fwd_loss(A, x, anchor, target, rho, gamma):
    r = A(x) - target
    return (_flat_dot(x-anchor, x-anchor)/(2*gamma) +
            .5*rho*_flat_dot(r,r))


def fast_effective_correction(A, anchor, y, gamma: float,
                              rho: float=200., epsilon: float=.05,
                              steps: int=5, stats:dict|None=None):
    """One effective K2 FAST-DIPS update; mirrors pinned native x_update FD branch.

    K2's initial x-gradient is zero: x=p, v=A(p), u=0.
    First v/u update is performed analytically. Second x-update receives b2.
    """
    with torch.no_grad():
        p = anchor.detach().float()
        Ap = A(p)
        r = Ap-y
        norm = r.flatten(1).norm(dim=1,keepdim=True).view(-1,1,1,1).clamp_min(1e-12)
        alpha = (epsilon/norm).clamp_max(1.0)
        v1 = y + alpha*r
        b2 = 2*v1-Ap
    x = p
    gamma=max(float(gamma),1e-10)
    for _ in range(steps):
        with torch.enable_grad():
            x_req=x.detach().requires_grad_(True)
            ax=A(x_req)
            r2=ax-b2
            Jtr=torch.autograd.grad(ax,x_req,grad_outputs=r2)[0]
        g=torch.nan_to_num((x-p)/gamma+rho*Jtr)
        # Exact pinned FAST pixel default: FD approximation to Jg, eta=1e-3.
        with torch.no_grad():
            eta=1e-3
            dA=A(x+eta*g)-ax.detach()
            num=(eta**2/gamma)*_flat_dot(x-p,g)+eta*rho*_flat_dot(r2.detach(),dA)
            den=(eta**2/gamma)*_flat_dot(g,g)+rho*_flat_dot(dA,dA)+1e-12
            step=torch.nan_to_num(num/den,nan=0.,posinf=0.,neginf=0.).clamp(0,1)
            f0=_fwd_loss(A,x,p,b2,rho,gamma)
            nxt=x-step*g
            f1=_fwd_loss(A,nxt,p,b2,rho,gamma)
            for _ in range(8):
                mask=f1>f0
                if not bool(mask.any()): break
                step=torch.where(mask,step*.5,step)
                nxt=x-step*g
                f1=_fwd_loss(A,nxt,p,b2,rho,gamma)
            x=x-step*g
    if stats is not None: stats["fast_updates"] += 1
    return x.detach().to(anchor.dtype)


def gradient_correction(A, anchor, y, gamma:float, rho:float,
                        steps:int=5, stats:dict|None=None):
    """Fixed normalized-gradient/Armijo control, no ADMM target and no FD JVP.

    Not claimed cost-equivalent until FFT/VJP and wall counters are compared.
    """
    p=anchor.detach().float()
    x=p
    gamma=max(float(gamma),1e-10)
    for _ in range(steps):
        with torch.enable_grad():
            xr=x.detach().requires_grad_(True)
            ax=A(xr)
            r=ax-y
            gdata=torch.autograd.grad(ax,xr,grad_outputs=r)[0]
        g=torch.nan_to_num((x-p)/gamma+rho*gdata)
        with torch.no_grad():
            gn=g.square().mean().sqrt().clamp_min(1e-12)
            eta=min(.1,float(.001/gn))
            f0=_fwd_loss(A,x,p,y,rho,gamma)
            for _ in range(8):
                trial=x-eta*g
                if bool((_fwd_loss(A,trial,p,y,rho,gamma)<=f0).all()):break
                eta *= .5
            x=x-eta*g
    if stats is not None:stats["gradient_updates"]+=1
    return x.detach().to(anchor.dtype)


def noise_moment_score(A, x, target, sigma=.05):
    """Exploratory moment score; NOT full DC-loss from the literature."""
    r=(A(x)-target)/sigma
    return r.mean().square()+(r.float().std(unbiased=False)-1).square()


@torch.no_grad()
def reconstruct(*, selector, model, scheduler, measurement_signed, pad,
                seed:int, arm:Arm):
    """Rebuild frozen NP sampling loop with explicit additions + counted calls.

    Native-replay arm (1000 steps, LF/S2, 5 then 1, no correction) is
    required to match original reconstruct_with_selector_stat on the SAME seed.
    """
    arm.validate()
    from prdiffusion.seed import seed_everything
    seed_everything(seed)
    scheduler.set_timesteps(arm.steps,device=measurement_signed.device)
    ts=scheduler.timesteps
    target_clipped=measurement_signed.clamp_min(0.)
    target=target_clipped if arm.target=="clipped" else measurement_signed
    # Preserve exact frozen B22 low-frequency NP ranking/projection target by default.
    selector_target=target
    proj_start=max(1,round(.3*arm.steps))
    proj_spec="300:0.2" if arm.steps==1000 else f"{proj_start}:0.2"
    proj_schedule=selector.base.parse_radius_schedule(proj_spec,.2)
    dev=measurement_signed.device
    x_t=torch.randn((1,3,model.config.sample_size,model.config.sample_size),device=dev)
    states=[dict(x=None,eps=None)]
    counters=dict(unet=0,branches_spawned=0,branches_pruned=0,fast_updates=0,
                  gradient_updates=0,correction_operator_evaluations=0,
                  score_fft_estimate=0,gs_fft_estimate=0,branch_max=1,
                  correction_time_s=0., score_time_s=0., selected=0)
    audit_trace=[]
    open_branches_until=-1

    def A(v):
        counters["correction_operator_evaluations"]+=1
        return selector.base.oversampled_magnitude(v,pad)

    def correct(p,i,t_next):
        ab=scheduler.alphas_cumprod[t_next].to(device=dev,dtype=p.dtype)
        gamma=float((1-ab).clamp_min(1e-12)/ab.clamp_min(1e-12))
        before=time.perf_counter()
        if arm.correction=="fast":
            c=fast_effective_correction(A,p, target,gamma,arm.rho,arm.epsilon,
                                        arm.correction_steps,counters)
        elif arm.correction=="gradient":
            c=gradient_correction(A,p,target,gamma,arm.rho,arm.correction_steps,counters)
        else: raise RuntimeError("correction requested with none")
        counters["correction_time_s"]+=time.perf_counter()-before
        if not bool(torch.isfinite(c).all()):
            raise RuntimeError(f"NONFINITE correction at step {i}")
        return c

    for i in range(len(ts)-1):
        t=int(ts[i]);tn=int(ts[i+1])
        frac=i/max(1,arm.steps-1)
        is_beam=arm.proposal=="beam2"
        wants_branch=is_beam and len(states)==1 and .15<=frac<=.70 and i%arm.stride==0
        candidate_k=k_for_step(arm,i,proj_start)
        if len(states)>1: candidate_k=1
        if i==0:  # frozen native initial unconditional UNet
            init_eps=model(x_t,torch.tensor([t],device=dev,dtype=torch.long)).sample
            counters["unet"]+=1
            ab0=scheduler.alphas_cumprod[t].to(device=dev,dtype=x_t.dtype)
            initial=(x_t-(1-ab0).sqrt()*init_eps)/ab0.sqrt()
        all_c=[]
        for state_id,state in enumerate(states):
            p=initial if i==0 else state["x"]
            if p is None:raise RuntimeError("lost trajectory state")
            projection_applied=i>=proj_start and not arm.replace_gs
            if projection_applied:
                radius=selector.base.radius_at_step(proj_schedule,i)
                p=selector.base.enforce_oversampled_lowfreq(p,selector_target,pad,radius)
                counters["gs_fft_estimate"]+=2
            ab=scheduler.alphas_cumprod[tn].to(device=dev,dtype=p.dtype)
            sq=ab.sqrt();sq1=(1-ab).sqrt()
            xs=[];eps_list=[];lf=[];prev=[];adjusted=[]
            for j in range(candidate_k):
                if j==0 and state["eps"] is not None and candidate_k>1:
                    eps=state["eps"]
                else:eps=torch.randn_like(p)
                xt=sq*p+sq1*eps
                eps_pred=model(xt,torch.tensor([tn],device=dev,dtype=torch.long)).sample
                counters["unet"]+=1
                xhat=(xt-sq1*eps_pred)/sq
                xs.append(xhat);eps_list.append(eps.detach().clone())
                prev.append(((selector.base.to01(xhat)-selector.base.to01(p))**2).mean().clamp_min(1e-12).sqrt())
                is_corr=should_correct(arm,i)
                if is_corr and arm.placement in ("pre_score","both"):
                    adjusted.append(correct(xhat,i,tn))
                else:adjusted.append(None)
                xv=adjusted[-1] if (arm.ranking=="corrected_lf" and adjusted[-1] is not None) else xhat
                lf.append(selector.base.oversampled_lowfreq_mag_l2(xv,selector_target,pad,.6))
                counters["score_fft_estimate"]+=1
            if arm.family=="S2" and i<proj_start:
                lfv=torch.stack(lf)
                prv=torch.stack(prev)
                vals=lfv/lfv.median().abs().clamp_min(1e-8)+.01*prv/prv.median().abs().clamp_min(1e-8)
            else:vals=torch.stack(lf)
            for j in range(candidate_k):
                if arm.ranking=="residual" and should_correct(arm,i):
                    vv=adjusted[j] if adjusted[j] is not None else xs[j]
                    score=float((A(vv)-target).square().mean())
                elif arm.ranking=="moments" and should_correct(arm,i):
                    vv=adjusted[j] if adjusted[j] is not None else xs[j]
                    score=float(noise_moment_score(A,vv,target))
                else:score=float(vals[j])
                all_c.append(dict(score=score,x=xs[j],adjusted=adjusted[j],
                                  eps=eps_list[j],parent=state_id,j=j))
        all_c.sort(key=lambda q:(q["score"],q["parent"],q["j"]))
        keep=2 if wants_branch or (len(states)>1 and i<open_branches_until) else 1
        keep=min(keep,len(all_c))
        if wants_branch and keep==2:
            open_branches_until=i+arm.branch_hold
            counters["branches_spawned"]+=1
        if len(states)>1 and keep==1:
            counters["branches_pruned"]+=1
        next_states=[]
        for rank,cand in enumerate(all_c[:keep]):
            xx=cand["x"]
            if should_correct(arm,i) and arm.placement in ("post","both"):
                xx=cand["adjusted"] if cand["adjusted"] is not None else correct(xx,i,tn)
            next_states.append(dict(x=xx.detach(),eps=cand["eps"]))
        states=next_states
        counters["branch_max"]=max(counters["branch_max"],len(states))
        if i%max(1,arm.steps//10)==0 or i==arm.steps-2:
            audit_trace.append(dict(step=i, timestep=t, k=candidate_k,
                                    active_parent_count=len(states),
                                    score=all_c[0]["score"],
                                    denoiser_calls=counters["unet"],
                                    correction_calls=counters["fast_updates"]+counters["gradient_updates"]))
    if len(states)!=1:
        raise RuntimeError("beam never collapsed; adjust branch-hold schedule")
    out=states[0]["x"].detach()
    assert tuple(out.shape)==(1,3,256,256)
    counters["branch_max"]=max(counters["branch_max"],len(states))
    return out, dict(counters=counters,trace=audit_trace)
