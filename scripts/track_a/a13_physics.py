#!/usr/bin/env python3
"""A1.3 CPU-only Fourier and selection primitives. No checkpoint/model imports.

Exact pinned DAPS/SITCOM fastmri_utils.fft2c_new convention:
ifftshift -> fftn(norm='ortho') -> fftshift of complex zero-padded [0,1] RGB.
All observable scores here are NEW external terminal diagnostics, not native
historical K4 selectors. GT is unavailable in score/selection functions.
"""
from __future__ import annotations
import math
import os
import time

def require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STOP_CUDA_VISIBLE_DEVICES: must be empty")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("STOP_CUDA_INITIALIZED")
    return torch

def magnitude(x, *, pad=64):
    """x: CPU float32 [1,3,256,256] model range [-1,1], -> [1,3,384,384]."""
    torch = require_cpu()
    import torch.nn.functional as F
    if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
        raise ValueError("CPU tensor required")
    if x.dtype != torch.float32 or tuple(x.shape) != (1,3,256,256):
        raise ValueError("wrong model tensor dtype/shape")
    if pad != 64 or not bool(torch.isfinite(x).all()):
        raise ValueError("invalid pad or nonfinite image")
    x01 = x * 0.5 + 0.5  # native does NOT clamp the floating model image
    padded = F.pad(x01, (pad,pad,pad,pad)).to(torch.complex64)
    out = torch.fft.fftshift(torch.fft.fftn(
        torch.fft.ifftshift(padded, dim=(-2,-1)),
        dim=(-2,-1), norm="ortho"), dim=(-2,-1)).abs()
    if tuple(out.shape) != (1,3,384,384) or not bool(torch.isfinite(out).all()):
        raise RuntimeError("operator output invalid")
    return out

def residuals(image_model, observation, *, eps=1e-12):
    torch = require_cpu()
    if observation.device.type!="cpu" or observation.dtype!=torch.float32 or tuple(observation.shape)!=(1,3,384,384):
        raise ValueError("observation shape/dtype/device incompatible")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("nonfinite observation")
    t0=time.perf_counter()
    pred = magnitude(image_model)
    target_plus = observation.clamp_min(0)
    def stats(target):
        err = (pred-target).double()
        den = torch.linalg.vector_norm(target.double()).item()
        sq = err.square()
        norm = torch.linalg.vector_norm(err).item()
        return dict(relative_l2=norm/(den+eps), sse=float(sq.sum()),
                    rmse=math.sqrt(float(sq.mean())),
                    per_channel_sse=[float(v) for v in sq.sum(dim=(0,2,3)).flatten()],
                    target_l2=den)
    signed=stats(observation)
    clipped=stats(target_plus)
    return dict(signed=signed, clipped=clipped,
                negative_count=int((observation<0).sum()),
                total_measurements=observation.numel(),
                negative_fraction=float((observation<0).float().mean()),
                fft_forward_calls=1, extra_cpu_score_seconds=time.perf_counter()-t0)

def png8_gt(gt):
    torch=require_cpu()
    if gt.device.type!="cpu" or gt.dtype not in (torch.float32,torch.float64) or tuple(gt.shape)!=(1,3,256,256):
        raise ValueError("GT mismatch")
    return torch.round(((gt.float().clamp(-1,1)+1)/2)*255).clamp(0,255)/255

def psnr_png8(pred01, quantized_gt01):
    torch=require_cpu()
    if tuple(pred01.shape)!=(1,3,256,256) or tuple(quantized_gt01.shape)!=(1,3,256,256):
        raise ValueError("PNG8 PSNR tensors wrong shape")
    if not bool(torch.isfinite(pred01).all()) or not bool(torch.isfinite(quantized_gt01).all()):
        raise ValueError("nonfinite PSNR inputs")
    mse=(pred01-quantized_gt01).square().mean().clamp_min(1e-12)
    return float(-10*torch.log10(mse))

def png_to_model_and_01(path):
    torch=require_cpu()
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        if im.size!=(256,256):
            raise ValueError("saved PNG not 256x256")
        # Exactly A1.2 score_pngs: convert to RGB even for tagged PNG modes.
        a=np.asarray(im.convert("RGB"),dtype=np.float32)
    img01=torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0)/255.0
    return (img01*2.0-1.0).contiguous(), img01

def select_min(rows, key):
    """GT-free stable argmin. Rejects missing/nonfinite; ties: original rep order."""
    if not rows:
        raise ValueError("empty candidate set")
    for row in rows:
        if key not in row or not isinstance(row[key],(float,int)) or not math.isfinite(float(row[key])):
            raise ValueError("missing/nonfinite score; no silent fallback")
    return min(enumerate(rows),key=lambda pair:(float(pair[1][key]),pair[0]))[1]

def select_oracle(rows):
    """OFFLINE EVALUATION ONLY; keep separate from select_min and features."""
    if not rows or any("psnr_raw_rgb8_db" not in x or not math.isfinite(x["psnr_raw_rgb8_db"]) for x in rows):
        raise ValueError("unavailable oracle GT metric")
    return max(enumerate(rows),key=lambda z:(z[1]["psnr_raw_rgb8_db"],-z[0]))[1]

def source_witness():
    """Known centered-FFT impulse/DC witness, no native module or model imports."""
    torch=require_cpu()
    x=torch.full((1,3,256,256),-1.,dtype=torch.float32)
    x[:,:,128,128]=1.
    a=magnitude(x)
    expected=1.0/384.0  # one spatial unit in each channel, ortho 384x384 FFT
    if not torch.allclose(a,torch.full_like(a,expected),atol=2e-7,rtol=2e-5):
        raise RuntimeError("STOP_OPERATOR_NOT_RECONCILED: impulse/scale/centering")
    z=torch.full_like(x,-1.)
    zz=magnitude(z)
    if float(zz.abs().max())!=0:
        raise RuntimeError("STOP_OPERATOR_NOT_RECONCILED: zero")
    return {"impulse_max_absolute_error":float((a-expected).abs().max()),"fft_norm":"ortho",
            "pad":64,"shape":[1,3,384,384]}

def symmetry_witness():
    torch=require_cpu()
    y=torch.arange(256,dtype=torch.float32)
    xx=y[None,:]; yy=y[:,None]
    u=(0.13*torch.sin(xx/21)+0.11*torch.cos(yy/18)+0.16*(xx>yy)).clamp(-.9,.9)
    x=u[None,None].repeat(1,3,1,1).contiguous()
    flipped=torch.rot90(x,2,dims=(-2,-1))
    a=magnitude(x); b=magnitude(flipped)
    diff=torch.linalg.vector_norm((a-b).double()).item()
    rel=diff/(torch.linalg.vector_norm(a.double()).item()+1e-12)
    channels=[float(torch.linalg.vector_norm((a-b)[0,i].double())) for i in range(3)]
    # For even 384 FFT and centered symmetric padding, 180-degree reversal
    # coincides with spatial inversion modulo lattice translation.
    if rel>2e-5:
        raise RuntimeError("STOP_OPERATOR_NOT_RECONCILED: symmetry")
    return {"relative_rot180_difference":rel,"per_channel_l2_difference":channels,
            "rot180_operator_symmetry_reconciled":True}
