#!/usr/bin/env python3
"""A1.1a: one wholly procedural RGB image and native DAPS-operator measurement.

Engineering-only synthetic input. CPU-only. Never enumerate/read FFHQ or B24
data, inspect protected confirmation, load checkpoint or run native diffusion.
Generates no solver candidates and makes no paper-quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path("/egr/research-pac/huang248")
REPO=ROOT/"pr_diffusion_track_a_a1_oracle_budget"
DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
OUT_PARENT=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
DAPS_HEAD="e7a77d094167084faed19b599b96673b7bb11447"
DAPS_DIFF="fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69"
PROBE_LABEL="A1_SYNTHETIC_PROCEDURAL_PROBE_V1"
N=256
SIGMA=0.05
OVERSAMPLE=2.0
MEAS_SEED=12061018


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_content_sha256(value: torch.Tensor) -> str:
    v=value.detach().cpu().contiguous()
    header=json.dumps({"dtype":str(v.dtype),"shape":list(v.shape)},
                       separators=(",",":"),sort_keys=True).encode()
    h=hashlib.sha256()
    h.update(header)
    h.update(b"\0")
    h.update(v.numpy().tobytes(order="C"))
    return h.hexdigest()


def procedural_rgb_png() -> np.ndarray:
    """No external image, training sample, RNG or GT metadata dependence."""
    yy,xx=np.mgrid[0:N,0:N].astype(np.float32)
    u=(xx-127.5)/127.5
    v=(yy-127.5)/127.5
    envelope=np.exp(-2.25*(u*u+v*v))
    waves=np.sin(5*np.pi*u+3*v)+0.5*np.cos(6*np.pi*v-2*u)
    blob1=np.exp(-((u+0.34)**2+(v-0.05)**2)/0.05)
    blob2=np.exp(-((u-0.30)**2+(v-0.07)**2)/0.045)
    b3=np.exp(-((u)**2+(v-0.32)**2)/0.11)
    r=0.36+0.26*envelope+0.09*waves+0.17*blob1
    g=0.39+0.19*envelope+0.13*np.cos(3*np.pi*u*v)+0.21*blob2
    b=0.47+0.16*envelope+0.12*np.sin(6*v+2*u)+0.23*b3
    raw=np.stack((r,g,b),axis=-1)
    return np.rint(np.clip(raw,0,1)*255).astype(np.uint8)


def model_range_tensor(png_bytes: np.ndarray) -> torch.Tensor:
    if png_bytes.shape!=(N,N,3) or png_bytes.dtype!=np.uint8:
        raise ValueError("unexpected synthetic RGB bytes")
    # Matches torchvision.transforms.ToTensor() for an 8-bit 256x256 PNG
    t=torch.from_numpy(png_bytes.copy()).permute(2,0,1).float().div(255)
    return (2*t-1).unsqueeze(0).contiguous()


def run_git(path: Path, args: list[str]) -> str:
    r=subprocess.run(["git","-C",str(path),*args],capture_output=True,
                     text=True,timeout=30,check=False,
                     env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if r.returncode:
        raise RuntimeError("native source read-only Git validation failure")
    return r.stdout.strip()


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--expected-head",required=True)
    args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=OUT_PARENT or out.name!="a1_1a_synthetic_probe_v1" or out.exists():
        raise RuntimeError("A1.1a output not new or outside exclusive allowed namespace")
    if run_git(REPO,["rev-parse","HEAD"])!=args.expected_head:
        raise RuntimeError("A1 git source SHA not pinned")
    if run_git(DAPS,["rev-parse","HEAD"])!=DAPS_HEAD:
        raise RuntimeError("native DAPS HEAD changed")
    diff=subprocess.run(["git","-C",str(DAPS),"diff","HEAD","--"],
                        capture_output=True,check=False,timeout=30,
                        env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if diff.returncode or hashlib.sha256(diff.stdout).hexdigest()!=DAPS_DIFF:
        raise RuntimeError("native DAPS tracked source diff identity changed")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("this engineering input stage requires disabled CUDA visibility")
    # Do not import external code unless all source checks passed.
    sys.path.insert(0,str(DAPS))
    from forward_operator import PhaseRetrieval  # noqa: E402

    raw=procedural_rgb_png()
    x=model_range_tensor(raw)
    if not torch.isfinite(x).all() or tuple(x.shape)!=(1,3,256,256):
        raise RuntimeError("synthetic GT tensor identity violation")
    torch.manual_seed(MEAS_SEED)
    np.random.seed(MEAS_SEED)
    operator=PhaseRetrieval(oversample=OVERSAMPLE,resolution=N,sigma=SIGMA)
    with torch.no_grad():
        y=operator.measure(x).detach().cpu().contiguous()
    if tuple(y.shape)!=(1,3,384,384) or y.dtype!=torch.float32 or not torch.isfinite(y).all():
        raise RuntimeError(f"native measurement schema incompatible: {y.shape}, {y.dtype}")
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected CUDA initialization on CPU-only stage")
    out.mkdir(parents=True,exist_ok=False)
    png=out/"synthetic_probe.png"
    gt=out/"ground_truth_tensor.pt"
    obs=out/"measurement.pt"
    Image.fromarray(raw,mode="RGB").save(png)
    torch.save({"ground_truth":x},gt)
    torch.save(y,obs)
    payload={
        "schema":"prdiff.a1.procedural-input.v1",
        "probe_label":PROBE_LABEL,
        "synthetic_only":True,
        "ffhq_accessed":False,
        "quality_inference_valid":False,
        "gpu_used":False,
        "checkpoint_loaded":False,
        "solver_executed":False,
        "locked_confirmation_access":False,
        "source_a1_head":args.expected_head,
        "native_daps_head":DAPS_HEAD,
        "native_daps_full_tracked_diff_sha256":DAPS_DIFF,
        "image_path":str(png),
        "image_file_sha256":sha256_file(png),
        "ground_truth_tensor_path":str(gt),
        "gt_file_sha256":sha256_file(gt),
        "ground_truth_tensor_sha256":tensor_content_sha256(x),
        "measurement_path":str(obs),
        "measurement_file_sha256":sha256_file(obs),
        "measurement_tensor_sha256":tensor_content_sha256(y),
        "measurement_seed":MEAS_SEED,
        "measurement_min":float(y.min()),
        "measurement_max":float(y.max()),
        "shape":[1,3,384,384],
        "dtype":str(y.dtype),
        "model_range_shape":[1,3,256,256],
        "native_forward_operator":"DAPS PhaseRetrieval.measure",
        "sigma_y":SIGMA,
        "oversample":OVERSAMPLE,
        "noisy_signed_observation_preserved":True,
        "np_historical_preprocessing":"clamp_min_zero_in_memory ONLY at NP solver",
        "GT_use_policy":"SHAPE-ONLY FOR SITCOM START; OFFLINE SCORING AFTER RUN",
    }
    path=out/"A1_1A_PROBE_MANIFEST.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps({"status":"A1_1A_SYNTHETIC_NATIVE_MEASUREMENT_PASS",
                      "shape":payload["shape"],"native_head":DAPS_HEAD,
                      "gt_content_sha256":payload["ground_truth_tensor_sha256"],
                      "measurement_content_sha256":payload["measurement_tensor_sha256"],
                      "gpu_used":False,"ffhq_accessed":False},sort_keys=True))
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"A1_1A_STOP: {type(e).__name__}: {e}")
        sys.exit(3)
