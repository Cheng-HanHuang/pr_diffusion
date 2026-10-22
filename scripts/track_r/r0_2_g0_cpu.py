#!/usr/bin/env python3
"""R0.2 CPU G0 operator/pairing, symmetry and replay checks.

FAIL is a real stop; no GPU work is authorized by this script. The cross-GPU
determinism subgate is explicitly PENDING until R0.3. Do not hide mismatches.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from prdiffusion.track_r.physics import (
    magnitude, orbit_invariance_relative, to_rgb8, load_png_rgb8,
    psnr_rgb8, rho_signed, centered_fft_witness
)
from prdiffusion.track_r.manifest import read_manifest,assert_no_forbidden

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
PASS_LO=0.0495
PASS_HI=0.0505
WARN_HI=0.055
REPLAY_TOL=0.01
REPLAY_REQUIRED=0.99
GT_TRANSFORM=T.Compose([T.ToTensor(),T.Resize(256),T.CenterCrop(256)])

def clean(path):
    p=Path(assert_no_forbidden(str(path)))
    if not p.is_absolute() or not p.is_file():
        raise RuntimeError("STOP_G0_INPUT_MISSING "+str(p))
    assert_no_forbidden(str(p.resolve()))
    return p

def read_tensor(p,keys,shape):
    p=clean(p)
    try:
        value=torch.load(str(p),map_location="cpu",weights_only=True)
    except TypeError:
        raise RuntimeError("STOP_G0_WEIGHTS_ONLY_UNSUPPORTED")
    if isinstance(value,dict):
        present=[k for k in keys if k in value and isinstance(value[k],torch.Tensor)]
        if len(present)!=1:
            raise RuntimeError("STOP_G0_TENSOR_CONTAINER "+str(p)+" keys="+str(list(value)[:12]))
        value=value[present[0]]
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape:
        raise RuntimeError("STOP_G0_BAD_TENSOR_SHAPE "+str(p)+" "+str(getattr(value,"shape",None)))
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise RuntimeError("STOP_G0_BAD_TENSOR_DTYPE_OR_FINITE "+str(p))
    return value.detach().to(dtype=torch.float32,device="cpu")

def gt_source(row):
    typ=row["gt_format"];p=clean(row["gt_path"])
    if typ=="ffhq1024":
        with Image.open(p) as img:
            if img.size!=(1024,1024):
                raise RuntimeError("STOP_G0_FFHQ_SOURCE_NOT1024 "+str(p))
            return (GT_TRANSFORM(img.convert("RGB"))*2-1).unsqueeze(0)
    if typ=="pt":
        return read_tensor(p,("ground_truth","image","x"),(1,3,256,256))
    if typ=="png":
        return load_png_rgb8(p).float()/127.5-1
    raise RuntimeError("STOP_G0_UNSUPPORTED_GT_FORMAT")

def candidate_source(row):
    p=clean(row["candidate_path"])
    if row["candidate_format"]=="png":
        return load_png_rgb8(p).float()/127.5-1
    if row["candidate_format"]=="pt":
        return read_tensor(p,("reconstruction","image","x","sample","terminal"),(1,3,256,256))
    raise RuntimeError("STOP_G0_UNSUPPORTED_CANDIDATE_FORMAT")

def psnr_float(x,gt):
    mse=(x.clamp(-1,1).float()-gt.clamp(-1,1).float()).square().mean()/4
    return float(-10*torch.log10(mse.clamp_min(1e-12)))

def status_gt(v):
    if PASS_LO<=v<=PASS_HI:return "PASS"
    if PASS_HI<v<=WARN_HI:return "WARN"
    return "FAIL"

def save(path,obj):
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--repo-root",required=True)
    parser.add_argument("--expected-head",required=True)
    args=parser.parse_args()
    out=Path(args.out);repo=Path(args.repo_root).resolve()
    if not str(out).startswith(str(BASE)+os.sep) or out.exists():
        raise RuntimeError("STOP_G0_OUTPUT_FRESHNESS")
    head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True)
    if head!=args.expected_head or dirty:
        raise RuntimeError("STOP_G0_GIT_IDENTITY")
    rows=read_manifest(clean(args.manifest))
    if len(rows)!=2520:
        raise RuntimeError("STOP_G0_MANIFEST_COUNT "+str(len(rows)))
    out.mkdir(parents=True,exist_ok=False)
    receipt={"schema":"prdiff.track_r.r0_2.g0_cpu.v1",
             "git_head":head,"manifest_path":str(args.manifest),
             "n_candidates":len(rows),"gpu_calls":0,"cross_gpu_determinism":"PENDING_R0_3",
             "gates":{}, "mismatches":[],
             "source_family":{},"gt_warnings":[],"gt_failures":[]}
    checks=receipt["gates"]
    try:
        witness=centered_fft_witness()
        checks["G0_fft_witness"]={"status":"PASS","detail":witness}
        rng=torch.Generator(device="cpu").manual_seed(20261021)
        inv=[]
        for i in range(16):
            x=torch.rand((1,3,256,256),generator=rng)*2-1
            inv.append(max(orbit_invariance_relative(x).values()))
        unique={}
        for r in rows:
            key=(r["source"],r["image_id"],r["measurement_path"],r["gt_path"])
            unique.setdefault(key,r)
        for key,row in list(unique.items())[:3]:
            inv.append(max(orbit_invariance_relative(gt_source(row)).values()))
        checks["G0_orbit_invariance"]={
            "status":"PASS" if max(inv)<=1e-4 else "FAIL",
            "max_relative":max(inv),"random":16,"real":3}
        per_pair={}; gtcache={};ycache={}
        for key,row in unique.items():
            gt=gt_source(row)
            y=read_tensor(row["measurement_path"],("measurement","y","observation"),(1,3,384,384))
            r=float(rho_signed(gt,y).item())
            per_pair[key]=r
            gtcache[key]=gt;ycache[key]=y
            status=status_gt(r)
            info={"source":row["source"],"image_id":row["image_id"],"rho_gt":r}
            if status=="WARN":receipt["gt_warnings"].append(info)
            if status=="FAIL":receipt["gt_failures"].append(info)
        checks["G0_operator_pairing"]={
            "status":"FAIL" if receipt["gt_failures"] else
                     ("WARN" if receipt["gt_warnings"] else "PASS"),
            "n_pairs":len(per_pair),"n_warn":len(receipt["gt_warnings"]),
            "n_fail":len(receipt["gt_failures"]),
            "rho_min":min(per_pair.values()),"rho_max":max(per_pair.values())}
        nmatch=0
        family_counts=Counter(); family_match=Counter()
        for i,row in enumerate(rows):
            key=(row["source"],row["image_id"],row["measurement_path"],row["gt_path"])
            c=candidate_source(row)
            gt=gtcache[key]
            if row["recorded_psnr_convention"].endswith("FLOAT_CLAMP"):
                replay=psnr_float(c,gt)
            else:
                replay=float(psnr_rgb8(to_rgb8(c),to_rgb8(gt))[0])
            reference=float(row["recorded_psnr_db"])
            diff=abs(replay-reference)
            fk=row["source"]+"/"+row["family"]
            family_counts[fk]+=1
            matched=diff<=REPLAY_TOL
            if matched:nmatch+=1;family_match[fk]+=1
            else:
                receipt["mismatches"].append({
                    "uid":row["uid"],"recorded":reference,
                    "replayed":replay,"abs_diff_db":diff,
                    "original_candidate_format":row["candidate_format"]})
            if (i+1)%250==0:
                print("G0_REPLAY_PROGRESS",i+1,"of",len(rows),flush=True)
        checks["G0_recorded_psnr_replay"]={
            "status":"PASS" if nmatch/len(rows)>=REPLAY_REQUIRED else "FAIL",
            "n_total":len(rows),"n_match":nmatch,"fraction":nmatch/len(rows),
            "tolerance_db":REPLAY_TOL}
        receipt["source_family"]={
            k:{"n":v,"matched":family_match[k],
               "fraction":family_match[k]/v} for k,v in sorted(family_counts.items())}
        checks["G0_protected_path"]={"status":"PASS","checked_paths":len(rows)*3,
            "policy":"every candidate, GT, y path checked before read"}
        required=("G0_fft_witness","G0_orbit_invariance",
                  "G0_operator_pairing","G0_recorded_psnr_replay","G0_protected_path")
        receipt["status"]="G0_CPU_PRECHECK_PASS_PENDING_DETERMINISM" if all(
            checks[k]["status"]=="PASS" for k in required) else "G0_CPU_PRECHECK_NOT_PASS"
        save(out/"R0_2_G0_CPU_RECEIPT.json",receipt)
        print("R0_2_G0_STATUS",receipt["status"],"REPLAY",nmatch,len(rows),
              "PAIRS",len(per_pair),"GT_WARN",len(receipt["gt_warnings"]),
              "GT_FAIL",len(receipt["gt_failures"]),flush=True)
        return 0 if receipt["status"].endswith("PENDING_DETERMINISM") else 7
    except Exception as e:
        fail={"status":"G0_CPU_PRECHECK_EXCEPTION","error":repr(e),
              "traceback":traceback.format_exc()[-2500:]}
        save(out/"R0_2_G0_EXCEPTION.json",fail)
        print("STOP_G0_EXCEPTION",repr(e),flush=True)
        return 8

if __name__=="__main__":
    sys.exit(main())
