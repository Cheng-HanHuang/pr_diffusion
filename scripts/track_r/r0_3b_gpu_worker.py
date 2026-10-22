#!/usr/bin/env python3
"""One physical GPU: R0.3b batch-8 or batch-64 prior smoke; optional UnivFD.

Only reads the frozen candidate bytes. Never accesses GT or oracle summaries.
Writes per-candidate append-only JSONL and final receipt; fails closed.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def safe(p):
    p=Path(p)
    for s in (str(p).lower(),str(p.resolve(strict=False)).lower()):
        if "confirmation" in s or "confirm305" in s:
            raise RuntimeError("STOP_SMOKE_PROTECTED_PATH")
    if not p.is_file():raise RuntimeError("STOP_SMOKE_INPUT_MISSING "+str(p))
    return p


def emit(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


def test_gpu(gpu):
    if gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(gpu):
        raise RuntimeError("STOP_SMOKE_PHYSICAL_GPU_BINDING")
    lines=subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines()
    usage={}
    for s in lines:
        i,mb=(v.strip() for v in s.split(","))
        usage[int(i)]=int(mb)
    if gpu not in usage or usage[gpu]>1024:
        raise RuntimeError("STOP_SMOKE_GPU_BUSY "+str(usage))
    return usage[gpu]


def preflight(a):
    p=Path(a.out)
    if not str(p).startswith(str(BASE)+"/") or p.exists() or p.is_symlink():
        raise RuntimeError("STOP_SMOKE_OUTPUT_NOT_FRESH")
    got=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if got!=a.expected_head or dirty or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_SMOKE_BAD_GIT_OR_BYTECODE_GUARD")
    if a.batch_size not in (8,64):raise RuntimeError("STOP_SMOKE_UNEXPECTED_BATCH")
    selection=safe(a.selection)
    if sha(selection)!=a.selection_sha:
        raise RuntimeError("STOP_SMOKE_SELECTION_SHA")
    manifest=json.loads(selection.read_text())
    if manifest.get("schema")!="prdiff.track_r.r0.3b.smoke-selection.v1" or manifest.get("head")!=a.expected_head:
        raise RuntimeError("STOP_SMOKE_SELECTION_SCHEMA_OR_HEAD")
    rows=manifest["candidates"]
    if len(rows)!=8 or len({r["uid"] for r in rows})!=8:
        raise RuntimeError("STOP_SMOKE_SELECTION_COUNT")
    for r in rows:
        if set(r)!={"role","uid","candidate_path","candidate_format",
                    "candidate_sha256","source","family","image_id"}:
            raise RuntimeError("STOP_SMOKE_SELECTION_UNEXPECTED_FIELD")
        if sha(safe(r["candidate_path"]))!=r["candidate_sha256"]:
            raise RuntimeError("STOP_SMOKE_CANDIDATE_SHA "+r["uid"])
    cfg=json.loads((WT/"configs/track_r/pac_paths_track_r_v1.json").read_text())
    weight=safe(cfg["models"]["ffhq_ddpm"]["path"])
    if sha(weight)!=cfg["models"]["ffhq_ddpm"]["sha256"]:
        raise RuntimeError("STOP_SMOKE_DDPM_SHA")
    dif=cfg["external"]["difffpr"]
    if subprocess.check_output(
        ["git","-C",dif["path"],"rev-parse","HEAD"],text=True).strip()!=dif["commit"]:
        raise RuntimeError("STOP_SMOKE_DIFF_FPR_HEAD")
    if a.univfd:
        ext=cfg["external"]["univfd"]
        root=Path(ext["path"])
        if subprocess.check_output(
            ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()!=ext["commit"]:
            raise RuntimeError("STOP_SMOKE_UNIVFD_HEAD")
        if sha(safe(root/"pretrained_weights/fc_weights.pth"))!=ext["fc_weights_sha256"]:
            raise RuntimeError("STOP_SMOKE_UNIVFD_FC_SHA")
        clip=cfg["models"]["clip_vit_l14"]
        if sha(safe(clip["path"]))!=clip["sha256"]:
            raise RuntimeError("STOP_SMOKE_CLIP_SHA")
    return cfg,rows,got


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--selection",required=True)
    ap.add_argument("--selection-sha",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",required=True,type=int)
    ap.add_argument("--batch-size",required=True,type=int)
    ap.add_argument("--univfd",action="store_true")
    a=ap.parse_args()
    cfg,rows,head=preflight(a)
    usage=test_gpu(a.physical_gpu)
    dest=Path(a.out)
    dest.mkdir(parents=True,exist_ok=False)
    receipt={"schema":"prdiff.track_r.r0.3b.gpu-worker.v1",
             "status":"INCOMPLETE","git_head":head,
             "physical_gpu":a.physical_gpu,
             "batch_size":a.batch_size,"univfd_enabled":a.univfd,
             "selection_sha256":a.selection_sha,"gpu_initial_memory_mib":usage,
             "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "completed":0,"gpu_hours_measured":None,"uids":[]}
    emit(dest/"R0_3B_RUNNING.json",receipt)
    started=time.monotonic()
    try:
        import torch
        from prdiffusion.track_r.physics import (
            to_rgb8,from_rgb8,apply_orbit,ORBIT_PATTERNS)
        from prdiffusion.track_r.prior_score import PriorScorer,frozen_alpha_schedule
        from prdiffusion.guided_backend import load_guided_diffusion_model
        from scripts.track_r.r0_2_g0_cpu import candidate_source

        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("STOP_SMOKE_EXACTLY_ONE_CUDA")
        props=torch.cuda.get_device_properties(0)
        if (props.major,props.minor)!=(12,0):
            raise RuntimeError("STOP_SMOKE_NOT_BLACKWELL")
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device("cuda:0")
        bundle=load_guided_diffusion_model(
            model_path=cfg["models"]["ffhq_ddpm"]["path"],device=device,
            preset="difffpr_ffhq_10m",
            guided_diffusion_dir=cfg["external"]["difffpr"]["path"],strict=True)
        alpha_diff=float((bundle.scheduler.alphas_cumprod.cpu() -
                          frozen_alpha_schedule()).abs().max())
        if alpha_diff>1e-6:raise RuntimeError("STOP_SMOKE_SCHEDULE_DRIFT")
        scorer=PriorScorer(
            eps_model=lambda xt,t:bundle.unet(xt,t).sample,
            timesteps=(50,100,200,300,400,500,600,700),
            draws=4,device=device,batch_size=a.batch_size)
        fd=None
        if a.univfd:
            from prdiffusion.track_r.univfd import UnivFD
            uroot=Path(cfg["external"]["univfd"]["path"])
            fd=UnivFD(
                uroot,uroot/"pretrained_weights/fc_weights.pth",
                cfg["models"]["clip_vit_l14"]["path"],device=device)
        torch.cuda.reset_peak_memory_stats(device)
        rows_out=dest/"R0_3B_CANDIDATES.jsonl"
        with rows_out.open("x",encoding="utf-8") as f:
            for i,r in enumerate(rows):
                source={"candidate_path":r["candidate_path"],
                        "candidate_format":r["candidate_format"]}
                x8=to_rgb8(candidate_source(source))
                torch.cuda.synchronize(device)
                start=time.monotonic()
                score=scorer.score_orbit(from_rgb8(x8),r["uid"])
                fd_result=None
                if fd:
                    imgs=torch.cat([apply_orbit(x8,p) for p in ORBIT_PATTERNS],dim=0)
                    scores=fd.p_fake(imgs).detach().cpu().tolist()
                    repaired=apply_orbit(x8,score["s_star"])
                    actual=fd.p_fake(repaired).item()
                    chosen=scores[ORBIT_PATTERNS.index(score["s_star"])]
                    if abs(actual-chosen)>1e-5:
                        raise RuntimeError("STOP_SMOKE_UNIVFD_REPAIRED_REPLAY")
                    fd_result={"per_orbit":dict(zip(ORBIT_PATTERNS,scores)),
                               "repaired":actual,
                               "argmin":ORBIT_PATTERNS[min(
                                   range(8),key=lambda j:(scores[j],j))]}
                torch.cuda.synchronize(device)
                elapsed=time.monotonic()-start
                row={"uid":r["uid"],"role":r["role"],
                     "input_sha256":r["candidate_sha256"],
                     "prior":score,"univfd":fd_result,
                     "wall_seconds":elapsed}
                f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
                f.flush()
                receipt["uids"].append(r["uid"])
                receipt["completed"]=i+1
                print("SMOKE_PROGRESS",a.physical_gpu,a.batch_size,i+1,
                      "of",len(rows),"uid",r["uid"],
                      "s_star",score["s_star"],
                      "elapsed",round(elapsed,3),flush=True)
        receipt["gpu_peak_reserved_mib"]=torch.cuda.max_memory_reserved()/1048576
        receipt["unet_forwards"]=sum(256 for _ in rows)
        receipt["univfd_image_forwards"]=fd.forward_images if fd else 0
        receipt["candidate_jsonl_sha256"]=sha(rows_out)
        receipt["elapsed_seconds"]=time.monotonic()-started
        receipt["gpu_hours_measured"]=receipt["elapsed_seconds"]/3600.0
        receipt["status"]="R0_3B_GPU_WORKER_PASS"
        emit(dest/"R0_3B_WORKER_RECEIPT.json",receipt)
        print("R0_3B_GPU_WORKER_PASS",a.physical_gpu,
              a.batch_size,receipt["elapsed_seconds"],flush=True)
        return 0
    except Exception as e:
        receipt["status"]="STOP_R0_3B_GPU_WORKER"
        receipt["error"]=repr(e)
        receipt["traceback"]=traceback.format_exc()[-2500:]
        receipt["elapsed_seconds"]=time.monotonic()-started
        receipt["gpu_hours_measured"]=receipt["elapsed_seconds"]/3600.0
        emit(dest/"R0_3B_WORKER_FAILURE.json",receipt)
        print("STOP_R0_3B_GPU_WORKER",repr(e),flush=True)
        return 8

if __name__=="__main__":
    sys.exit(main())
