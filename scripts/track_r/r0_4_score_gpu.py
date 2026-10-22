#!/usr/bin/env python3
"""R0.4 per-shard GT-free prior, UnivFD and signed/clipped residual worker.

Each candidate has one immutable, independently verifiable receipt. An
interrupted shard can resume by verifying existing candidate receipts.
The only external inputs are sealed candidate + signed measurement bytes.
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

import torch

from prdiffusion.track_r.physics import (
    ORBIT_PATTERNS,apply_orbit,from_rgb8,rho_signed,rho_clipped,to_rgb8,
    load_png_rgb8)
from prdiffusion.track_r.prior_score import PriorScorer,frozen_alpha_schedule
from prdiffusion.track_r.univfd import UnivFD

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def safe(path):
    p=Path(path)
    if (not p.is_file() or not p.is_absolute() or p.is_symlink()
        or any(t in s for t in ("confirmation","confirm305") for s in
          (str(p).lower(),str(p.resolve(strict=False)).lower()))):
        raise RuntimeError("STOP_R04_PROTECTED_OR_MISSING_FILE "+str(p))
    return p

def write_x(path,data):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def git_head(path):
    return subprocess.check_output(
        ["git","-C",str(path),"rev-parse","HEAD"],text=True,timeout=30).strip()

def validate_input(payload,expected_head):
    if (payload.get("status")!="R0_4_TASKS_FROZEN" or
        payload.get("head")!=expected_head or
        payload.get("schema")!="prdiff.track_r.r0.4.gt_free_tasks.v1" or
        payload.get("n_candidates")!=2520 or
        payload.get("gt_or_psnr_fields_in_gpu_tasks") is not False):
        raise RuntimeError("STOP_R04_BAD_TASKS_RECEIPT")
    if len(payload["tasks"])!=2520:
        raise RuntimeError("STOP_R04_TASKS_LENGTH")
    allowed={"uid","source","family","image_id","run_index",
             "candidate_path","candidate_format","candidate_sha256",
             "measurement_path","measurement_sha256"}
    if any(set(t)!=allowed for t in payload["tasks"]):
        raise RuntimeError("STOP_R04_WORKER_TASKS_NOT_GT_FREE")
    return payload["tasks"]

def candidate_tensor(row):
    path=safe(row["candidate_path"])
    if sha(path)!=row["candidate_sha256"]:
        raise RuntimeError("STOP_R04_CANDIDATE_SHA "+row["uid"])
    if row["candidate_format"]=="png":
        return load_png_rgb8(path)
    if row["candidate_format"]!="pt":
        raise RuntimeError("STOP_R04_CANDIDATE_FORMAT")
    record=torch.load(path,map_location="cpu",weights_only=True)
    if isinstance(record,dict):
        keys=[k for k in ("reconstruction","image","x","sample","terminal")
              if isinstance(record.get(k),torch.Tensor)]
        if len(keys)!=1:
            raise RuntimeError("STOP_R04_TENSOR_CONTAINER")
        record=record[keys[0]]
    if (not isinstance(record,torch.Tensor) or
        record.shape!=(1,3,256,256) or not record.is_floating_point() or
        not bool(torch.isfinite(record).all())):
        raise RuntimeError("STOP_R04_IMAGE_SHAPE_DTYPE")
    return to_rgb8(record.float())

def measurement_tensor(row,cache):
    path=safe(row["measurement_path"])
    if path not in cache:
        if sha(path)!=row["measurement_sha256"]:
            raise RuntimeError("STOP_R04_MEAS_SHA")
        data=torch.load(path,map_location="cpu",weights_only=True)
        if isinstance(data,dict):
            keys=[k for k in ("measurement","y","observation")
                  if isinstance(data.get(k),torch.Tensor)]
            if len(keys)!=1:
                raise RuntimeError("STOP_R04_MEAS_CONTAINER")
            data=data[keys[0]]
        if (not isinstance(data,torch.Tensor) or
            tuple(data.shape)!=(1,3,384,384) or
            not data.is_floating_point() or
            not bool(torch.isfinite(data).all())):
            raise RuntimeError("STOP_R04_MEAS_SCHEMA")
        if len(cache)>4:cache.clear()
        cache[path]=data.float()
    return cache[path]

def output_name(row):
    return hashlib.sha256(row["uid"].encode("utf-8")).hexdigest()+".json"

def validate_complete(path,row,task_sha,head):
    r=json.loads(safe(path).read_text())
    if (r.get("status")!="R0_4_CANDIDATE_PASS" or
        r.get("uid")!=row["uid"] or
        r.get("candidate_sha256")!=row["candidate_sha256"] or
        r.get("task_file_sha256")!=task_sha or
        r.get("executor_head")!=head or
        r.get("unet_forwards")!=256 or
        set(r.get("p_fake_orbit",{}))!=set(ORBIT_PATTERNS)):
        raise RuntimeError("STOP_R04_EXISTING_RESULT_INVALID "+row["uid"])
    return True

def run(a):
    out=Path(a.out)
    if not str(out).startswith(str(BASE)+"/"):
        raise RuntimeError("STOP_R04_OUTPUT_NAMESPACE")
    if git_head(WT)!=a.expected_head:
        raise RuntimeError("STOP_R04_GIT_HEAD")
    if subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True):
        raise RuntimeError("STOP_R04_DIRTY_WORKTREE")
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_R04_BYTECODE_GUARD")
    if a.physical_gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.physical_gpu):
        raise RuntimeError("STOP_R04_GPU_BINDING")
    if a.batch_size!=64 or a.count not in range(1,65) or a.start<0 or a.start+a.count>2520:
        raise RuntimeError("STOP_R04_SHARD_PARAMETERS")
    task_file=safe(a.tasks)
    task_sha=sha(task_file)
    if task_sha!=a.tasks_sha:
        raise RuntimeError("STOP_R04_TASK_FILE_SHA")
    payload=json.loads(task_file.read_text())
    # R0.4 four-GPU continuation reuses the frozen, audited 16-candidate
    # canary task file. Its producing Git head is deliberately *different*
    # from this worker's newer, parallel-orchestration Git head.
    producer=a.tasks_source_head or a.expected_head
    if producer != a.expected_head:
        canary=BASE/"r0_4_score_20261022T193238Z"/"prepare"/"R0_4_GT_FREE_TASKS.json"
        if (producer!="8aee60d98acfeba836fff4ec824137769b0a4698" or
            task_sha!="38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7" or
            task_file!=canary or task_file.is_symlink()):
            raise RuntimeError("STOP_R04_UNAPPROVED_SOURCE_TASKS_HEAD")
    rows=validate_input(payload,producer)[a.start:a.start+a.count]
    cfg=json.loads((WT/"configs/track_r/pac_paths_track_r_v1.json").read_text())
    ffhq=safe(cfg["models"]["ffhq_ddpm"]["path"])
    if sha(ffhq)!=cfg["models"]["ffhq_ddpm"]["sha256"]:
        raise RuntimeError("STOP_R04_DDPM_SHA")
    dif=cfg["external"]["difffpr"]
    if git_head(dif["path"])!=dif["commit"]:
        raise RuntimeError("STOP_R04_DIFFFPR_HEAD")
    univ=cfg["external"]["univfd"]
    univ_root=Path(univ["path"])
    if git_head(univ_root)!=univ["commit"]:
        raise RuntimeError("STOP_R04_UNIVFD_HEAD")
    if sha(safe(univ_root/"pretrained_weights/fc_weights.pth"))!=univ["fc_weights_sha256"]:
        raise RuntimeError("STOP_R04_FC_SHA")
    clip=safe(cfg["models"]["clip_vit_l14"]["path"])
    if sha(clip)!=cfg["models"]["clip_vit_l14"]["sha256"]:
        raise RuntimeError("STOP_R04_CLIP_SHA")
    if not out.exists():
        out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir() or out.is_symlink():
        raise RuntimeError("STOP_R04_SHARD_OUTPUT_NOT_DIR")
    result_root=out/"results"
    result_root.mkdir(exist_ok=True)
    job={"schema":"prdiff.track_r.r0.4.shard.v1",
         "executor_head":a.expected_head,"task_file_sha256":task_sha,
         "start":a.start,"count":a.count,"batch_size":a.batch_size,
         "gpu":a.physical_gpu,"uids":[row["uid"] for row in rows]}
    old=out/"R0_4_SHARD_META.json"
    if old.is_file():
        if json.loads(old.read_text())!=job:
            raise RuntimeError("STOP_R04_SHARD_META_MISMATCH")
    else:
        write_x(old,job)
    complete=out/"R0_4_SHARD_COMPLETE.json"
    if complete.exists():
        rec=json.loads(complete.read_text())
        if rec.get("status")!="R0_4_SHARD_PASS" or rec.get("meta_sha256")!=sha(old):
            raise RuntimeError("STOP_R04_SHARD_COMPLETE_MISMATCH")
        for row in rows:
            validate_complete(result_root/output_name(row),row,task_sha,a.expected_head)
        print("R04_SHARD_ALREADY_COMPLETE",a.start,a.count,flush=True)
        return 0
    busy={}
    for textline in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        gpu,used=[x.strip() for x in textline.split(",")]
        busy[int(gpu)]=int(used)
    if busy.get(a.physical_gpu,999999)>1024:
        raise RuntimeError("STOP_R04_GPU_BUSY "+str(busy))
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP_R04_DEVICE_COUNT")
    props=torch.cuda.get_device_properties(0)
    if (props.major,props.minor)!=(12,0):
        raise RuntimeError("STOP_R04_GPU_ARCH")
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from prdiffusion.guided_backend import load_guided_diffusion_model
    device=torch.device("cuda:0")
    start=time.monotonic()
    bundle=load_guided_diffusion_model(
        model_path=ffhq,device=device,preset="difffpr_ffhq_10m",
        guided_diffusion_dir=dif["path"],strict=True)
    if float((bundle.scheduler.alphas_cumprod.cpu() -
              frozen_alpha_schedule()).abs().max())>1e-6:
        raise RuntimeError("STOP_R04_DDPM_SCHEDULE")
    scorer=PriorScorer(lambda x,t:bundle.unet(x,t).sample,
        timesteps=(50,100,200,300,400,500,600,700),
        draws=4,device=device,batch_size=64)
    detector=UnivFD(univ_root,univ_root/"pretrained_weights/fc_weights.pth",
                    clip,device=device)
    memcache={}
    new=0; skipped=0
    for index,row in enumerate(rows):
        path=result_root/output_name(row)
        if path.exists():
            validate_complete(path,row,task_sha,a.expected_head)
            skipped+=1
            continue
        x8=candidate_tensor(row)
        y=measurement_tensor(row,memcache)
        xc=from_rgb8(x8)
        score_begin=time.monotonic()
        prior=scorer.score_orbit(xc,row["uid"])
        with torch.inference_mode():
            orbit=torch.cat([apply_orbit(x8,p) for p in ORBIT_PATTERNS])
            p_fake=detector.p_fake(orbit).detach().cpu().tolist()
            repaired=apply_orbit(x8,prior["s_star"])
            rep_fake=float(detector.p_fake(repaired)[0])
            corresponding=float(p_fake[ORBIT_PATTERNS.index(prior["s_star"])])
            if abs(rep_fake-corresponding)>1e-5:
                raise RuntimeError("STOP_R04_UNIVFD_REPAIRED_MISMATCH")
        rho0=float(rho_signed(xc,y)[0])
        rho1=float(rho_signed(from_rgb8(repaired),y)[0])
        clipped0=float(rho_clipped(xc,y)[0])
        clipped1=float(rho_clipped(from_rgb8(repaired),y)[0])
        wall=time.monotonic()-score_begin
        if not all(math.isfinite(z) for z in
                   (rho0,rho1,clipped0,clipped1,rep_fake,wall)):
            raise RuntimeError("STOP_R04_NONFINITE_METRICS")
        result={
            "schema":"prdiff.track_r.r0.4.candidate.v1",
            "status":"R0_4_CANDIDATE_PASS","uid":row["uid"],
            "source":row["source"],"family":row["family"],
            "image_id":row["image_id"],"run_index":row["run_index"],
            "executor_head":a.expected_head,"task_file_sha256":task_sha,
            "candidate_sha256":row["candidate_sha256"],
            "measurement_sha256":row["measurement_sha256"],
            "s_star":prior["s_star"],"S_repaired":prior["S"][prior["s_star"]],
            "S_raw":prior["S"]["000"],"prior":prior,
            "p_fake_orbit":dict(zip(ORBIT_PATTERNS,p_fake)),
            "p_fake_repaired":rep_fake,
            "p_fake_ufd_choice":ORBIT_PATTERNS[min(
                range(8),key=lambda j:(p_fake[j],j))],
            "rho_signed_raw":rho0,"rho_signed_repaired":rho1,
            "rho_clipped_raw":clipped0,"rho_clipped_repaired":clipped1,
            "unet_forwards":256,"univfd_forwards":9,
            "gpu_score_wall_seconds":wall,
            "gt_access":False
        }
        write_x(path,result)
        new+=1
        if (new+skipped)%8==0 or index+1==len(rows):
            print("R04_PROGRESS",a.start,"of",a.start+a.count,
                new+skipped,"/",len(rows),"GPU",a.physical_gpu,flush=True)
    receipt={"schema":"prdiff.track_r.r0.4.shard-complete.v1",
             "status":"R0_4_SHARD_PASS",
             "meta_sha256":sha(old),"task_file_sha256":task_sha,
             "start":a.start,"count":a.count,
             "new":new,"reused":skipped,
             "elapsed_seconds_this_attempt":time.monotonic()-start,
             "gpu_peak_reserved_mib":torch.cuda.max_memory_reserved()/1048576}
    for row in rows:
        validate_complete(result_root/output_name(row),row,task_sha,a.expected_head)
    write_x(complete,receipt)
    print("R04_SHARD_PASS",a.start,a.count,"NEW",new,"REUSED",skipped,
          "GPU_SEC",receipt["elapsed_seconds_this_attempt"],flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("out","tasks","tasks-sha","expected-head"):
        p.add_argument("--"+arg,required=True)
    for arg in ("physical-gpu","batch-size","start","count"):
        p.add_argument("--"+arg,required=True,type=int)
    p.add_argument("--tasks-source-head",default=None)
    a=p.parse_args()
    try:sys.exit(run(a))
    except Exception as exc:
        print("STOP_R04_WORKER",repr(exc),flush=True)
        traceback.print_exc()
        sys.exit(8)
