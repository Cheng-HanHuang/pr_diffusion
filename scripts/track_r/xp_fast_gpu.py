#!/usr/bin/env python3
"""Track R XP FAST-DIPS generator adapted from pinned R0.5 native sampler, GT-free.

One GPU; manifest-sliced roots, independently seeded. Strictly verifies
official FAST source, FFHQ checkpoint and signed-measurement SHA. Saves
one immutable source+output receipt per completed independent root.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
PATHS=WT/"configs/track_r/pac_paths_track_r_v1.json"
FROZEN=WT/"configs/track_r/r0_frozen_v1.json"
A15=WT/"configs/track_a/a15_fastdips_exposed_v1.json"
SOURCE_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"
CANARY_TASK_PATH=(BASE/"r0_5_canary_20261023T004927Z"/
                  "prepare"/"R0_5_GT_FREE_ROOTS.json")
CANARY_TASK_SHA="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
CANARY_HEAD="c3170587e58e9610c97c6f4ae48964eab2db1b46"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def source(path):
    p=Path(path)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink() or
        any(x in part for x in ("confirmation","confirm305") for part in
            (str(p).lower(),str(p.resolve(strict=False)).lower()))):
        raise RuntimeError("STOP_R05_PROTECTED_OR_MISSING_SOURCE "+str(p))
    return p

def git_head(path):
    return subprocess.check_output(
        ["git","-C",str(path),"rev-parse","HEAD"],
        text=True,timeout=30).strip()

def write_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def validate_native_conf(a15,frozen):
    native=a15["fast_native_pixel"]
    f=frozen["fast_generation"]
    for k in ("sampler","T","K_ADMM","S_x","rho","epsilon",
              "backtrack","alpha_max"):
        if native[k]!=f[k]:
            raise RuntimeError("STOP_R05_FAST_CONFIG_DRIFT "+k)
    if f["roots_per_image"]!=4 or f["sources"]!=["S1_b22","S2_b24_dev80"]:
        raise RuntimeError("STOP_R05_SOURCE_CONFIG_DRIFT")
    return native

def frozen_roots(payload,head):
    if (payload.get("schema")!="prdiff.track_r.xp.fast-tasks.v1"
        or payload.get("status")!="XP_FAST_TASKS_FROZEN"
        or payload.get("head")!=head or
        payload.get("fast_head")!=SOURCE_HEAD or
        payload.get("n_roots")!=len(payload.get("rows",[])) or
        payload.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_XP_FAST_TASKS_DRIFT")
    rows=payload["rows"]
    fields={"uid","source","measurement_source","image_id","rep","seed",
            "measurement_path","measurement_sha256"}
    if not rows or any(set(x)!=fields for x in rows):
        raise RuntimeError("STOP_XP_FAST_ROWS_SCHEMA")
    if len({r["uid"] for r in rows})!=len(rows):
        raise RuntimeError("STOP_XP_FAST_DUPLICATE_UID")
    return rows

def root_folder(root,row):
    token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
    return Path(root)/("root_"+token)

def validate_complete(folder,row,head,rootsha):
    data=json.loads(source(folder/"R0_FAST_ROOT_COMPLETE.json").read_text())
    if (data.get("status")!="R0_FAST_ROOT_PASS" or
        data.get("uid")!=row["uid"] or
        data.get("seed")!=row["seed"] or
        data.get("source_head")!=head or
        data.get("frozen_root_sha256")!=rootsha or
        data.get("measurement_sha256")!=row["measurement_sha256"] or
        data.get("gt_access") is not False):
        raise RuntimeError("STOP_R05_ROOT_RECEIPT_NOT_MATCHED "+row["uid"])
    for k,f in (("reconstruction_pt_sha256","reconstruction.pt"),
                ("reconstruction_png_sha256","reconstruction.png")):
        if sha(source(folder/f))!=data[k]:
            raise RuntimeError("STOP_R05_ROOT_RESULT_SHA "+row["uid"])
    return data

def preflight(a):
    if Path(sys.executable).resolve()!=FAST_PY.resolve():
        raise RuntimeError("STOP_XP_FAST_ENV")
    if (git_head(WT)!=a.expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                text=True,timeout=30)):
        raise RuntimeError("STOP_XP_FAST_GIT")
    if (os.environ.get("PYTHONDONTWRITEBYTECODE")!="1" or
        os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.physical_gpu) or
        a.physical_gpu not in range(4)):
        raise RuntimeError("STOP_XP_FAST_GPU_VISIBILITY")
    out=Path(a.out)
    if (not str(out).startswith(str(BASE/"xp_")) or out.is_symlink() or
        (out.exists() and not a.resume) or
        (a.resume and not out.is_dir())):
        raise RuntimeError("STOP_XP_FAST_OUTPUT")
    roots=source(a.roots)
    if sha(roots)!=a.roots_sha:
        raise RuntimeError("STOP_XP_FAST_MANIFEST_SHA")
    manifest=json.loads(roots.read_text())
    if manifest.get("dataset")!=a.dataset or float(manifest.get("oversample",-1))!=a.oversample:
        raise RuntimeError("STOP_XP_FAST_OPERATOR_SETTINGS")
    producer=a.roots_source_head or a.expected_head
    rows=frozen_roots(manifest,producer)
    if producer!=a.expected_head:
        if (manifest.get("stage")=="D1_P2" and a.dataset=="imagenet"):
            from scripts.track_r.xp_a6_d1p2_contract import validate
            validate(roots,a.roots_sha,manifest,rows,
                     producer,a.start,a.count)
        else:
            from scripts.track_r.xp_a16_contract import validate_a16_source
            validate_a16_source(roots,a.roots_sha,manifest,rows,
                                producer,a.start,a.count)
    if a.start<0 or not 1<=a.count<=120 or a.start+a.count>len(rows):
        raise RuntimeError("STOP_XP_FAST_SLICE")
    chosen=rows[a.start:a.start+a.count]
    paths=json.loads(PATHS.read_text())
    frozen=json.loads(FROZEN.read_text())
    conf=validate_native_conf(json.loads(A15.read_text()),frozen)
    fast=Path(paths["external"]["fast_dips"]["path"])
    if (git_head(fast)!=SOURCE_HEAD or
        subprocess.check_output(["git","-C",str(fast),"status","--porcelain",
                                 "--untracked-files=no"],text=True,timeout=30)):
        raise RuntimeError("STOP_XP_FAST_SOURCE")
    model=source(ROOT/"models"/("imagenet256.pt" if a.dataset=="imagenet" else "ffhq_10m.pt"))
    model_sha=sha(model)
    if a.dataset=="ffhq" and model_sha!=paths["models"]["ffhq_ddpm"]["sha256"]:
        raise RuntimeError("STOP_XP_FAST_FFHQ_CHECKPOINT")
    if a.dataset=="imagenet" and model_sha!=manifest.get("model_sha256"):
        raise RuntimeError("STOP_XP_FAST_IMAGENET_CHECKPOINT")
    for r in chosen:
        if sha(source(r["measurement_path"]))!=r["measurement_sha256"]:
            raise RuntimeError("STOP_XP_FAST_Y_SHA "+r["uid"])
    return chosen,paths,fast,conf,model,model_sha

def execute(a):
    chosen,paths,fast,conf,model_path,model_sha=preflight(a)
    import numpy as np
    import torch
    from PIL import Image
    from scripts.track_a.a15_fastdips_smoke import load_checkpoint_verified
    from scripts.track_a.a15_fastdips_preflight import load_tensor
    from scripts.track_a.a15_fastdips_preflight import tensor_hash
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP_R05_EXPECT_SINGLE_CUDA")
    prop=torch.cuda.get_device_properties(0)
    if (prop.major,prop.minor)!=(12,0):
        raise RuntimeError("STOP_R05_EXPECT_BLACKWELL")
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(fast))
    import main as upstream
    import misc as settings
    from forward_operators import get_operator
    from model import get_model
    if Path(upstream.__file__).resolve()!=fast/"main.py":
        raise RuntimeError("STOP_R05_FAST_IMPORT_ORIGIN")
    device=torch.device("cuda:0")
    # Build ONCE. Reset RNG with the frozen independent seed for every root.
    random.seed(20261022)
    np.random.seed(20261022)
    torch.manual_seed(20261022)
    ddpm=copy.deepcopy(settings.MODEL_PRESETS["imagenet256ddpm" if a.dataset=="imagenet" else "ffhq256ddpm"])
    if ddpm["name"]!="ddpm":raise RuntimeError("STOP_R05_NOT_PIXEL_DDPM")
    ddpm["model_config"]["model_path"]=str(model_path)
    network=get_model(**ddpm,device="cuda:0",requires_grad=False)
    load_checkpoint_verified(network,model_path,
        model_sha)
    network.eval()
    dest=Path(a.out)
    meta={"schema":"prdiff.track_r.r0.5.fast-chunk.v1",
          "status":"R0_5_CHUNK_CONFIG","head":a.expected_head,
          "root_sha256":a.roots_sha,
          "start":a.start,"count":a.count,
          "gpu":a.physical_gpu,
          "uids":[r["uid"] for r in chosen]}
    if a.resume:
        prior=json.loads(source(dest/"R0_5_CHUNK_META.json").read_text())
        if prior!=meta:raise RuntimeError("STOP_R05_RESUME_DIFFERENT_CHUNK")
    else:
        dest.mkdir(parents=True,exist_ok=False)
        write_x(dest/"R0_5_CHUNK_META.json",meta)
    complete_file=dest/"R0_5_CHUNK_COMPLETE.json"
    if complete_file.exists():
        previous=json.loads(source(complete_file).read_text())
        if (previous.get("status")!="R0_5_CHUNK_PASS" or
            previous.get("meta_sha256")!=sha(dest/"R0_5_CHUNK_META.json")):
            raise RuntimeError("STOP_R05_COMPLETE_CHUNK_RECEIPT")
        for row in chosen:
            validate_complete(root_folder(dest,row),row,
                              a.expected_head,a.roots_sha)
        print("R05_CHUNK_ALREADY_PASS",a.start,a.count,flush=True)
        return 0
    t0=time.monotonic()
    new=0
    reused=0
    by_image={}
    for row in chosen:
        key=(row["measurement_source"],row["image_id"])
        if key not in by_image:
            observation=load_tensor(
                source(row["measurement_path"]),("measurement","y","observation"),
                (1,3,256+64*int(a.oversample),256+64*int(a.oversample)))
            # Check file and content identity; CPU no GT.
            if not bool(torch.isfinite(observation).all()):
                raise RuntimeError("STOP_R05_OBSERVATION_NONFINITE")
            by_image[key]=observation
    for row in chosen:
        folder=root_folder(dest,row)
        if folder.exists():
            validate_complete(folder,row,a.expected_head,a.roots_sha)
            reused+=1
            continue
        seed=int(row["seed"])
        random.seed(seed)
        np.random.seed(seed%(2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        preset=copy.deepcopy(settings.SAMPLER_PRESETS[conf["sampler"]])
        if preset.pop("latent") is not False:
            raise RuntimeError("STOP_R05_FAST_LATENT_PRESET")
        preset["diffusion_scheduler_config"]["num_steps"]=conf["T"]
        sampler=upstream.FAST_DIPS(
            diffusion_scheduler_config=preset["diffusion_scheduler_config"],
            K=conf["K_ADMM"],S=conf["S_x"],rho=conf["rho"],
            epsilon=conf["epsilon"],backtrack=conf["backtrack"],
            alpha_max=conf["alpha_max"])
        operator=get_operator(name="phase_retrieval",oversample=a.oversample,
                              resolution=256,sigma=0.05)
        y=by_image[row["measurement_source"],row["image_id"]].to(device)
        x0=sampler.get_start(1,network)
        torch.cuda.synchronize(device)
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        wall_start=time.monotonic()
        start.record()
        # Native FAST needs gradients through its measurement correction;
        # NEVER put sampler.sample inside torch.no_grad/inference_mode.
        output=sampler.sample(
            network,x0,operator,y,evaluator=None,
            record=False,verbose=False)
        end.record()
        torch.cuda.synchronize(device)
        elapsed=time.monotonic()-wall_start
        cuda_s=float(start.elapsed_time(end)/1000)
        native_s=float(sampler.last_sampling_time)
        if (tuple(output.shape)!=(1,3,256,256) or
            not bool(torch.isfinite(output).all()) or
            not (0<native_s<=elapsed*1.1 and 0<cuda_s<=elapsed*1.1)):
            raise RuntimeError("STOP_R05_NATIVE_OUTPUT_OR_TIMING "+row["uid"])
        cpu=output.detach().float().cpu()
        im=torch.round((cpu[0].clamp(-1,1)+1)*127.5).to(torch.uint8)
        folder.mkdir(parents=True,exist_ok=False)
        torch.save({"reconstruction":cpu},folder/"reconstruction.pt")
        Image.fromarray(im.permute(1,2,0).numpy(),"RGB").save(
            folder/"reconstruction.png")
        receipt={
            "schema":"prdiff.track_r.r0.5.fast-root.v1",
            "status":"R0_FAST_ROOT_PASS",
            "uid":row["uid"],"source_head":a.expected_head,
            "fast_source_head":SOURCE_HEAD,
            "measurement_source":row["measurement_source"],
            "image_id":row["image_id"],"rep":row["rep"],
            "seed":seed,"frozen_root_sha256":a.roots_sha,
            "measurement_sha256":row["measurement_sha256"],
            "reconstruction_pt_sha256":sha(folder/"reconstruction.pt"),
            "reconstruction_png_sha256":sha(folder/"reconstruction.png"),
            "native_sample_wall_seconds":elapsed,
            "sample_cuda_event_seconds":cuda_s,
            "native_step_sum_seconds":native_s,
            "gt_access":False,
            "method":"UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5",
            "dataset":a.dataset,"oversample":a.oversample}
        write_x(folder/"R0_FAST_ROOT_COMPLETE.json",receipt)
        new+=1
        print("R05_ROOT_PASS",a.start,"uid",row["uid"],
              "seconds",round(elapsed,3),flush=True)
        del sampler,operator,x0,output,cpu,im,y
    for row in chosen:
        validate_complete(root_folder(dest,row),row,a.expected_head,a.roots_sha)
    complete={
        "schema":"prdiff.track_r.r0.5.fast-chunk-complete.v1",
        "status":"R0_5_CHUNK_PASS",
        "start":a.start,"count":a.count,
        "new":new,"reused":reused,
        "head":a.expected_head,
        "meta_sha256":sha(dest/"R0_5_CHUNK_META.json"),
        "elapsed_gpu_allocation_seconds":time.monotonic()-t0}
    write_x(complete_file,complete)
    print("R05_CHUNK_PASS",a.start,a.count,"NEW",new,"REUSED",reused,
          "SECONDS",round(complete["elapsed_gpu_allocation_seconds"],3),flush=True)
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for k in ("expected-head","roots","roots-sha","out"):
        parser.add_argument("--"+k,required=True)
    for k in ("start","count","physical-gpu"):
        parser.add_argument("--"+k,type=int,required=True)
    parser.add_argument("--dataset",choices=("ffhq","imagenet"),default="ffhq")
    parser.add_argument("--oversample",type=float,choices=(2.0,4.0),default=2.0)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--roots-source-head",default=None)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    try:
        if args.preflight_only:
            selected,paths,fast,conf,model,modelsha=preflight(args)
            print("XP_FAST_PREFLIGHT_PASS roots="+str(len(selected))+
                  " dataset="+args.dataset+" canvas="+
                  str(256+64*int(args.oversample))+" head="+args.expected_head)
            sys.exit(0)
        sys.exit(execute(args))
    except Exception as err:
        print("STOP_R05_FAST_WORKER",repr(err),flush=True)
        traceback.print_exc()
        sys.exit(8)
