#!/usr/bin/env python3
"""Track R R0.3a: one-candidate G1 canonical GPU sanity gate.

No S1–S4 full scoring, no FAST generation, no completion claim for G0.
Use exactly one user-selected idle physical GPU. This is an independent,
fail-closed evidence gate before the remaining GPU work.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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
    with open(path,"rb") as file:
        for chunk in iter(lambda:file.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write_x(path, obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


def safe_file(path):
    p=Path(path)
    if not p.is_file() or any(
        s in str(p).lower() or s in str(p.resolve()).lower()
        for s in ("confirmation","confirm305")
    ):
        raise RuntimeError("STOP_G1_PROTECTED_OR_MISSING_INPUT "+str(p))
    return p


def preflight(args):
    if args.physical_gpu not in range(4):
        raise RuntimeError("STOP_G1_BAD_GPU_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("STOP_G1_PHYSICAL_GPU_BINDING")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("STOP_G1_NO_BYTECODE_GUARD")
    out=Path(args.out)
    if not str(out).startswith(str(BASE)+os.sep) or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_G1_OUTPUT_OCCUPIED_OR_OUTSIDE_TRACK_R")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if head!=args.expected_head or dirty:
        raise RuntimeError("STOP_G1_GIT_IDENTITY "+head)
    gpu_rows=subprocess.check_output([
        "nvidia-smi","--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"
    ],text=True,timeout=20).splitlines()
    usage={}
    for row in gpu_rows:
        pieces=[x.strip() for x in row.split(",")]
        if len(pieces)!=2:raise RuntimeError("STOP_G1_GPU_MONITOR_FORMAT")
        usage[int(pieces[0])]=int(pieces[1])
    if args.physical_gpu not in usage or usage[args.physical_gpu]>1024:
        raise RuntimeError("STOP_G1_GPU_NOT_IDLE "+str(usage))
    mr=Path(args.manifest)
    gr=Path(args.g0_receipt)
    if not mr.is_file() or not gr.is_file():
        raise RuntimeError("STOP_G1_MISSING_FROZEN_CPU_RECEIPTS")
    g0=json.loads(gr.read_text())
    if g0.get("status")!="G0_CPU_PRECHECK_PASS_PENDING_DETERMINISM":
        raise RuntimeError("STOP_G1_G0_CPU_NOT_PASS")
    if g0["git_head"]!=args.previous_head and g0["git_head"]!=args.expected_head:
        raise RuntimeError("STOP_G1_G0_HEAD_NOT_RECOGNIZED")
    for name in ("G0_fft_witness","G0_orbit_invariance",
                 "G0_operator_pairing","G0_recorded_psnr_replay",
                 "G0_protected_path"):
        if g0["gates"][name]["status"]!="PASS":
            raise RuntimeError("STOP_G1_G0_COMPONENT "+name)
    if g0["gates"]["G0_recorded_psnr_replay"]["n_total"]!=2520:
        raise RuntimeError("STOP_G1_G0_REPLAY_UNIVERSE")
    manifest_receipt=mr.parent/"R0_2_MANIFEST_RECEIPT.json"
    manifest_receipt=safe_file(manifest_receipt)
    receipt=json.loads(manifest_receipt.read_text())
    if (receipt.get("n_candidates")!=2520 or
            receipt.get("manifest_sha256")!=sha(mr) or
            g0.get("manifest_path")!=str(mr)):
        raise RuntimeError("STOP_G1_MANIFEST_IDENTITY")
    paths=json.loads((WT/"configs/track_r/pac_paths_track_r_v1.json").read_text())
    ckpt=paths["models"]["ffhq_ddpm"]
    f=safe_file(ckpt["path"])
    if sha(f)!=ckpt["sha256"]:
        raise RuntimeError("STOP_G1_DDPM_CHECKPOINT_SHA")
    external=paths["external"]["difffpr"]
    root=Path(external["path"])
    if subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                               text=True).strip()!=external["commit"]:
        raise RuntimeError("STOP_G1_DIFFFPR_IDENTITY")
    return head,paths,usage,sha(mr)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--g0-receipt",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--previous-head",default="")
    ap.add_argument("--physical-gpu",type=int,required=True)
    args=ap.parse_args()
    head,paths,usage,manifest_sha=preflight(args)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    write_x(out/"RUNNING.json",{
        "status":"NOT_A_PASS", "phase":"R0_3A_G1",
        "head":head, "manifest_sha256":manifest_sha,
        "physical_gpu":args.physical_gpu,
        "initial_gpu_memory_mib":usage[args.physical_gpu],
        "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    receipt={"schema":"prdiff.track_r.g1_gpu.v1","status":"G1_FAIL",
             "source_head":head,"physical_gpu":args.physical_gpu,
             "manifest_sha256":manifest_sha,
             "not_full_g0":True,"full_r0_gpu_budget_used_hours":None,
             "s5_fast_runs_generated":0}
    try:
        import torch
        from scripts.track_r.r0_2_g0_cpu import candidate_source,gt_source
        from prdiffusion.track_r.manifest import read_manifest,uid_for
        from prdiffusion.track_r.physics import (ORBIT_PATTERNS,to_rgb8,
            psnr_rgb8,apply_orbit,from_rgb8)
        from prdiffusion.track_r.prior_score import (
            PriorScorer,frozen_alpha_schedule)
        from prdiffusion.guided_backend import load_guided_diffusion_model

        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("STOP_G1_DEVICE_NOT_SINGLE_CUDA")
        device=torch.device("cuda:0")
        props=torch.cuda.get_device_properties(device)
        if (props.major,props.minor)!=(12,0):
            raise RuntimeError("STOP_G1_GPU_ARCH_NOT_BLACKWELL")
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20261022)
        torch.cuda.manual_seed_all(20261022)

        rows=read_manifest(safe_file(args.manifest))
        uid=uid_for("S3_a12","DAPS","68539","rep2")
        chosen=[r for r in rows if r["uid"]==uid]
        if len(chosen)!=1:
            raise RuntimeError("STOP_G1_CANONICAL_CANDIDATE_NOT_UNIQUE")
        row=chosen[0]
        candidate=candidate_source(row)
        gt=gt_source(row)
        image8=to_rgb8(candidate)
        gt8=to_rgb8(gt)
        orbit_db={
            p:float(psnr_rgb8(apply_orbit(image8,p),gt8)[0])
            for p in ORBIT_PATTERNS}
        receipt["canonical_gt_orbit_psnr_db"]=orbit_db
        if not (abs(orbit_db["000"]-6.364)<=0.025 and
                abs(orbit_db["111"]-31.871)<=0.025):
            raise RuntimeError("STOP_G1_CANONICAL_SOURCE_PSNR_MISMATCH")
        model_file=paths["models"]["ffhq_ddpm"]["path"]
        print("G1_LOAD_GUIDED_MODEL",model_file,flush=True)
        loaded_start=time.monotonic()
        bundle=load_guided_diffusion_model(
            model_path=model_file,device=device,
            preset="difffpr_ffhq_10m",
            guided_diffusion_dir=paths["external"]["difffpr"]["path"],
            strict=True)
        receipt["model_load_wall_seconds"]=time.monotonic()-loaded_start
        expected_alphas=frozen_alpha_schedule()
        actual=bundle.scheduler.alphas_cumprod.detach().cpu()
        alpha_err=float((expected_alphas-actual).abs().max())
        receipt["max_diff_backend_schedule"]=alpha_err
        if len(actual)!=1000 or alpha_err>1e-6:
            raise RuntimeError("STOP_G1_DIFFUSION_SCHEDULE_DRIFT")
        scorer=PriorScorer(
            eps_model=lambda xt,t: bundle.unet(xt,t).sample,
            timesteps=(50,100,200,300,400,500,600,700),
            draws=4,device=device,batch_size=8)
        torch.cuda.synchronize()
        start=time.monotonic()
        result=scorer.score_orbit(from_rgb8(image8),uid)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        receipt["prior"]=result
        receipt["wall_score_seconds"]=elapsed
        receipt["measured_unet_forwards"]=result["unet_forwards"]
        receipt["repaired_psnr_db"]=orbit_db[result["s_star"]]
        receipt["G1_pass"]=bool(receipt["repaired_psnr_db"]>=25.0)
        receipt["status"]="G1_PASS" if receipt["G1_pass"] else "G1_FAIL"
        receipt["gpu_peak_reserved_mib"]=(torch.cuda.max_memory_reserved()/1048576)
        receipt["only_one_canonical_candidate_scored"]=True
        print("G1_RESULT",json.dumps({
            "status":receipt["status"],"uid":uid,
            "s_star":result["s_star"],
            "raw_psnr":orbit_db["000"],
            "repaired_psnr":receipt["repaired_psnr_db"],
            "wall_seconds":elapsed,
            "batch_calls":result["unet_batch_calls"]},
            sort_keys=True),flush=True)
        write_x(out/"R0_3A_G1_RECEIPT.json",receipt)
        return 0 if receipt["G1_pass"] else 7
    except Exception as e:
        receipt["status"]="G1_EXCEPTION_STOP"
        receipt["error"]=repr(e)
        receipt["traceback"]=traceback.format_exc()[-2500:]
        write_x(out/"R0_3A_G1_RECEIPT.json",receipt)
        print("STOP_G1",repr(e),flush=True)
        return 8


if __name__=="__main__":
    sys.exit(main())
