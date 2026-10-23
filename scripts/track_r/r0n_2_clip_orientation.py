#!/usr/bin/env python3
"""R0-N N2 frozen CLIP orientation: GT-free candidate manifest + offline eval.

Modes: prepare (CPU, only candidate hashes from immutable R0.7 catalog),
score (GPU, no GT inputs), evaluate (CPU, uses frozen R0.7 GT-derived PSNRs).
No new reconstructions or measurements. No prompts or threshold tuning.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from prdiffusion.track_r.r0_7_stats import clustered_ratio_ci

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z"
CFG=WT/"configs/track_r/r0n_letter_analysis_v1.json"
PAC_CFG=WT/"configs/track_r/pac_paths_track_r_v1.json"
CLIP_SHA="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
R07_HEAD="c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b"
PROMPTS_UP=("a photo of a face.","a photo of a person's face.","a portrait photo of a person.")
PROMPTS_INV=("an upside-down photo of a face.",
  "an upside-down photo of a person's face.",
  "an upside-down portrait photo of a person.")
FORBIDDEN=("confirmation","confirm305")
CEILING=900.
SMOKE=64
BATCH=32

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
    return h.hexdigest()

def valid_file(p):
    p=Path(p)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink() or
        any(v in str(p.resolve()).lower() for v in FORBIDDEN)):
        raise RuntimeError("STOP_N2_MISSING_OR_PROTECTED_SOURCE "+str(p))
    return p

def load(p):
    return json.loads(valid_file(p).read_text(encoding="utf-8"))

def write_x(path,payload):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def frozen_config():
    config=load(CFG)
    n=config["n2_clip_zero_shot"]
    if (n["gpu_hours_ceiling"]!=.25 or n["cpu_allowed"] is not True or
        n["prompts_upright"]!=list(PROMPTS_UP) or
        n["prompts_inverted"]!=list(PROMPTS_INV) or
        n["decision_global"]!="choose 111 iff u(R_111 x) > u(x) strictly; else 000" or
        n["no_prompt_tuning"] is not True or
        n["gt_access_in_scoring"] is not False):
        raise RuntimeError("STOP_N2_FROZEN_CONFIG")
    return config

def exact_head(expected):
    if (subprocess.check_output(
          ["git","-C",str(WT),"rev-parse","HEAD"],
          text=True,timeout=30).strip()!=expected or
        subprocess.check_output(
          ["git","-C",str(WT),"status","--porcelain"],
          text=True,timeout=30)):
        raise RuntimeError("STOP_N2_HEAD_OR_DIRTY")

def stage_prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_N2_PREP_GPU_VISIBLE")
    frozen_config();exact_head(args.expected_head)
    out=Path(args.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_N2_OUTPUT")
    catalog=load(R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json")
    receipt=load(R07/"analysis/R0_7_COMPLETE_RECEIPT.json")
    if (catalog.get("head")!=R07_HEAD or
        len(catalog.get("rows",[]))!=3240 or
        receipt.get("status")!="R0_7_ANALYSIS_COMPLETE" or
        receipt.get("n_scored")!=3240):
        raise RuntimeError("STOP_N2_R07_INCOMPLETE")
    # Stage canonical PNG copies of the *same saved reconstructions*.
    # N2 model scoring must read PNGs only, never historical .pt or
    # any GT/measurement file. This conversion runs on CPU before GPU.
    from scripts.track_r.r0_4_score_gpu import candidate_tensor
    from prdiffusion.track_r.physics import load_png_rgb8
    from PIL import Image
    import torch
    names=("uid","source","family","image_id","run_index",
           "candidate_path","candidate_format","candidate_sha256")
    originals=[{name:r[name] for name in names} for r in catalog["rows"]]
    if len({r["uid"] for r in originals})!=3240:
        raise RuntimeError("STOP_N2_UID_COLLISION")
    out.mkdir(parents=True,exist_ok=False)
    folder=out/"canonical_png"
    folder.mkdir()
    rows=[]
    for i,r in enumerate(originals):
        if set(r)!=set(names):
            raise RuntimeError("STOP_N2_GT_FREE_PREP")
        valid_file(r["candidate_path"])
        if (len(r["candidate_sha256"])!=64 or
            any(c not in "0123456789abcdef" for c in r["candidate_sha256"])):
            raise RuntimeError("STOP_N2_HASH_SCHEMA")
        x8=candidate_tensor(r)
        if x8.dtype!=torch.uint8 or x8.shape!=(1,3,256,256):
            raise RuntimeError("STOP_N2_CANONICAL_RGB8")
        path=folder/(hashlib.sha256(r["uid"].encode()).hexdigest()+".png")
        image=x8[0].permute(1,2,0).numpy()
        Image.fromarray(image,"RGB").save(path,format="PNG")
        if not torch.equal(load_png_rgb8(path),x8):
            raise RuntimeError("STOP_N2_PNG_ROUNDTRIP "+r["uid"])
        rows.append({"uid":r["uid"],"source":r["source"],
           "family":r["family"],"image_id":r["image_id"],
           "run_index":r["run_index"],"png_path":str(path),
           "png_sha256":sha(path)})
        if (i+1)%500==0:
            print("N2_CPU_CANONICAL_PNG",i+1,"/3240",flush=True)
    task=out/"N2_GT_FREE_CANDIDATES.json"
    write_x(task,{"status":"R0N_N2_GT_FREE_CANDIDATES_FROZEN",
        "executor_head":args.expected_head,
        "r07_catalog_sha256":sha(R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json"),
        "count":3240,"no_gt":True,"tasks":rows})
    write_x(out/"N2_PREP_RECEIPT.json",{
        "status":"R0N_N2_PREP_PASS","tasks_sha256":sha(task),
        "n_candidates":3240,"executor_head":args.expected_head,
        "r07_analysis_sha256":receipt["analysis_sha256"],
        "gt_paths_in_gpu_tasks":False,
        "gpu_reads_canonical_png_only":True})
    print("N2_GT_FREE_PREP_PASS",len(rows),sha(task),flush=True)

def check_gpu(device):
    if not device in (0,1,2,3) or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(device):
        raise RuntimeError("STOP_N2_PHYSICAL_GPU_BINDING")
    usage=subprocess.check_output(
       ["nvidia-smi","--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"],text=True,timeout=30)
    gpu_mem={}
    for row in usage.splitlines():
        a,b=[x.strip() for x in row.split(",")]
        gpu_mem[int(a)]=int(b)
    if gpu_mem.get(device,99999)>1024:
        raise RuntimeError("STOP_N2_FOREIGN_GPU_ACTIVITY "+str(gpu_mem))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP_N2_CUDA_NOT_SINGLE_VISIBLE_DEVICE")

def clip_score(args):
    import importlib
    import torch
    from PIL import Image
    from prdiffusion.track_r.physics import load_png_rgb8
    frozen_config();exact_head(args.expected_head)
    if args.budget_seconds>CEILING or args.budget_seconds<=0:
        raise RuntimeError("STOP_N2_GPU_BUDGET")
    check_gpu(args.physical_gpu)
    start=time.monotonic()
    tasks_file=valid_file(args.tasks)
    if sha(tasks_file)!=args.tasks_sha:
        raise RuntimeError("STOP_N2_TASKS_SHA")
    payload=load(tasks_file)
    rows=payload["tasks"]
    if (payload.get("status")!="R0N_N2_GT_FREE_CANDIDATES_FROZEN"
       or payload.get("executor_head")!=args.expected_head
       or payload.get("no_gt") is not True or
       payload.get("count")!=3240 or len(rows)!=3240):
        raise RuntimeError("STOP_N2_TASKS")
    allowed={"uid","source","family","image_id","run_index",
             "png_path","png_sha256"}
    if any(set(r)!=allowed for r in rows):
        raise RuntimeError("STOP_N2_GT_LEAK_IN_TASKS")
    source=load(PAC_CFG)
    clip_path=valid_file(source["models"]["clip_vit_l14"]["path"])
    if sha(clip_path)!=CLIP_SHA:
        raise RuntimeError("STOP_N2_CLIP_CHECKPOINT_SHA")
    u_root=Path(source["external"]["univfd"]["path"])
    if u_root.is_symlink() or not (u_root/"models/clip").is_dir():
        raise RuntimeError("STOP_N2_UNIVFD_EXTERNAL_PATH")
    observed_head=subprocess.check_output(
        ["git","-C",str(u_root),"rev-parse","HEAD"],text=True,timeout=30).strip()
    if observed_head!=source["external"]["univfd"]["commit"]:
        raise RuntimeError("STOP_N2_UNIVFD_SOURCE_COMMIT_CHANGED")
    if str(u_root) not in sys.path:sys.path.insert(0,str(u_root))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clip_path),device="cuda:0",jit=False)
    model=model.eval()
    prompts=list(PROMPTS_UP+PROMPTS_INV)
    with torch.inference_mode():
        tokens=clip.tokenize(prompts).to("cuda:0")
        encoded=model.encode_text(tokens)
        encoded=encoded.float()
        encoded=encoded/encoded.norm(dim=-1,keepdim=True)
        u=encoded[:3].mean(dim=0)
        d=encoded[3:].mean(dim=0)
        u=u/u.norm();d=d/d.norm()
        results=[]
        smoken=0
        model_ready=time.monotonic()
        init_seconds=model_ready-start
        for i in range(0,len(rows),BATCH):
            if time.monotonic()-start>=args.budget_seconds-10:
                raise RuntimeError("STOP_N2_GPU_BUDGET_NEAR_EXHAUSTION")
            subset=rows[i:i+BATCH]
            arr=[]
            for r in subset:
                file=valid_file(r["png_path"])
                if file.parent!=tasks_file.parent/"canonical_png" or sha(file)!=r["png_sha256"]:
                    raise RuntimeError("STOP_N2_CANONICAL_PNG_SHA")
                x8=load_png_rgb8(file)
                x=x8[0].permute(1,2,0).numpy()
                upright=Image.fromarray(x,"RGB")
                flipped=Image.fromarray(np.rot90(x,2),"RGB")
                arr.extend([preprocess(upright),preprocess(flipped)])
            x=torch.stack(arr).to("cuda:0")
            with torch.inference_mode():
                enc=model.encode_image(x)
                enc=enc.float()
                enc=enc/enc.norm(dim=-1,keepdim=True)
                score=(enc@u)-(enc@d)
                values=score.cpu().tolist()
            for k,r in enumerate(subset):
                s0=float(values[2*k]);s1=float(values[2*k+1])
                if not all(map(math.isfinite,(s0,s1))):
                    raise RuntimeError("STOP_N2_CLIP_NONFINITE")
                results.append({
                    "uid":r["uid"],"u_identity":s0,"u_global180":s1,
                    "choice":"111" if s1>s0 else "000"})
            smoken+=len(subset)
            # Projection comes after 64 first candidates, excludes one-time
            # model load from per-candidate cost and preserves a reserve.
            if smoken==SMOKE:
                processing=time.monotonic()-model_ready
                # Model/text loading is one-time overhead, not per-batch.
                # Include a 1.5x batch projection and 60-second reserve.
                projected=init_seconds+processing*len(rows)/SMOKE*1.5+60.
                print("N2_MODEL_LOAD_SECONDS",round(init_seconds,3),
                      "SMOKE64_PROCESSING_SECONDS",round(processing,3),
                      "CONSERVATIVE_PROJECTION_SECONDS",round(projected,3),
                      "BUDGET",args.budget_seconds,flush=True)
                if projected>args.budget_seconds:
                    raise RuntimeError("STOP_N2_PROJECTED_BUDGET_EXCESS")
            if (i//BATCH)%10==0:
                print("N2_GT_FREE_CLIP_PROGRESS",len(results),"/3240",
                      "GPU_S",round(time.monotonic()-start,1),flush=True)
        torch.cuda.synchronize()
    gpu_seconds=time.monotonic()-start
    if gpu_seconds>args.budget_seconds:
        raise RuntimeError("STOP_N2_GPU_CEILING_REACHED")
    out=Path(args.out)
    if out.exists() or not str(out).startswith(str(BASE)+"/"):
        raise RuntimeError("STOP_N2_NEW_OUTPUT_ONLY")
    out.mkdir(parents=True,exist_ok=False)
    scored=out/"N2_GT_FREE_CLIP_SCORES.json"
    write_x(scored,{"status":"R0N_N2_ALL_3240_GT_FREE_CLIP_PASS",
      "head":args.expected_head,"tasks_sha256":args.tasks_sha,
      "prompts_up":list(PROMPTS_UP),"prompts_inverted":list(PROMPTS_INV),
      "scoring_gt_access":False,"n":3240,"rows":results})
    write_x(out/"N2_CLIP_GPU_RECEIPT.json",{
      "status":"R0N_N2_CLIP_GPU_PASS","n":3240,
      "scored_sha256":sha(scored),"head":args.expected_head,
      "gpu_seconds":gpu_seconds,"gpu_hours":gpu_seconds/3600,
      "physical_gpu":args.physical_gpu,"gpu_ceiling_seconds":args.budget_seconds,
      "gt_access":False})
    print("R0N_N2_CLIP_GPU_PASS 3240 GPU_SECONDS",round(gpu_seconds,2),flush=True)

def eval_scores(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_N2_EVAL_GPU_VISIBLE")
    frozen_config();exact_head(args.expected_head)
    clipfile=valid_file(args.scores)
    meta=load(clipfile.parent/"N2_CLIP_GPU_RECEIPT.json")
    score=load(clipfile)
    if (meta.get("status")!="R0N_N2_CLIP_GPU_PASS" or
        sha(clipfile)!=meta["scored_sha256"] or
        meta["gpu_seconds"]>CEILING or
        score.get("head")!=args.expected_head or
        score.get("status")!="R0N_N2_ALL_3240_GT_FREE_CLIP_PASS" or
        len(score["rows"])!=3240 or
        score.get("scoring_gt_access") is not False):
        raise RuntimeError("STOP_N2_CLIP_RECEIPT")
    clipchoices={r["uid"]:r for r in score["rows"]}
    frozen=[]
    with valid_file(R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl").open() as f:
        for line in f:frozen.append(json.loads(line))
    if len(frozen)!=3240 or set(clipchoices)!={r["uid"] for r in frozen}:
        raise RuntimeError("STOP_N2_R07_UIDS")
    prior_catalog=load(R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json")
    source={r["uid"]:r for r in prior_catalog["rows"]}
    results=[]
    methods=("CLIP","PRIOR_GLOBAL","PRIOR_MID_GLOBAL","UNIVFD_GLOBAL")
    for row in frozen:
        uid=row["uid"]
        receipt=valid_file(source[uid]["score_receipt_path"])
        if sha(receipt)!=source[uid]["score_receipt_sha256"]:
            raise RuntimeError("STOP_N2_SCORE_SOURCE_DRIFT "+uid)
        r=load(receipt)
        s=r["prior"]["S"];smid=r["prior"]["S_mid"];p=r["p_fake_orbit"]
        choices={
          "CLIP":clipchoices[uid]["choice"],
          "PRIOR_GLOBAL":"111" if s["111"]<s["000"]-1e-7 else "000",
          "PRIOR_MID_GLOBAL":"111" if smid["111"]<smid["000"]-1e-7 else "000",
          "UNIVFD_GLOBAL":"111" if p["111"]<p["000"] else "000"}
        results.append({
           "uid":uid,"image_id":row["image_id"],
           "source":row["source"],"family":row["family"],
           "raw_label":row["raw_label"],
           "s_gt":row["s_gt"],
           "psnr_raw":row["psnr_raw"],
           "psnr_by_orbit":{"000":row["psnr_by_orbit"]["000"],
                            "111":row["psnr_by_orbit"]["111"]},
           "chosen":choices})
    out=Path(args.out)
    if out.exists() or not str(out).startswith(str(BASE)+"/"):
        raise RuntimeError("STOP_N2_EVAL_OUTPUT")
    out.mkdir(parents=True,exist_ok=False)
    bymethod=[]
    categories=(
      ("ALL","ALL"),
      *[(x,y) for x,y in
        sorted({(r["source"],r["family"]) for r in results})])
    for src,fam in categories:
        group=[r for r in results
            if (src=="ALL" or (r["source"],r["family"])==(src,fam))]
        glob=[r for r in group if r["raw_label"]=="ORBIT_ERROR" and r["s_gt"]=="111"]
        good=[r for r in group if r["raw_label"]=="GOOD"]
        partial=[r for r in group if r["raw_label"]=="ORBIT_ERROR" and r["s_gt"]!="111"]
        for meth in methods:
            success=clustered_ratio_ci(
                group,
                lambda r:r["raw_label"]=="ORBIT_ERROR" and
                  r["s_gt"]=="111" and
                  r["psnr_by_orbit"][r["chosen"][meth]]>=25,
                lambda r:r["raw_label"]=="ORBIT_ERROR" and r["s_gt"]=="111",
                "R0N_N2|"+meth+"|"+src+"|"+fam+"|success")
            harm=clustered_ratio_ci(
                group,
                lambda r:r["raw_label"]=="GOOD" and
                  r["psnr_by_orbit"][r["chosen"][meth]]<25,
                lambda r:r["raw_label"]=="GOOD",
                "R0N_N2|"+meth+"|"+src+"|"+fam+"|harm")
            bymethod.append({
              "method":meth,"source":src,"family":fam,
              "n_global_orbit":len(glob),"n_good":len(good),
              "n_partial_orbit":len(partial),
              "global_repair_success":success,"good_harm":harm,
              "partial_orbit_left_unfixed":len(partial),
              "n_selected_global_flip":sum(r["chosen"][meth]=="111" for r in group)})
    result={"status":"R0N_N2_CLIP_OFFLINE_EVAL_PASS",
      "source_score_sha256":meta["scored_sha256"],
      "head":args.expected_head,"total_n":3240,
      "gt_access":"offline precomputed R0.7 PSNR only",
      "gpu_hours":meta["gpu_hours"],
      "per_candidate_gpu_wall_sec":meta["gpu_seconds"]/3240,
      "rows":bymethod,
      "K2_reserved_for_planner":True}
    save_x=write_x
    output=out/"R0N_N2_CLIP_COMPARISON.json"
    save_x(output,result)
    write_x(out/"R0N_N2_COMPLETE_RECEIPT.json",{
      "status":result["status"],"head":args.expected_head,
      "comparison_sha256":sha(output),"n":3240,
      "gpu_hours":meta["gpu_hours"],
      "planner_k2_decision_pending":True})
    print("R0N_N2_CLIP_OFFLINE_EVAL_PASS 3240",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","score","evaluate"])
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--tasks")
    p.add_argument("--tasks-sha")
    p.add_argument("--scores")
    p.add_argument("--physical-gpu",type=int,default=0)
    p.add_argument("--budget-seconds",type=float,default=900.)
    a=p.parse_args()
    try:
        {"prepare":stage_prepare,"score":clip_score,
         "evaluate":eval_scores}[a.mode](a)
    except Exception as e:
        print("STOP_R0N_N2",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(8)
