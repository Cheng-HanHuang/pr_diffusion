#!/usr/bin/env python3
"""R0.3c CPU-only FAST replay adjudication and GPU-hours projection.

The native FAST worker never reads GT. Here, only AFTER sampling, compare
the A1.5 69073/rep0 replay with the frozen historical PNG and PSNR.
Stop on replay >0.05 dB, integrity mismatch or 16 GPU-hour projection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
G0_GPU_HEAD="6767f1850235a3df0c94fea9b53135016dbe271f"
MANIFEST_SHA="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
MAX_TOTAL_GPU_HOURS=16.
REPLAY_MAX_DB_DIFF=0.05
NEW_FAST_ROOTS=720


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def safe(path):
    p=Path(path)
    names=(str(p).lower(),str(p.resolve(strict=False)).lower())
    if any(token in s for token in ("confirmation","confirm305") for s in names):
        raise RuntimeError("STOP_R03C_PROTECTED_PATH")
    if not p.is_file():raise RuntimeError("STOP_R03C_MISSING_SOURCE "+str(p))
    return p


def load(path):
    return json.loads(safe(path).read_text())


def write_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


def budget_projection(existing_gpu_h,s5_gpu_h,wall_seconds,
                      gpu_smoke_h,additional_gpu_h=0.,
                      safety_multiplier=1.25):
    if (not wall_seconds or any(not math.isfinite(float(v)) or float(v)<0
        for v in [existing_gpu_h,s5_gpu_h,*wall_seconds,
                  gpu_smoke_h,additional_gpu_h])):
        raise ValueError("invalid budget inputs")
    native_reference_s=max(wall_seconds)
    native_projected_h=NEW_FAST_ROOTS*native_reference_s/3600.0
    subtotal=(existing_gpu_h+s5_gpu_h+native_projected_h+
              gpu_smoke_h+additional_gpu_h)
    adjusted=subtotal*safety_multiplier
    return {"remaining_existing_scoring_gpu_h":existing_gpu_h,
        "remaining_s5_scoring_gpu_h":s5_gpu_h,
        "native_fast_smoke_wall_s":list(wall_seconds),
        "conservative_native_wall_per_root_s":native_reference_s,
        "new_fast_roots":NEW_FAST_ROOTS,
        "native_generation_projected_gpu_h":native_projected_h,
        "measured_prior_smoke_gpu_h":gpu_smoke_h,
        "measured_fast_smoke_gpu_h":additional_gpu_h,
        "raw_total_gpu_h":subtotal,
        "projection_multiplier":safety_multiplier,
        "conservative_total_gpu_h":adjusted,
        "ceiling_gpu_h":MAX_TOTAL_GPU_HOURS,
        "status":"PASS" if adjusted<=MAX_TOTAL_GPU_HOURS else "STOP_BUDGET_PROJECTION_EXCEEDS_16_GPUH"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--fast-smoke-root",required=True)
    ap.add_argument("--g0-gpu-receipt",required=True)
    ap.add_argument("--g1-receipt",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_R03C_OUT_OF_NAMESPACE_OR_OCCUPIED")
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if actual!=args.expected_head or dirty:
        raise RuntimeError("STOP_R03C_GIT_HEAD_OR_DIRTY")
    gpu=load(Path(args.fast_smoke_root)/"R0_3C_FAST_GPU_RECEIPT.json")
    if gpu.get("status")!="R0_3C_FAST_GPU_SMOKE_COMPLETE_NEEDS_CPU_ADJUDICATION" or gpu.get("head")!=actual:
        raise RuntimeError("STOP_R03C_FAST_SMOKE_RECEIPT")
    by_source={r["source"]:r for r in gpu["cases"]}
    if set(by_source)!={"A15_REPLAY","S2_b24_dev80"}:
        raise RuntimeError("STOP_R03C_FAST_SMOKE_CASES")
    g0=load(args.g0_gpu_receipt)
    if (g0.get("status")!="G0_FULL_PASS_R0_3B_NOT_FULL_R0_3" or
        g0.get("head")!=G0_GPU_HEAD or
        g0["comparison"]["status"]!="PASS" or
        g0["comparison"]["max_relative_difference"]>1e-5):
        raise RuntimeError("STOP_R03C_NO_FULL_G0")
    g1=load(args.g1_receipt)
    if g1.get("status")!="G1_PASS" or g1.get("manifest_sha256")!=MANIFEST_SHA:
        raise RuntimeError("STOP_R03C_NO_G1_PASS")
    for source,sub in (("A15_REPLAY","a15_replay"),
                       ("S2_b24_dev80","s2_new")):
        rec=by_source[source]
        folder=Path(args.fast_smoke_root)/sub
        if rec["output"]!=str(folder) or rec["status"]!="R0_FAST_SMOKE_CASE_PASS":
            raise RuntimeError("STOP_R03C_OUTPUT_OR_STATUS")
        if (sha(safe(folder/"reconstruction.pt"))!=rec["reconstruction_pt_sha256"] or
            sha(safe(folder/"reconstruction.png"))!=rec["reconstruction_png_sha256"]):
            raise RuntimeError("STOP_R03C_FAST_OUTPUT_SHA")
        if rec.get("gt_access") is not False:
            raise RuntimeError("STOP_R03C_GPU_WORKER_GT_ACCESS")
    import numpy as np
    from prdiffusion.track_r.physics import load_png_rgb8,psnr_rgb8,to_rgb8
    from scripts.track_r.r0_2_g0_cpu import gt_source
    from prdiffusion.track_r.manifest import read_manifest
    paths=load(WT/"configs/track_r/pac_paths_track_r_v1.json")
    a15root=Path(paths["sources"]["S4_a15"]["root"])/"69073"/"rep0"
    ref=load(a15root/"A15_SMOKE_COMPLETE.json")
    if (ref.get("status")!="A15_FAST_CANDIDATE_PASS" or
        str(ref.get("image_id"))!="69073" or ref.get("rep")!=0 or
        ref.get("seed")!=by_source["A15_REPLAY"]["seed"]):
        raise RuntimeError("STOP_R03C_WRONG_HISTORICAL_A15_CASE")
    if (ref["measurement_sha256"]!=by_source["A15_REPLAY"]["measurement_sha256"] or
        ref["measurement_tensor_sha256"]!=by_source["A15_REPLAY"]["measurement_tensor_sha256"]):
        raise RuntimeError("STOP_R03C_A15_MEAS_IDENTITY")
    oldfile=safe(a15root/"reconstruction.png")
    if sha(oldfile)!=ref["reconstruction_png_sha256"]:
        raise RuntimeError("STOP_R03C_A15_REFERENCE_PNG_SHA")
    freeze=Path(paths["sources"]["S3_a12"]["inputs_root"])/"A1_2_INPUT_FREEZE_COMPLETE.json"
    frozen=load(freeze)
    target=[r for r in frozen["rows"] if str(r["image_id"])=="69073"]
    if len(target)!=1:raise RuntimeError("STOP_R03C_GT_SOURCE_ID")
    newfile=Path(args.fast_smoke_root)/"a15_replay"/"reconstruction.png"
    gt=to_rgb8(gt_source({
        "gt_path":target[0]["ground_truth_tensor_path"],"gt_format":"pt"}))
    oldrgb=load_png_rgb8(oldfile)
    newrgb=load_png_rgb8(newfile)
    oldpsnr=float(psnr_rgb8(oldrgb,gt)[0])
    newpsnr=float(psnr_rgb8(newrgb,gt)[0])
    if abs(oldpsnr-float(ref["metric"]["raw_rgb8_psnr_db"]))>0.01:
        raise RuntimeError("STOP_R03C_A15_RECORDED_REPLAY")
    aa=oldrgb.flatten().numpy().astype(np.float64)
    bb=newrgb.flatten().numpy().astype(np.float64)
    corr=float(np.corrcoef(aa,bb)[0,1])
    if not math.isfinite(corr):
        raise RuntimeError("STOP_R03C_PIXEL_CORRELATION_UNDEFINED")
    same=sha(newfile)==sha(oldfile)
    replay={"status":"PASS" if (same or abs(oldpsnr-newpsnr)<=REPLAY_MAX_DB_DIFF) else "FAIL",
        "image_id":"69073","rep":0,
        "bit_exact_png":same,"historical_raw_rgb8_psnr_db":oldpsnr,
        "new_raw_rgb8_psnr_db":newpsnr,
        "abs_psnr_difference_db":abs(oldpsnr-newpsnr),
        "pixel_correlation":corr,
        "psnr_tolerance_db":REPLAY_MAX_DB_DIFF,
        "reference_png_sha256":ref["reconstruction_png_sha256"],
        "generated_png_sha256":by_source["A15_REPLAY"]["reconstruction_png_sha256"]}
    score_time=(float(g0["projected_existing_scoring_gpu_hours"]),
                float(g0["projected_future_s5_scoring_gpu_hours"]))
    wall=[float(case["sample_only_wall_seconds"]) for case in gpu["cases"]]
    prior_gpu=float(g0["measured_smoke_gpu_hours"])
    # R0.3a used one GPU for 9.3 seconds, load overhead ignored by previous receipt;
    # include a conservatively accounted 0.01 GPUh for that earlier pilot.
    prior_gpu += 0.01
    native_h=sum(wall)/3600.0
    budget=budget_projection(*score_time,wall,
                             gpu_smoke_h=prior_gpu,
                             additional_gpu_h=native_h)
    out.mkdir(parents=True,exist_ok=False)
    result={
        "schema":"prdiff.track_r.r0.3c.adjudication.v1",
        "source_head":actual,
        "status":"R0_3C_PASS_READY_FOR_R04_IMPLEMENTATION" if
          replay["status"]=="PASS" and budget["status"]=="PASS" else
          "STOP_R0_3C_REPLAY_OR_BUDGET",
        "g0":"PASS","g1":"PASS","replay":replay,"budget":budget,
        "input_root":args.fast_smoke_root,
        "no_protected_access":True,"gt_only_cpu_after_native_generation":True,
        "full_r0_gpu_execution_automatically_launched":False}
    write_x(out/"R0_3C_ADJUDICATION.json",result)
    print(json.dumps({"status":result["status"],
        "replay":replay,"budget":budget},sort_keys=True),flush=True)
    return 0 if result["status"].startswith("R0_3C_PASS") else 7

if __name__=="__main__":
    sys.exit(main())
