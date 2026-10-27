#!/usr/bin/env python3
"""XP-D1 phase-1 (ImageNet FAST K=4) CPU-only prospective v2 handoff.

Only scorer authorized to read the separate ImageNet GT-only manifest.
Scores **only D1_P1 indices 0..3**, never D1_P2, A64 or XP-A16.
Uses canonical RGB8, eight-channel orbit PSNR, frozen signed y,
Good20 and q0.95 rho among raw Good20 for FAST ImageNet alone.
Per-image strings are committed later to open v2 theory forecast window.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path

from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA,ROOT,WT,BASE
from scripts.track_r import xp_night_supervisor as night
from scripts.track_r.xp_night_publish_cpu import (
    RUN,GEN_HEAD,EXPECTED_PREP_HASH,EXPECTED_MANIFEST_HASHES,safe,digest)
from scripts.track_r.xp_fast_gpu import root_folder,validate_complete,frozen_roots
from prdiffusion.track_r.physics import to_rgb8,rho_signed,psnr_rgb8
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse

SOURCE=IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json"
COMMON=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
OUTPUT_PREFIX="xp_d1p1_eval_"
GOOD=20.
THRESHOLD_MIN=0.0
FORBIDDEN=("confirmation","confirm305","test-ffhq","demo-ffhq")

def read(path):
    return json.loads(safe(path).read_text())

def validate_setup():
    if (digest(RUN/"XP_NIGHT_PREP.json")!=EXPECTED_PREP_HASH
        or digest(RUN/"D1_P1.json")!=EXPECTED_MANIFEST_HASHES["D1_P1"]
        or digest(COMMON)!=COMMON_Y_SHA):
        raise RuntimeError("STOP_D1P1_SOURCE_HASH")
    state=read(RUN/"XP_NIGHT_CONTROLLER_STATE.json")
    if not (state.get("status")=="MAIN_PASS"
            and state.get("completed_roots")==9440
            and state.get("head")==GEN_HEAD
            and state.get("outcomes_scored") is False
            and state.get("sealed_outcomes_not_inspected") is True):
        raise RuntimeError("STOP_D1P1_GENERATION_GATE")
    stage=read(RUN/"D1_P1.json")
    rows=frozen_roots(stage,GEN_HEAD)
    common=read(COMMON)
    gt=read(SOURCE)
    if (len(rows)!=400 or
        stage.get("dataset")!="imagenet" or
        stage.get("stage")!="D1_P1" or
        stage.get("outcome_embargo") is not False or
        stage.get("gt_paths_supplied") is not False or
        common.get("n_images")!=100 or len(common["rows"])!=100 or
        gt.get("status")!="GT_ONLY_OFFLINE_NOT_FOR_GENERATION" or
        len(gt["rows"])!=100):
        raise RuntimeError("STOP_D1P1_SCHEMA")
    inputs={r["basename"]:r for r in common["rows"]}
    sources={r["basename"]:r for r in gt["rows"]}
    if set(inputs)!=set(sources) or len(inputs)!=100:
        raise RuntimeError("STOP_D1P1_GT_MAP")
    for basename,src in sources.items():
        if (src["source_sha256"]!=inputs[basename]["source_image_sha256"]
            or src["source_path"] is None or
            not str(src["source_path"]).startswith(str(ROOT/"data/imagenet/test-imagenet")+"/") or
            digest(src["source_path"])!=src["source_sha256"]):
            raise RuntimeError("STOP_D1P1_IMAGE_SHA")
    groups=defaultdict(list)
    for row in rows:
        if (row["source"]!="XP_D1" or
            row["measurement_source"]!="XP_IMAGENET100" or
            row["image_id"] not in inputs or
            row["measurement_path"]!=inputs[row["image_id"]]["measurement_path"] or
            row["measurement_sha256"]!=inputs[row["image_id"]]["measurement_sha256"]):
            raise RuntimeError("STOP_D1P1_MEASUREMENT_LINEAGE")
        groups[row["image_id"]].append(row)
    if (len(groups)!=100 or
        any([x["rep"] for x in group]!=[0,1,2,3] for group in groups.values())):
        raise RuntimeError("STOP_D1P1_RUN_ORDER")
    return rows,inputs,sources

def labels(rows):
    import numpy as np
    good=[r["rho_signed"] for r in rows if r["psnr_raw"]>=GOOD]
    if not good:raise RuntimeError("STOP_D1P1_NO_GOOD20_TAU")
    tau=float(np.quantile(np.array(good,dtype=np.float64),.95,method="linear"))
    if not (math.isfinite(tau) and tau>THRESHOLD_MIN):
        raise RuntimeError("STOP_D1P1_INVALID_TAU")
    for row in rows:
        if row["psnr_raw"]>=GOOD:
            row["label"]="G"
        elif row["psnr_orbit_best"]>=GOOD:
            row["label"]="O" if row["s_gt"]=="111" else "P"
        elif row["rho_signed"]<=tau:
            row["label"]="C"
        else:
            row["label"]="I"
    return tau

def per_image(rows,tau):
    group=defaultdict(list)
    for r in rows:group[r["image_id"]].append(r)
    output=[]
    for iid,runs in sorted(group.items()):
        runs.sort(key=lambda x:x["rep"])
        if len(runs)!=4 or [r["rep"] for r in runs]!=[0,1,2,3]:
            raise RuntimeError("STOP_D1P1_IMAGE_RUNS")
        rank=sorted(runs,key=lambda x:(x["rho_signed"],x["rep"]))
        counts=Counter(r["label"] for r in runs)
        output.append({
            "image_id":iid,
            "source":"XP_D1_P1",
            "family":"FAST",
            "n_runs":4,
            "n_GOOD":counts["G"],"n_ORBIT_global":counts["O"],
            "n_ORBIT_partial":counts["P"],
            "n_CONSISTENT":counts["C"],"n_INCONSISTENT":counts["I"],
            "run_labels":"".join(r["label"] for r in runs),
            "e0_rank_labels":"".join(r["label"] for r in rank),
            "e0_rank_order":",".join(str(r["rep"]) for r in rank),
            "rho_signed_by_run":";".join(format(r["rho_signed"],".6g") for r in runs),
            "tau_signed_good_q95":format(tau,".17g"),
            "orbit_best_member":"".join(
                str(int(r["s_gt"],2)) if r["label"] in ("O","P") else "."
                for r in runs),
        })
    if len(output)!=100 or sum(int(x["n_runs"]) for x in output)!=400:
        raise RuntimeError("STOP_D1P1_OUTPUT_COUNTS")
    return output

def evaluate(rows,inputs,sources):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from prdiffusion.track_r.physics import magnitude
    torch.set_num_threads(min(torch.get_num_threads(),4))
    tf=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),
                           transforms.CenterCrop(256)])
    ycache={}
    gcache={}
    final=[]
    for k,row in enumerate(rows):
        iid=row["image_id"]
        if iid not in gcache:
            with Image.open(safe(sources[iid]["source_path"])) as im:g=tf(im)
            if tuple(g.shape)==(1,256,256):g=g.repeat(3,1,1)
            if tuple(g.shape)!=(3,256,256):
                raise RuntimeError("STOP_D1P1_GT_CHANNELS")
            gx=(g*2-1).unsqueeze(0).float()
            gcache[iid]=(gx,to_rgb8(gx))
            record=torch.load(safe(inputs[iid]["measurement_path"]),
                         map_location="cpu",weights_only=True)
            y=record["measurement"]
            if tuple(y.shape)!=(1,3,384,384):
                raise RuntimeError("STOP_D1P1_SIGNED_Y")
            if digest(inputs[iid]["measurement_path"])!=inputs[iid]["measurement_sha256"]:
                raise RuntimeError("STOP_D1P1_Y_DRIFT")
            ycache[iid]=y.float()
        gx,g8=gcache[iid]
        y=ycache[iid]
        chunk=next(j for j in night.jobs() if
              j["stage"]=="D1_P1" and
              j["start"]<=k<j["start"]+j["count"])
        folder=root_folder(night.paths(RUN,chunk),row)
        proof=validate_complete(folder,row,GEN_HEAD,
                 EXPECTED_MANIFEST_HASHES["D1_P1"])
        rec=torch.load(safe(folder/"reconstruction.pt"),
                 map_location="cpu",weights_only=True)
        tensor=rec["reconstruction"]
        if (not isinstance(tensor,torch.Tensor) or
            tensor.shape!=(1,3,256,256) or
            not torch.isfinite(tensor).all()):
            raise RuntimeError("STOP_D1P1_RECONSTRUCTION_SCHEMA")
        tensor=tensor.float()
        x8=to_rgb8(tensor)
        orbit=orbit_psnr_channel_mse(x8,g8)
        raw=float(psnr_rgb8(x8,g8)[0])
        if abs(orbit["000"]-raw)>0.0001:
            raise RuntimeError("STOP_D1P1_PSNR_CANONICAL")
        rho=float(rho_signed(tensor,y)[0])
        if not math.isfinite(rho):raise RuntimeError("STOP_D1P1_RHO")
        best=max(sorted(orbit),key=lambda b:orbit[b])
        final.append({
           "uid":row["uid"],"image_id":iid,"rep":row["rep"],
           "psnr_raw":raw,"psnr_orbit_best":orbit[best],
           "s_gt":best,"rho_signed":rho,
        })
        if (k+1)%100==0:
            print("XP_D1P1_CPU_SCORE_PROGRESS",k+1,"/400",flush=True)
    return final

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_D1P1_GPU_VISIBLE")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
        text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_D1P1_GIT")
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith(OUTPUT_PREFIX)
        or out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_D1P1_OUTPUT_NAMESPACE")
    rows,inputs,sources=validate_setup()
    # Only after all frozen-input checks, evaluate exact K<=4.
    scored=evaluate(rows,inputs,sources)
    tau=labels(scored)
    summary=per_image(scored,tau)
    out.mkdir(parents=True,exist_ok=False)
    csvfile=out/"XP_D1_P1_A8_STYLE_PER_IMAGE.csv"
    with csvfile.open("x",newline="",encoding="utf-8") as f:
        fields=list(summary[0]);w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(summary)
    jsonfile=out/"XP_D1_P1_K4_SUMMARY.json"
    counts=Counter(r["label"] for r in scored)
    receipt={
        "status":"XP_D1_P1_K4_EVAL_PASS",
        "evaluation_head":a.expected_head,
        "native_generation_head":GEN_HEAD,
        "stage_manifest_sha256":EXPECTED_MANIFEST_HASHES["D1_P1"],
        "n_images":100,"n_runs":400,"run_indices":[0,1,2,3],
        "family":"FAST","dataset":"imagenet",
        "primary_good_psnr_db":GOOD,
        "rho_signed_good_q95_tau":tau,
        "tau_estimation":"numpy quantile linear, raw Good20 only, original indices 0-3",
        "label_counts":dict(counts),
        "rows_sha256":digest(csvfile),
        "per_run_quality_not_published":True,
        "sealed_D1_P2_scored":False,"sealed_A64_scored":False,
        "sealed_XP_A16_scored":False,
        "use_for_v2_prediction_freeze":True
    }
    with jsonfile.open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print("XP_D1_P1_K4_EVAL_PASS images=100 runs=400")
    print("D1P1_A8_STYLE_SHA="+receipt["rows_sha256"])
    print("D1P1_K4_RECEIPT="+str(jsonfile))
    print("D1P1_RESULT_DIR="+str(out))
    print("ONLY_D1_P1_GT_EVALUATED=true")
    print("D1_PHASE2_OUTCOME_EMBARGO=ACTIVE")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as ex:
        traceback.print_exc()
        print("ACTION NEEDED: XP_D1P1_SCORE "+str(ex),flush=True)
        sys.exit(8)
