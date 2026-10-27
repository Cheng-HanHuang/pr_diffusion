#!/usr/bin/env python3
"""CPU-only publication of exact PAC SITCOM XP-T FFHQ/ImageNet timing receipts.

No GPU, no scoring and no ground truth. Verify frozen source manifest, exact
completion SHA and all 18 individual native model-load/timing receipts before
staging a new GitHub evidence directory. Preserve original outputs unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from scripts.track_r.xp_t_sitcom_worker_gpu import (
    ROOT,WT,RUN,PREP_HEAD,MANIFEST_SHA,record_folder,sha,safe)

RUNNER_HEAD="dacd57d0739ee218490befde50b2bed5991bb80b"
COMPLETE_SHA="bb2c3c377f07be2982e412eb96ff1a8b35805146dc021cc8baab3abd6ddc075e"
GPU_H=0.36916
DEST=WT/"docs/track_r/executors/xp/returns/XP_T_SITCOM_BATCH1_20261026T233451Z"

def read(p):return json.loads(safe(p).read_text())

def validate(status,complete,manifest,roots):
    if not (status.get("state")=="PASS" and status.get("reason") is None
        and status.get("completed_roots")==18
        and status.get("running_roots")==0
        and status.get("active_pids")==[]
        and status.get("head")==RUNNER_HEAD
        and status.get("manifest_sha256")==MANIFEST_SHA):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_STATUS")
    if not (complete.get("status")=="XP_T_SITCOM_BATCH1_PASS"
        and complete.get("head")==RUNNER_HEAD
        and complete.get("prep_head")==PREP_HEAD
        and complete.get("manifest_sha256")==MANIFEST_SHA
        and complete.get("family")=="SITCOM"
        and complete.get("n_roots")==18
        and complete.get("n_images")==6
        and complete.get("workers_per_physical_gpu")==1
        and complete.get("physical_gpu_used")==[0,1,2,3]
        and complete.get("gt_free_generation") is True
        and complete.get("all_model_loads_strictly_verified") is True
        and complete.get("historical_population_not_extended") is True
        and complete.get("evaluation_outcomes_embargo")=="NOT_SCORED"
        and complete.get("clean_timing_eligible") is True):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_COMPLETE")
    total_h=float(complete.get("total_gpu_hours",-1))
    if not (0<total_h<2 and abs(total_h-GPU_H)<0.002
        and abs(total_h-float(status.get("gpu_hours",-1)))<1e-8):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_GPU_BUDGET")
    if (manifest.get("status")!="XP_T_SITCOM_TASKS_FROZEN" or
        manifest.get("head")!=PREP_HEAD or manifest.get("n_roots")!=18 or
        manifest.get("gt_paths_supplied") is not False or
        manifest.get("historical_paths_supplied") is not False):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_MANIFEST")
    timing=complete.get("per_device_timing")
    if not isinstance(timing,dict) or set(timing)!={"0","1","2","3"}:
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_DEVICE_SET")
    for gpu,t in timing.items():
        if (t.get("timing_class")!="CLEAN" or t.get("timing_valid") is not True
            or t.get("foreign_pids")!=[] or t.get("reasons")!=[]
            or t.get("n_samples",0)<2):
            raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_TELEMETRY_"+gpu)
    datasets=complete.get("per_dataset",{})
    if set(datasets)!={"ffhq","imagenet"}:
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_DATASETS")
    for dataset in ("ffhq","imagenet"):
        value=datasets[dataset]
        if (value.get("n_roots")!=9 or value.get("n_images")!=3
            or value.get("runs_per_image")!=3
            or value.get("gt_access") is not False
            or not 0<float(value.get("sampling_wall_s_median",0))<3600
            or not 0<float(value.get("sampling_cuda_event_s_median",0))<3600):
            raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_MEDIANS")
    if (abs(float(datasets["ffhq"]["sampling_wall_s_median"])-32.104)>0.001
        or abs(float(datasets["imagenet"]["sampling_wall_s_median"])-58.343)>0.001):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_UNEXPECTED_MEDIANS")
    report_roots=complete.get("per_root",[])
    frozen_roots=manifest.get("rows",[])
    if (not isinstance(report_roots,list) or len(report_roots)!=18
        or len(frozen_roots)!=18 or len(roots)!=18
        or len({r["uid"] for r in report_roots})!=18):
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_ROOT_COUNT")
    for task,summary,original in zip(frozen_roots,report_roots,roots):
        if (task["uid"]!=summary.get("uid") or
            task["uid"]!=original.get("uid") or
            original.get("runner_head")!=RUNNER_HEAD or
            original.get("status")!="XP_T_SITCOM_ROOT_PASS" or
            original.get("model_loaded_strictly") is not True or
            original.get("strict_load_trace",{}).get("unet_calls")!=1 or
            original.get("gt_free_generation") is not True or
            original.get("evaluation_outcomes_embargo")!="NOT_SCORED" or
            original.get("historical_candidate_access") is not False or
            original.get("measurement_sha256")!=task["measurement_sha256"] or
            summary.get("measurement_sha256")!=task["measurement_sha256"] or
            original.get("gpu_physical")!=summary.get("gpu_physical") or
            original.get("dataset")!=task["dataset"] or
            summary.get("dataset")!=task["dataset"] or
            original.get("model_sha256")!=summary.get("model_sha256") or
            summary.get("model_loading_wall_s")!=original.get("native_model_loading_wall_s") or
            summary.get("native_sampling_wall_s")!=original.get("native_sampling_wall_s")):
            raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_ROOT_PAIR_"+task["uid"])
    return True

def run(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_GPU_VISIBLE")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_CHECKOUT")
    if sha(RUN/"XP_T_SITCOM_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_FROZEN_SHA")
    if sha(RUN/"XP_T_SITCOM_COMPLETE.json")!=COMPLETE_SHA:
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_COMPLETE_SHA")
    manifest=read(RUN/"XP_T_SITCOM_TASKS.json")
    status=read(RUN/"XP_T_SITCOM_STATUS.json")
    complete=read(RUN/"XP_T_SITCOM_COMPLETE.json")
    originals=[]
    for row in manifest["rows"]:
        folder=record_folder(row)
        receipt_path=folder/"XP_T_SITCOM_ROOT_COMPLETE.json"
        original=read(receipt_path)
        originals.append(original)
        if (sha(receipt_path)!=next(
                r["root_receipt_sha256"] for r in complete["per_root"]
                if r["uid"]==row["uid"])
            or sha(folder/"reconstruction.pt")!=original.get("reconstruction_pt_sha256")
            or sha(folder/"reconstruction.png")!=original.get("reconstruction_png_sha256")):
            raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_FILE_INTEGRITY")
    validate(status,complete,manifest,originals)
    if DEST.exists() or DEST.is_symlink():
        raise RuntimeError("STOP_SITCOM_TIMING_PUBLISH_DEST_EXISTS")
    DEST.mkdir(parents=True,exist_ok=False)
    names=["XP_T_SITCOM_PREP.json","XP_T_SITCOM_STATUS.json",
           "XP_T_SITCOM_COMPLETE.json"]
    hashes=[]
    for name in names:
        shutil.copyfile(RUN/name,DEST/name)
        hashes.append((sha(DEST/name),name))
    with (DEST/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
        for digest,name in hashes:f.write(digest+"  "+name+"\n")
    print("XP_T_SITCOM_TIMING_EVIDENCE_STAGE_PASS")
    print("COMPLETE_SHA256="+COMPLETE_SHA)
    print("ROOTS=18 GPU_H="+str(complete["total_gpu_hours"]))
    print("FFHQ_S="+str(complete["per_dataset"]["ffhq"]["sampling_wall_s_median"]))
    print("IMAGENET_S="+str(complete["per_dataset"]["imagenet"]["sampling_wall_s_median"]))
    print("TIMING_CLASS=CLEAN_ON_FOUR_GPUS")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:run(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_T_SITCOM_TIMING_PUBLISH "+str(e),flush=True)
        sys.exit(8)
