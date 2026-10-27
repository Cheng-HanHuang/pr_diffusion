#!/usr/bin/env python3
"""CPU-only publication of exact DAPS-4K native calibration evidence.

Checks the immutable PAC manifest, pristine DAPS source proof, original
completion SHA256, the two-root native smoke and all 18 checkpoint-loaded
GT-free worker receipts. No GPU, no dataset or GT, no worker invocation.
Copy only safe, non-GT final metadata; never overwrite source outputs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,subprocess,sys
from pathlib import Path
from scripts.track_r.xp_t_daps4k_worker_gpu import (
    ROOT,WT,RUN,PREP_HEAD,MANIFEST_SHA,NATIVE_SHA,DAPS_HEAD,
    root_folder,sha,read)
from scripts.track_r.xp_t_daps4k_native_cpu import RECEIPT_DIR

RUNNER_HEAD="e3a7f80e713939230fcacf830e0c11a68df0aa28"
COMPLETE_SHA="6f07bc2672c3d643996c3f5653b1e7eea63a676b3050a10766ae883e06fc2c41"
DEST=WT/"docs/track_r/executors/xp/returns/XP_T_DAPS4K_BATCH1_20261027T011832Z"

def near(value,expected,tol):
    try:return math.isfinite(float(value)) and abs(float(value)-expected)<=tol
    except (ValueError,TypeError):return False

def validate(status,complete,manifest,smoke,roots):
    if not (status.get("state")=="PASS" and status.get("phase")=="COMPLETE"
            and status.get("reason") is None
            and status.get("completed_roots")==18
            and status.get("running_roots")==0
            and status.get("active_pids")==[]
            and status.get("head")==RUNNER_HEAD
            and status.get("manifest_sha256")==MANIFEST_SHA):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_STATUS")
    if not (complete.get("status")=="XP_T_DAPS4K_BATCH1_PASS"
            and complete.get("head")==RUNNER_HEAD
            and complete.get("prep_head")==PREP_HEAD
            and complete.get("manifest_sha256")==MANIFEST_SHA
            and complete.get("family")=="DAPS"
            and complete.get("method")=="DAPS-4K"
            and complete.get("n_roots")==18 and complete.get("n_images")==6
            and complete.get("batch_size")==1
            and complete.get("workers_per_physical_gpu")==1
            and complete.get("physical_gpu_used")==[0,1,2,3]
            and complete.get("gt_free_generation") is True
            and complete.get("all_model_loads_strictly_verified") is True
            and complete.get("original_measurements_unchanged") is True
            and complete.get("historical_population_not_extended") is True
            and complete.get("evaluation_outcomes_embargo")=="NOT_SCORED"
            and complete.get("clean_timing_eligible") is True):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_COMPLETE")
    gpu_h=complete.get("total_gpu_hours")
    if not (near(gpu_h,0.9600079935751596,0.00001)
            and near(status.get("gpu_hours"),float(gpu_h),1e-8)
            and near(complete.get("total_worker_seconds"),float(gpu_h)*3600,0.001)
            and 0<float(gpu_h)<6):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_BUDGET")
    if not (manifest.get("status")=="XP_T_DAPS4K_TASKS_FROZEN"
            and manifest.get("head")==PREP_HEAD
            and manifest.get("source_head")==DAPS_HEAD
            and manifest.get("native_source_receipt_sha256")==NATIVE_SHA
            and manifest.get("n_roots")==18
            and manifest.get("gt_paths_supplied") is False
            and manifest.get("historical_paths_supplied") is False
            and manifest.get("historical_population_not_extended") is True
            and manifest.get("evaluation_outcomes_embargo")=="NOT_SCORED"):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_MANIFEST")
    cfg=manifest.get("settings",{})
    if not (cfg.get("method")=="DAPS-4K"
            and cfg.get("annealing_scheduler_config",{}).get("num_steps")==400
            and cfg.get("diffusion_scheduler_config",{}).get("num_steps")==10
            and cfg.get("mcmc_sampler_config",{}).get("num_steps")==100
            and cfg.get("mcmc_sampler_config",{}).get("mc_algo")=="langevin"):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_NOT_PUBLISHED_4K")
    if not (smoke.get("status")=="XP_T_DAPS4K_SMOKE_PASS"
            and smoke.get("indices")==[0,9]
            and smoke.get("model_loads_strictly_verified") is True
            and smoke.get("gt_free") is True
            and 0<float(smoke.get("conservative_total_projected_gpu_hours",-1))<6
            and near(smoke.get("conservative_total_projected_gpu_hours"),
                     1.0127296631314675,0.00001)
            and smoke==complete.get("smoke_gate")):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_SMOKE")
    telemetry=complete.get("per_device_timing")
    if not isinstance(telemetry,dict) or set(telemetry)!={"0","1","2","3"}:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_TELEMETRY_SET")
    for gpu,t in telemetry.items():
        if not (t.get("timing_class")=="CLEAN"
                and t.get("timing_valid") is True
                and t.get("foreign_pids")==[]
                and t.get("reasons")==[]
                and t.get("n_samples",0)>=2):
            raise RuntimeError("STOP_DAPS4K_PUBLICATION_TIMING_"+gpu)
    by_dataset=complete.get("per_dataset",{})
    if set(by_dataset)!={"ffhq","imagenet"}:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_DATASETS")
    targets={"ffhq":66.01368511701003,"imagenet":166.23148921877146}
    for name,target in targets.items():
        d=by_dataset[name]
        if not (d.get("n_roots")==9 and d.get("n_images")==3
                and d.get("runs_per_image")==3 and d.get("gt_scored") is False
                and near(d.get("sampling_wall_s_median"),target,0.00001)
                and 0<float(d.get("sampling_cuda_event_s_median",0))<3600
                and 0<float(d.get("model_loading_wall_s_median",0))<3600):
            raise RuntimeError("STOP_DAPS4K_PUBLICATION_MEDIANS_"+name)
    frozen=manifest.get("rows",[])
    summaries=complete.get("per_root",[])
    if (len(frozen)!=18 or len(summaries)!=18 or len(roots)!=18
            or len({x["uid"] for x in summaries})!=18):
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_ROOT_COUNT")
    if sorted(x["dataset"] for x in summaries)!=["ffhq"]*9+["imagenet"]*9:
        # Sorted alphabetical: ffhq before imagenet.
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_ROOT_DATASETS")
    for task,summary,root in zip(frozen,summaries,roots):
        trace=root.get("model_load_trace",{})
        if not (
            task["uid"]==summary.get("uid")==root.get("uid")
            and task["dataset"]==summary.get("dataset")==root.get("dataset")
            and root.get("status")=="XP_T_DAPS4K_ROOT_PASS"
            and root.get("runner_head")==RUNNER_HEAD
            and root.get("prep_head")==PREP_HEAD
            and root.get("manifest_sha256")==MANIFEST_SHA
            and root.get("source_receipt_sha256")==NATIVE_SHA
            and root.get("native_head")==DAPS_HEAD
            and root.get("method")=="DAPS-4K"
            and root.get("model_load_strict") is True
            and trace.get("unet_calls")==1
            and trace.get("strict") is True
            and trace.get("missing_keys")==[]
            and trace.get("unexpected_keys")==[]
            and isinstance(trace.get("tensor_count"),int)
            and trace["tensor_count"]>=100
            and root.get("seed")==task["seed"]
            and root.get("rep")==task["rep"]
            and root.get("measurement_sha256")==task["measurement_sha256"]
            and summary.get("measurement_sha256")==task["measurement_sha256"]
            and root.get("model_sha256")==summary.get("model_sha256")
            and root.get("gpu_physical")==summary.get("gpu_physical")
            and root.get("gt_free_generation") is True
            and root.get("historical_candidate_access") is False
            and root.get("evaluation_outcomes_embargo")=="NOT_SCORED"
            and near(summary.get("native_sampling_wall_s"),
                     float(root.get("native_sampling_wall_s",-1)),1e-10)
            and near(summary.get("native_sampling_cuda_event_s"),
                     float(root.get("native_sampling_cuda_event_s",-1)),1e-10)
            and 0<float(root.get("native_sampling_wall_s",0))<3600
        ):
            raise RuntimeError("STOP_DAPS4K_PUBLICATION_ROOT_"+task["uid"])
    return True

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_GPU_NOT_DISABLED")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_UNCLEAN_XP_WORKTREE")
    if sha(RUN/"XP_T_DAPS4K_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_INPUT_HASH")
    if sha(RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json")!=NATIVE_SHA:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_NATIVE_SOURCE_HASH")
    if sha(RUN/"XP_T_DAPS4K_COMPLETE.json")!=COMPLETE_SHA:
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_COMPLETE_HASH")
    status=read(RUN/"XP_T_DAPS4K_STATUS.json")
    complete=read(RUN/"XP_T_DAPS4K_COMPLETE.json")
    manifest=read(RUN/"XP_T_DAPS4K_TASKS.json")
    smoke=read(RUN/"XP_T_DAPS4K_SMOKE_GATE.json")
    originals=[]
    for row in manifest["rows"]:
        dest=root_folder(row)
        path=dest/"XP_T_DAPS4K_ROOT_COMPLETE.json"
        r=read(path)
        summary=next(x for x in complete["per_root"] if x["uid"]==row["uid"])
        if (sha(path)!=summary.get("root_receipt_sha256")
                or sha(dest/"reconstruction.pt")!=r.get("reconstruction_pt_sha256")
                or sha(dest/"reconstruction.png")!=r.get("reconstruction_png_sha256")):
            raise RuntimeError("STOP_DAPS4K_PUBLICATION_ROOT_FILES")
        originals.append(r)
    validate(status,complete,manifest,smoke,originals)
    if DEST.exists() or DEST.is_symlink():
        raise RuntimeError("STOP_DAPS4K_PUBLICATION_ALREADY_STAGED")
    DEST.mkdir(parents=True,exist_ok=False)
    files=[
        ("XP_T_DAPS4K_PREP.json",RUN/"XP_T_DAPS4K_PREP.json"),
        ("XP_T_DAPS4K_NATIVE_SOURCE.json",
            RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json"),
        ("XP_T_DAPS4K_STATUS.json",RUN/"XP_T_DAPS4K_STATUS.json"),
        ("XP_T_DAPS4K_SMOKE_GATE.json",RUN/"XP_T_DAPS4K_SMOKE_GATE.json"),
        ("XP_T_DAPS4K_COMPLETE.json",RUN/"XP_T_DAPS4K_COMPLETE.json"),
    ]
    hashes=[]
    for name,source in files:
        shutil.copyfile(source,DEST/name)
        hashes.append((sha(DEST/name),name))
    with (DEST/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
        for digest,name in hashes:f.write(digest+"  "+name+"\n")
    print("XP_T_DAPS4K_TIMING_EVIDENCE_STAGE_PASS")
    print("COMPLETE_SHA256="+COMPLETE_SHA)
    print("ROOTS=18 GPU_H="+str(complete["total_gpu_hours"]))
    print("FFHQ_S="+str(complete["per_dataset"]["ffhq"]["sampling_wall_s_median"]))
    print("IMAGENET_S="+str(complete["per_dataset"]["imagenet"]["sampling_wall_s_median"]))
    print("TIMING_CLASS=CLEAN_ON_FOUR_GPUS")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_T_DAPS4K_TIMING_PUBLISH "+str(e),flush=True)
        sys.exit(8)
