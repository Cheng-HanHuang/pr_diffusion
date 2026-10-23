#!/usr/bin/env python3
"""R0.7 CPU-only audited join of all 3,240 frozen GT-free score receipts.

The output catalog is PRIVATE/OFFLINE: it deliberately contains GT paths.
Never feed this catalog to a GPU worker. Exposed cohort only; fail closed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from prdiffusion.track_r.manifest import read_manifest,assert_no_forbidden
from scripts.track_r.r0_4_score_gpu import (
    output_name as old_result_name,validate_complete as old_validate)
from scripts.track_r.r0_6_score_gpu import (
    output_name as fast_result_name,validate_complete as fast_validate)
from scripts.track_r.r0_5_prepare_cpu import verify_r04
from scripts.track_r.r0_6_prepare_cpu import audit as audit_fast_generation, manifest_rows
from scripts.track_r.r0_6_parallel_autopilot import (
    sealed_canary as audit_score_canary,
    validate_shard as audit_fast_score_shard,
    check_candidate_consistency)

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
ORIGINAL=BASE/"r0_autopilot_20261022T180315Z"
SCORE_CANARY=BASE/"r0_4_score_20261022T193238Z"
SCORE_OLD=BASE/"r0_4_parallel_20261022T200149Z"
SCORE_RECOVERY=BASE/"r0_4_recovery_20261022T235412Z"
FAST_GEN=BASE/"r0_5_parallel_20261023T012732Z"
FAST_SCORES=BASE/"r0_6_parallel_20261023T023235Z"
R06_CANARY=BASE/"r0_6_canary_20261023T020016Z"
ORIGINAL_SHA="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
OLD_SCORE_TASK_SHA="38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7"
FAST_SCORE_TASK_SHA="f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596"
OLD_HEADS={
    "canary":"8aee60d98acfeba836fff4ec824137769b0a4698",
    "parallel":"717100a17a2326f3fbbccb7cbe782ccf4e90126d",
    "recovery":"62eaaafe3e22cd6222fc66858cad6dcef532a2cd"}
FAST_HEADS={"canary":"bc62488aa088ae681cbedd4264fa3a099bed82bb",
            "parallel":"99a9e5b6a2544c56490a95f2498dd4c2cb54262e"}
FROZEN_PATH=WT/"configs/track_r/r0_frozen_v1.json"

def safe(path):
    p=Path(assert_no_forbidden(str(path)))
    if not p.is_absolute() or not p.is_file() or p.is_symlink():
        raise RuntimeError("STOP_R07_SOURCE_MISSING "+str(p))
    assert_no_forbidden(str(p.resolve()))
    return p

def filehash(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(safe(path).read_text())

def dump_new(path,payload):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def frozen_spec():
    spec=load(FROZEN_PATH)
    if (spec.get("status")!="FROZEN_V1_2026-10-21" or
        spec.get("scope")!="EXPOSED_DATA_DIAGNOSTIC_ONLY" or
        spec["labels"]["good_db"]!=25. or
        spec["labels"]["bad_db"]!=20. or
        spec["labels"]["catastrophic_db"]!=15. or
        spec["labels"]["nonorbit_consistent_residual_quantile_over_good"]!=0.95 or
        spec["policies"]["K_values"]!=[1,2,4] or
        spec["policies"]["K_values_b22_np"]!=[1,2,4,8] or
        spec["bootstrap"]["unit"]!="image" or
        spec["bootstrap"]["replicates"]!=10000 or
        spec["budget"]["gpu_hours_ceiling_aggregate"]!=16.):
        raise RuntimeError("STOP_R07_FROZEN_SPEC_CHANGED")
    return spec

def old_source_receipts(tasks):
    c=load(SCORE_CANARY/"AUTOPILOT_STATUS.json")
    p=load(SCORE_OLD/"AUTOPILOT_STATUS.json")
    r=load(SCORE_RECOVERY/"AUTOPILOT_STATUS.json")
    result={}
    def take(index,folder,head):
        if index in result:raise RuntimeError("STOP_R07_DUPLICATE_OLD_INDEX")
        row=tasks[index]
        path=folder/"results"/old_result_name(row)
        old_validate(path,row,OLD_SCORE_TASK_SHA,head)
        record=load(path)
        if (record.get("gt_access") is not False or
            record.get("uid")!=row["uid"] or
            record.get("measurement_sha256")!=row["measurement_sha256"]):
            raise RuntimeError("STOP_R07_BAD_OLD_SOURCE_OR_SCORE "+row["uid"])
        result[index]={
            "score_receipt_path":str(path),
            "score_receipt_sha256":filehash(path),
            "score_executor_head":head}
    if c.get("status")!="CANARY_COMPLETE" or c.get("head")!=OLD_HEADS["canary"]:
        raise RuntimeError("STOP_R07_OLD_CANARY_CHANGED")
    for i in range(16):
        take(i,SCORE_CANARY/"chunks/batch_0000_16",OLD_HEADS["canary"])
    if (p.get("status")!="STOP_WORKER" or
        p.get("head")!=OLD_HEADS["parallel"] or len(p["completed"])!=27):
        raise RuntimeError("STOP_R07_OLD_PARALLEL_CHANGED")
    for item in p["completed"]:
        i,n=item["start"],item["count"]
        for j in range(i,i+n):
            take(j,SCORE_OLD/"chunks"/f"batch_{i:04d}_{n:02d}",OLD_HEADS["parallel"])
    for j in range(1680,1725):
        take(j,SCORE_OLD/"chunks/batch_1680_64",OLD_HEADS["parallel"])
    if (r.get("status")!="R0_4_ALL_2520_COMPLETE_RECOVERED_4GPU" or
        r.get("head")!=OLD_HEADS["recovery"] or
        r.get("completed_count")!=2520 or len(r.get("completed",[]))!=13 or
        r.get("active_attempts")):
        raise RuntimeError("STOP_R07_RECOVERY_CHANGED")
    for item in r["completed"]:
        i,n=item["start"],item["count"]
        for j in range(i,i+n):
            take(j,SCORE_RECOVERY/"chunks"/f"batch_{i:04d}_{n:02d}",OLD_HEADS["recovery"])
    if set(result)!=set(range(2520)):
        raise RuntimeError("STOP_R07_OLD_SCORE_GAPS")
    return result

def fast_score_receipts(tasks):
    s=load(FAST_SCORES/"AUTOPILOT_STATUS.json")
    if (s.get("status")!="R0_6_ALL_720_FAST_SCORES_COMPLETE_4GPU" or
        s.get("head")!=FAST_HEADS["parallel"] or
        s.get("completed_scores")!=720 or
        len(s.get("completed",[]))!=12 or
        s.get("active_attempts") or
        s.get("tasks_sha256")!=FAST_SCORE_TASK_SHA or
        not 0<float(s.get("total_r0_gpu_seconds",0))<16*3600):
        raise RuntimeError("STOP_R07_R06_NOT_PASS")
    records={}
    def take(index,folder,head):
        if index in records:raise RuntimeError("STOP_R07_DUPLICATE_FAST_INDEX")
        row=tasks[index]
        path=folder/"results"/fast_result_name(row)
        fast_validate(path,row,FAST_SCORE_TASK_SHA,head)
        score=load(path)
        check_candidate_consistency(score,row)
        records[index]={
            "score_receipt_path":str(path),
            "score_receipt_sha256":filehash(path),
            "score_executor_head":head}
    canary=load(R06_CANARY/"AUTOPILOT_STATUS.json")
    if (canary.get("status")!="R0_6_EIGHT_CANDIDATE_CANARY_PASS" or
        canary.get("head")!=FAST_HEADS["canary"] or
        canary.get("tasks_sha256")!=FAST_SCORE_TASK_SHA):
        raise RuntimeError("STOP_R07_FAST_CANARY_CHANGED")
    for i in range(8):
        take(i,R06_CANARY/"first_8",FAST_HEADS["canary"])
    for item in s["completed"]:
        i,n=item["start"],item["count"]
        audit_fast_score_shard(FAST_SCORES,item,tasks,FAST_HEADS["parallel"])
        for j in range(i,i+n):
            take(j,FAST_SCORES/"chunks"/f"batch_{i:04d}_{n:02d}",FAST_HEADS["parallel"])
    if set(records)!=set(range(720)):
        raise RuntimeError("STOP_R07_FAST_SCORE_GAPS")
    return records,float(s["total_r0_gpu_seconds"])

def join_historical(oldrows,taskrows,scorerefs):
    mapping={r["uid"]:r for r in oldrows}
    if len(mapping)!=2520 or len(taskrows)!=2520:
        raise RuntimeError("STOP_R07_ORIGINAL_COUNT")
    out=[]
    for i,t in enumerate(taskrows):
        r=mapping.get(t["uid"])
        if r is None or any(str(r[key])!=str(t[key]) for key in
            ("source","family","image_id","run_index","candidate_path",
             "candidate_format","measurement_path","measurement_sha256")):
            raise RuntimeError("STOP_R07_ORIGINAL_TASK_GT_JOIN "+t["uid"])
        safe(r["gt_path"])
        out.append({
            "uid":t["uid"],"source":t["source"],"family":t["family"],
            "image_id":str(t["image_id"]),"run_index":int(t["run_index"]),
            "run_label":r["run_label"],
            "candidate_path":t["candidate_path"],
            "candidate_format":t["candidate_format"],
            "candidate_sha256":t["candidate_sha256"],
            "gt_path":r["gt_path"],"gt_format":r["gt_format"],
            "measurement_source":t["source"],
            "measurement_path":t["measurement_path"],
            "measurement_sha256":t["measurement_sha256"],
            "recorded_psnr_db":r["recorded_psnr_db"],
            "recorded_psnr_convention":r["recorded_psnr_convention"],
            "native_selector_name":r["native_selector_name"],
            "native_selector_value":r["native_selector_value"],
            "recorded_cost_seconds":r["recorded_cost_seconds"],
            "recorded_unet_forwards":r["recorded_unet_forwards"],
            **scorerefs[i]})
    return out

def join_fast(fastrows,score_refs,base_catalog):
    gt_map={}
    for r in base_catalog:
        if r["source"] in ("S1_b22","S2_b24_dev80"):
            key=(r["source"],r["image_id"])
            identity=(r["gt_path"],r["gt_format"],r["measurement_sha256"])
            if key in gt_map and gt_map[key]!=identity:
                raise RuntimeError("STOP_R07_INCONSISTENT_ORIGINAL_GT "+str(key))
            gt_map[key]=identity
    if len(gt_map)!=180:
        raise RuntimeError("STOP_R07_EXPECT_180_S1_S2_IMAGES")
    result=[]
    for i,r in enumerate(fastrows):
        key=(r["measurement_source"],str(r["image_id"]))
        if key not in gt_map:
            raise RuntimeError("STOP_R07_FAST_GT_NO_SOURCE "+str(key))
        gt_path,gt_format,meashash=gt_map[key]
        if r["measurement_sha256"]!=meashash:
            raise RuntimeError("STOP_R07_FAST_DIFFERENT_OBSERVATION "+r["uid"])
        result.append({
            "uid":r["uid"],"source":"S5_r0_fast","family":"FAST",
            "image_id":str(r["image_id"]), "run_index":int(r["run_index"]),
            "run_label":f"rep{r['run_index']}",
            "candidate_path":r["candidate_path"],
            "candidate_format":r["candidate_format"],
            "candidate_sha256":r["candidate_sha256"],
            "gt_path":gt_path,"gt_format":gt_format,
            "measurement_source":r["measurement_source"],
            "measurement_path":r["measurement_path"],
            "measurement_sha256":meashash,
            "recorded_psnr_db":None,"recorded_psnr_convention":"R0_RGB8_NEW",
            "native_selector_name":None,"native_selector_value":None,
            "recorded_cost_seconds":None,"recorded_unet_forwards":None,
            **score_refs[i]})
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    output=Path(a.out)
    if (not str(output).startswith(str(BASE)+"/") or output.exists()
        or output.is_symlink()):
        raise RuntimeError("STOP_R07_NEW_OUTPUT_ONLY")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30)
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_R07_GIT_HEAD_OR_DIRTY")
    spec=frozen_spec()
    original=safe(ORIGINAL/"r0_2_manifest/R0_CANDIDATE_MANIFEST.csv")
    if filehash(original)!=ORIGINAL_SHA:
        raise RuntimeError("STOP_R07_ORIGINAL_MANIFEST_SHA")
    oldrows=read_manifest(original)
    oldtasks,old_gpu_sec=verify_r04(SCORE_RECOVERY)
    oldref=old_source_receipts(oldtasks)
    fastroots=manifest_rows()
    state=load(FAST_GEN/"AUTOPILOT_STATUS.json")
    fasttasks=audit_fast_generation(state,fastroots)
    scored=load(R06_CANARY/"prepare/R0_6_GT_FREE_TASKS.json")
    if (filehash(R06_CANARY/"prepare/R0_6_GT_FREE_TASKS.json")!=FAST_SCORE_TASK_SHA or
        scored.get("head")!=FAST_HEADS["canary"] or
        len(scored.get("tasks",[]))!=720 or
        fasttasks!=scored["tasks"]):
        raise RuntimeError("STOP_R07_R06_PREPARATION_INPUT_MISMATCH")
    fastref,all_gpu_seconds=fast_score_receipts(fasttasks)
    catalog=join_historical(oldrows,oldtasks,oldref)
    catalog.extend(join_fast(fasttasks,fastref,catalog))
    if (len(catalog)!=3240 or len({r["uid"] for r in catalog})!=3240 or
        not 0<old_gpu_sec<all_gpu_seconds<16*3600):
        raise RuntimeError("STOP_R07_JOIN_COUNT_OR_BUDGET")
    output.mkdir(parents=True,exist_ok=False)
    filename=output/"R0_7_OFFLINE_GT_CATALOG.json"
    data={"schema":"prdiff.track_r.r0.7.offline-gt-catalog.v1",
          "status":"R0_7_GT_OFFLINE_CATALOG_PASS",
          "head":head,"scope":"EXPOSED_DATA_DIAGNOSTIC_ONLY",
          "WARNING":"GT PATHS: CPU OFFLINE SCIENTIFIC EVALUATION ONLY. NEVER USE AS GPU SCORER INPUT.",
          "frozen_spec_sha256":filehash(FROZEN_PATH),
          "original_manifest_sha256":ORIGINAL_SHA,
          "r04_score_tasks_sha256":OLD_SCORE_TASK_SHA,
          "r06_score_tasks_sha256":FAST_SCORE_TASK_SHA,
          "n_candidates":3240,"n_historical":2520,"n_fast":720,
          "n_sources":5,"gpu_calls":0,"no_protected_access":True,
          "all_r0_gpu_seconds":all_gpu_seconds,"rows":catalog}
    dump_new(filename,data)
    receipt={"status":"R0_7_CPU_JOIN_PASS","head":head,
             "n_candidates":3240,"n_historical":2520,"n_fast":720,
             "all_r0_gpu_hours":all_gpu_seconds/3600,
             "catalog_sha256":filehash(filename),
             "frozen_spec_sha256":filehash(FROZEN_PATH),
             "gpu_calls":0,"no_protected_access":True}
    dump_new(output/"R0_7_JOIN_RECEIPT.json",receipt)
    print("R07_JOIN_PASS 3240 RECEIPTS; GT_OFFLINE_ONLY",
          "CATALOG_SHA",receipt["catalog_sha256"],flush=True)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("STOP_R07_PREP",repr(exc),flush=True)
        traceback.print_exc()
        sys.exit(8)
