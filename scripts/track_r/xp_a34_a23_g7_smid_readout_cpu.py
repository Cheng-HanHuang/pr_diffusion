#!/usr/bin/env python3
"""A23 G7 S_mid global2: exact 280-image GT-only CPU readout and bootstrap.

Descriptive upright-retention rates only; the estimator was frozen in A20.3.
No GT-selection adjustment, no new main-paper claim, no other G7 policy claim.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.track_r.xp_a34_a23_g7_smid_controller_cpu import (
    ROOT,BASE,WT,N,SOURCE_SHA,C2_SOURCE_SHA,valid_result,git,sha,jsonread,fail,
    C2_USED,G7_CAP_H,PROJECT_CAP_H,
)

def ci(xs,key):
    seed=int.from_bytes(hashlib.sha256(
        ("A23G7SMIDBOOT|"+key).encode()).digest()[:8],"big")
    rng=np.random.default_rng(seed)
    x=np.asarray(xs,dtype=np.int64)
    return tuple(float(v) for v in np.quantile(
        rng.choice(x,(10000,len(x)),replace=True).mean(axis=1),
        [0.025,0.975]))

def main(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),
         "READOUT_CODE_CLEAN")
    run=Path(a.run)
    fail(run.parent==BASE and
         run.name.startswith("xp_a34_a23_g7_smid_prod_")
         and run.is_dir(),"RUN")
    s=jsonread(run/"XP_A34_G7_S_MID_FINAL_STATUS.json")
    fail(s["status"]=="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
         and s["n_complete"]==280
         and not s["workers_active"]
         and s["head"]==a.expected_head
         and s["frozen_source_manifest_sha256"]==SOURCE_SHA
         and s["G7_total_charged_gpu_h"]<G7_CAP_H
         and s["project_charged_gpu_h"]<PROJECT_CAP_H,
         "COMPLETE_FROZEN_RECEIPT")
    images=jsonread(BASE/"xp_c_c2_a22_20261104T025355Z/XP_C_GT_IMAGES.json")["images"]
    rows=[]
    for i in range(N):
        src=run/"results"/f"{i:03d}.json"
        fail(valid_result(src,i),"IMAGE_RECEIPT_"+str(i))
        rec=jsonread(src)
        score=rec["B8_fixed_t_native_score"]
        upright=int(rec["upright_kept"])
        rows.append(dict(
            index=i,dataset=rec["dataset"],basename=rec["basename"],
            original_source_sha256=rec["source_sha256"],
            uid=rec["uid"],L000=score["L000"],L111=score["L111"],
            prior_margin=score["prior_margin"],
            prior_choice=score["prior_choice"],upright_kept=upright,
            reused_A33_pilot=int(rec.get("reused_A33_pilot",False)),
            result_receipt_sha256=sha(src)))
    fail(len(rows)==280 and sum(r["dataset"]=="ffhq" for r in rows)==180
         and sum(r["dataset"]=="imagenet" for r in rows)==100
         and sum(r["reused_A33_pilot"] for r in rows)==2,
         "EXACT_COHORT_AND_PILOT_GRID")
    folder=run/"readout"
    fail(not folder.exists(),"NO_READOUT_OVERWRITE")
    folder.mkdir(mode=0o700)
    per=folder/"XP_A23_G7_S_MID_GLOBAL2_GT_PER_IMAGE.csv"
    grouped=folder/"XP_A23_G7_S_MID_GLOBAL2_GT_AGGREGATE.csv"
    with per.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    aggregates=[]
    for dataset in ("ffhq","imagenet","pooled"):
        arr=[r["upright_kept"] for r in rows
             if r["dataset"]==dataset or dataset=="pooled"]
        n=len(arr)
        lo,hi=ci(arr,dataset)
        aggregates.append(dict(
            policy="S_mid_global2",dataset=dataset,n=n,
            n_upright_kept=sum(arr),n_inverted=n-sum(arr),
            upright_retention_fraction=format(sum(arr)/n,".15g"),
            bootstrap_ci_lo=format(lo,".15g"),
            bootstrap_ci_hi=format(hi,".15g"),
            bootstrap_replicates=10000,
            bootstrap_seed_rule="SHA256('A23G7SMIDBOOT|'+dataset)[:8] BE",
            GT_only_diagnostic=True))
    with grouped.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]))
        w.writeheader();w.writerows(aggregates)
    r=dict(status="A34_G7_S_MID_GLOBAL2_GT_ONLY_READOUT_PASS",
        head=a.expected_head,original_G7_C_source_sha256=SOURCE_SHA,
        original_C2_task_source_sha256=C2_SOURCE_SHA,
        source_native_controller_state_sha256=sha(
            run/"XP_A34_G7_S_MID_FINAL_STATUS.json"),
        n_images=280,n_ffhq=180,n_imagenet=100,
        n_pilot_images_reused=2,
        per_image_sha256=sha(per),aggregate_sha256=sha(grouped),
        aggregate_rows=aggregates,
        n_upright_kept_by_dataset={
             x["dataset"]:x["n_upright_kept"] for x in aggregates},
        G7_total_gpu_h=s["G7_total_charged_gpu_h"],
        project_total_gpu_h=s["project_charged_gpu_h"],
        no_inverse_problem_solver_run=True,
        GT_only_diagnostic=True,
        remaining_G7_methods_not_done=["S_mid_all8","2_forward","UnivFD"],
        no_new_main_paper_claim=True,
        no_matched_compute_claim=True,no_A21_10_replication=True)
    out=folder/"XP_A23_G7_S_MID_GLOBAL2_READOUT_RECEIPT.json"
    with out.open("x") as f:
        json.dump(r,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("A34_G7_S_MID_GLOBAL2_GT_READOUT_PASS",
          [(x["dataset"],x["n_upright_kept"],x["n"]) for x in aggregates],
          "run",run,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A34_G7_S_MID_READOUT",repr(e),flush=True)
        sys.exit(8)
