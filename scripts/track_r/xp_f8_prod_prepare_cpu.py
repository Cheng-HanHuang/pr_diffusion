#!/usr/bin/env python3
"""Freeze embargo-safe XP-F8 920-row historical-lineage production manifest.

F8 = B22 reps 2..7 (100*6) + DEV80 reps 4..7 (80*4).
Only the 180 originally exposed R0 measurements. CPU reads the R0.7
catalog's metadata, never opens GT or any new F8 reconstruction.
The eight-root 400x5 EXACT replay gate is mandatory.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,os,sys,traceback
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
 ROOT,BASE,CAT,B21_MANIFEST,DEV80_MANIFEST,DEV80_RUN,
 DEV80_SEED_DOMAIN,SCREEN_SHA,ORIG_DAPS,DAPS_HEAD,MODEL_SHA,MODEL,
 B16,A10_DEADLINE,read_json,sha,git,fail,dev80_table,WT)

GATE=BASE/"xp_f8_legacy_replay_20261029T065122Z"
GATE_SHA={
 "XP_F8_LEGACY_REPLAY_WAVE1_OFFLINE.json":
 "45e69880b04adb3a8762a940b4cb3630a4f055ec7784368ef9d891476320b1da",
 "XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json":
 "ab34db50115ee7daef5b55ac81b321b47a318b67028ab21cda8a92b9e52d0385"}
B16_COMPLETE_SHA="5a3426464fb1164d98138bfbe52212fe38036f4401cf24c1379b6a283ac9e88d"
A10_LIMIT=70.
def write_x(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
def source_gate():
    fail(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD,"DAPS_HEAD_DRIFT")
    fail(sha(MODEL)==MODEL_SHA,"MODEL_SHA_DRIFT")
    for name,digest in GATE_SHA.items():
        p=GATE/name
        fail(sha(p)==digest and read_json(p)["status"]==
             "XP_F8_LEGACY_WAVE"+("1" if "WAVE1" in name else "2")+"_OFFLINE_PASS",
             "EIGHT_ROOT_GATE_RECEIPT_"+name)
    fail(read_json(GATE/"XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json")["n_pass"]==8,
         "EIGHT_ROOT_GATE_PASS_COUNT")
    fail(sha(B16/"XP_A11_B16_PROD_COMPLETE.json")==B16_COMPLETE_SHA,
         "B16_COMPLETE_SHA")
    base=read_json(B16/"B16_PRODUCTION_STATE.json")
    fail(base["status"]=="A10_B16_GENERATION_COMPLETE_WAIT_F8" and
         base["active"]==[] and base["A10_limit_gpu_hours"]==A10_LIMIT,
         "B16_FINAL_STATE")
    consumed=(base["gate_spent_gpu_seconds"]+base["spent_gpu_seconds"])/3600
    ledger=[]
    for p in sorted(BASE.glob(
      "xp_f8_legacy_replay_*/XP_F8_LEGACY_REPLAY_WAVE[12].json")):
        a=read_json(p)
        h=a["incremental_gpu_h"]
        fail(a["wave"] in (1,2) and a["native_400x5"] is True
             and isinstance(h,(float,int)) and 0<=h<70,
             "LEGACY_LEDGER_RECEIPT_"+str(p))
        ledger.append(dict(path=str(p),sha256=sha(p),gpu_h=h))
        consumed+=h
    fail(consumed<A10_LIMIT and
        datetime.datetime.now(ZoneInfo("America/Detroit"))<A10_DEADLINE,
        "A10_DEADLINE_OR_BUDGET")
    return dict(cap_gpu_h=A10_LIMIT,prior_gpu_h=consumed,
                remaining_gpu_h=A10_LIMIT-consumed,
                deadline_et=A10_DEADLINE.isoformat(),
                replay_gate_sha256=GATE_SHA,replay_ledger=ledger,
                b16_receipt_sha256=B16_COMPLETE_SHA)
def r07_measurements():
    old=read_json(CAT)
    fail(old.get("status")=="R0_7_GT_OFFLINE_CATALOG_PASS"
         and len(old["rows"])==3240,"R07_CATALOG")
    grouped=defaultdict(list)
    for r in old["rows"]:
        if (r["source"],r["family"]) in (
            ("S1_b22","DAPS"),("S2_b24_dev80","DAPS")):
            grouped[(r["source"],str(r["image_id"]).zfill(5))].append(r)
    fail(len([k for k in grouped if k[0]=="S1_b22"])==100
         and len([k for k in grouped if k[0]=="S2_b24_dev80"])==80,
         "R07_HISTORICAL_COHORTS")
    result={}
    for (cohort,iid),rr in grouped.items():
        rr.sort(key=lambda x:int(x["run_index"]))
        n=2 if cohort=="S1_b22" else 4
        fail([int(x["run_index"]) for x in rr]==list(range(n)),
             "R07_BASELINE_GRID_"+iid)
        m0=rr[0]
        fail(all(x["measurement_sha256"]==m0["measurement_sha256"]
                 and x["measurement_path"]==m0["measurement_path"]
                 for x in rr),"HIST_Y_IDENTITY_"+iid)
        fail(sha(m0["measurement_path"])==m0["measurement_sha256"],
             "HIST_Y_SHA_"+iid)
        result[(cohort,iid)]=dict(
            measurement_path=m0["measurement_path"],
            measurement_sha256=m0["measurement_sha256"])
    return result
def manifest_rows(meas):
    # B22: continuation of Fresh2 22000+row, 23000+row:
    # run r uses (22000 + 1000*r + frozen B21 panel row ID), r=2..7.
    with B21_MANIFEST.open(newline="") as f:
        hist=list(csv.DictReader(f,delimiter="\t"))
    fail(len(hist)==100,"B21_100")
    b21={}
    for r in hist:
        iid=str(r["image_id"]).zfill(5)
        idx=int(r["row_id"])
        fail(iid not in b21 and int(r["seed1"])==22000+idx
             and int(r["seed2"])==23000+idx,
             "B21_SEED_LINEAGE")
        b21[iid]=r
    dev=dev80_table()
    fail(len(dev)==80,"DEV80_80")
    tasks=[]
    for cohort,ids,extension in (
        ("S1_b22",sorted(b21),range(2,8)),
        ("S2_b24_dev80",sorted(dev),range(4,8))):
        for iid in ids:
            m=meas[(cohort,iid)]
            if cohort=="S1_b22":
                row=b21[iid]
                fail(row["measurement_path"]==m["measurement_path"],
                     "B21_MEAS_PATH_"+iid)
                canonical=[22000+1000*rep+int(row["row_id"])
                           for rep in range(8)]
                rule="B21_FRESH2_22000_PLUS_1000_REP_PLUS_ROW_ID"
            else:
                t=dev[iid]
                label=t["class_label"]
                canonical=[]
                for rep in range(8):
                    parts=(DEV80_SEED_DOMAIN,SCREEN_SHA,label,iid,str(rep))
                    val=int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16],16)&((1<<63)-1)
                    canonical.append(val)
                fail(canonical[:4]==[int(x) for x in t["daps_solver_seeds"]],
                     "DEV80_FROZEN_SEED_PREFIX_"+iid)
                rule="B24_DEV80_DAPS_SOLVER_V1_SEED63_MOD2POW32"
            native=[int(x)%(2**32) for x in canonical]
            fail(len(set(native))==8,"EIGHT_SEED_COLLISION_"+cohort+"_"+iid)
            for rep in extension:
                tasks.append(dict(
                   index=len(tasks),cohort=cohort,image_id=iid,rep=rep,
                   native_seed=native[rep],canonical_seed=canonical[rep],
                   seed_rule=rule,
                   measurement_path=m["measurement_path"],
                   measurement_sha256=m["measurement_sha256"],
                   uid="XP_F8_DAPS_"+cohort+"_"+iid+"_r"+str(rep),
                   annealing_steps=400,diffusion_steps=5))
    fail(len(tasks)==920 and
         sum(t["cohort"]=="S1_b22" for t in tasks)==600 and
         sum(t["cohort"]=="S2_b24_dev80" for t in tasks)==320 and
         len({t["uid"] for t in tasks})==920,"F8_920_TASK_GRID")
    # No GT, historical candidate, confirmation or bundled FFHQ path in
    # the serialised GPU-facing task manifest.
    serial=json.dumps(tasks,sort_keys=True)
    fail(not any(x in serial for x in (
       "ground_truth","gt_path","candidate_path",
       "confirmation305","test-ffhq","demo-ffhq")),
       "F8_GT_FREE_MANIFEST")
    return tasks
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--expected-head",required=True)
    a.add_argument("--out",required=True)
    args=a.parse_args()
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_PREP_ONLY")
    fail(git(WT,"rev-parse","HEAD")==args.expected_head and
         not git(WT,"status","--porcelain"),"WORKTREE_HEAD")
    accounting=source_gate()
    meas=r07_measurements()
    tasks=manifest_rows(meas)
    out=Path(args.out)
    fail(out.parent==BASE and out.name.startswith("xp_f8_prod_")
         and not out.exists() and not out.is_symlink(),
         "NEW_OUTPUT_REQUIRED")
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    manifest=out/"XP_F8_PROD_TASKS.json"
    write_x(manifest,dict(schema="prdiff.track_r.xp_f8.native_prod.v1",
      status="XP_F8_PROD_MANIFEST_GT_FREE_FROZEN",head=args.expected_head,
      n_images=180,n_b22=600,n_dev80=320,n_total=920,
      no_GT_loaded=True,no_new_outcome_opened=True,
      no_historical_candidates_supplied_to_worker=True,
      historical_native_daps_head=DAPS_HEAD,
      historical_checkpoint_sha256=MODEL_SHA,
      corrected_DEV80_seed_domain=DEV80_SEED_DOMAIN,
      historical_config="400x5_pixel_langevin_100",
      source_R07_catalog_sha256=sha(CAT),
      source_B21_manifest_sha256=sha(B21_MANIFEST),
      source_DEV80_manifest_sha256=sha(DEV80_MANIFEST),
      replay_gate=GATE_SHA,rows=tasks))
    prepare=out/"XP_F8_PROD_PREP.json"
    write_x(prepare,dict(
       status="XP_F8_920_GT_FREE_PRODUCTION_PREP_PASS",
       head=args.expected_head,n_total=920,n_images=180,
       manifest_sha256=sha(manifest),accounting=accounting,
       outcome_embargo="A9.4 until v3 registered/declined or 2026-10-31T12:00:00-04:00",
       only_generation_authorized=True,no_new_gpu_jobs_launched=True))
    print("XP_F8_920_PRODUCTION_PREP_PASS 600 B22 320 DEV80",flush=True)
    print("F8_PROD_RUN="+str(out),flush=True)
    print("F8_PROD_MANIFEST_SHA="+sha(manifest),flush=True)
    print("F8_PROD_PREP_SHA="+sha(prepare),flush=True)
    print("SHARED_GPU_H_REMAINING="+str(round(accounting["remaining_gpu_h"],6)),flush=True)
if __name__=="__main__":
    try:main()
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_PROD_PREP "+str(e),flush=True)
        sys.exit(8)
