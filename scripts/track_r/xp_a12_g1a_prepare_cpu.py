#!/usr/bin/env python3
"""A12 G1a FAST ImageNet 100x16 new roots, runs 16..31, GT-free CPU freeze.

Only original D1P2 signed-y and run-seed domain. Two historical D1
rep4 roots from *different* images are additionally frozen for a
mandatory native 2-root replay before G1a production.
No D4 outcome labels, PSNR or GT ever read in preparation.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from scripts.track_r.xp_a6_d1p2_contract import (
  D1P2,SHA as HIST_SHA,SOURCE_HEAD as HIST_HEAD,validate as validate_hist)
from scripts.track_r.xp_t_sitcom_prepare_cpu import (
  IMAGENET,COMMON_Y_SHA,IMAGENET_MODEL_SHA)
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as native_seed
from scripts.track_r.xp_fast_gpu import SOURCE_HEAD as FAST_HEAD
from scripts.track_r.xp_t_fast_prepare_cpu import sha
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def gate(ok,name):
    if not ok:raise RuntimeError("STOP_A12_G1A_"+name)
def write_x(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def source_contract():
    gate(sha(D1P2)==HIST_SHA,"ORIGINAL_D1P2_SHA")
    hist=json.loads(D1P2.read_text())
    old=hist["rows"]
    validate_hist(D1P2,HIST_SHA,hist,old,HIST_HEAD,0,1200)
    path=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
    gate(sha(path)==COMMON_Y_SHA,"ORIGINAL_IMAGENET_Y_MANIFEST_SHA")
    orig=json.loads(path.read_text())
    gate(orig["status"]=="XP_IMAGENET100_Y_FROZEN"
         and orig["n_images"]==100 and len(orig["rows"])==100
         and orig["gt_paths_supplied"] is False,"SHARED_Y_SOURCE")
    table={r["basename"]:r for r in orig["rows"]}
    gate(len(table)==100,"UNIQUE_IMAGE_IDS")
    by=defaultdict(list)
    for row in old:by[row["image_id"]].append(row)
    gate(set(table)==set(by) and
         all([r["rep"] for r in group]==list(range(4,16))
             for group in by.values()),"OLD_RUN_RANGE")
    return path,orig,table,by
def new_mf(a,rows,stage):
    return dict(schema="prdiff.track_r.xp.fast-tasks.v1",
      status="XP_FAST_TASKS_FROZEN",
      head=a.expected_head,fast_head=FAST_HEAD,
      stage=stage,dataset="imagenet",oversample=2.,
      model_sha256=IMAGENET_MODEL_SHA,
      n_roots=len(rows),n_images=len({x["image_id"] for x in rows}),
      gt_paths_supplied=False,
      outcome_embargo=True,output_scores_sealed=True,
      seed_rule="XP|FAST|imagenet|<basename>|<rep> first8 SHA256 unsigned BE mod2**31",
      original_historical_manifest_sha256=HIST_SHA,
      common_y_manifest_sha256=COMMON_Y_SHA,
      rows=rows)
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    proc=lambda *xs:subprocess.check_output(
         ["git","-C",str(WT),*xs],text=True,timeout=60)
    gate(proc("rev-parse","HEAD").strip()==a.expected_head
         and not proc("status","--porcelain"),"WORKTREE_HEAD")
    gate(sha(ROOT/"models/imagenet256.pt")==IMAGENET_MODEL_SHA,
         "IMAGENET_MODEL_SHA")
    yfile,common,table,prior=source_contract()
    new=[];replay=[]
    for iid in sorted(table):
        old=prior[iid];m=table[iid]
        previous=set()
        for row in old:
            rep=int(row["rep"])
            expected=native_seed("XP|FAST|imagenet",iid,rep,2**31)
            gate(row["seed"]==expected,"HIST_SEED_DOMAIN")
            previous.add(expected)
        allseeds=[
          native_seed("XP|FAST|imagenet",iid,r,2**31)
          for r in range(32)]
        gate(len(set(allseeds))==32
             and set(allseeds[16:]).isdisjoint(previous),
             "NO_SEED_COLLISIONS_"+iid)
        gate((m["measurement_path"],m["measurement_sha256"])==
             (old[0]["measurement_path"],old[0]["measurement_sha256"]),
             "Y_IDENTITY_"+iid)
        gate(sha(m["measurement_path"])==m["measurement_sha256"],
             "Y_BYTES_"+iid)
        for rep in range(16,32):
            new.append(dict(uid=f"XP|FAST|imagenet|{iid}|rep{rep}",
              source="XP_D1",measurement_source="XP_IMAGENET100",
              image_id=iid,rep=rep,seed=allseeds[rep],
              measurement_path=m["measurement_path"],
              measurement_sha256=m["measurement_sha256"]))
        if len(replay)<2:
            # Distinct image IDs; same native input/seed as D1 phase2.
            oldrep=old[0]
            gate(oldrep["rep"]==4,"REPLAY_REP4")
            replay.append(dict(oldrep))
    gate(len(new)==1600 and len(replay)==2
         and len({r["image_id"] for r in replay})==2
         and len({r["uid"] for r in new+replay})==1602,
         "G1A_GRID")
    allserialized=json.dumps(new+replay)
    gate(not any(x in allserialized.lower() for x in BAD+(
       "gt_path","candidate_path","psnr","labels","oracle")),
       "GT_OR_OUTCOME_LEAK")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a12_g1a_prepare_")
         and not out.exists() and not out.is_symlink(),
         "NEW_RUN_DIR")
    out.mkdir(mode=0o700)
    repfile=out/"XP_A12_G1A_REPLAY2_TASKS.json"
    mf=out/"XP_A12_G1A_1600_TASKS.json"
    write_x(repfile,new_mf(a,replay,"A12_G1A_HISTORICAL_REPLAY2"))
    write_x(mf,new_mf(a,new,"A12_G1A_FAST_IMAGENET_D4"))
    receipt=out/"XP_A12_G1A_PREP_RECEIPT.json"
    write_x(receipt,dict(
      status="XP_A12_G1A_GT_FREE_PREP_PASS",
      head=a.expected_head,n_images=100,n_replay=2,n_new=1600,
      original_D1P2_SHA=HIST_SHA,original_D1P2_head=HIST_HEAD,
      shared_y_sha256=COMMON_Y_SHA,
      imagenet_model_sha256=IMAGENET_MODEL_SHA,
      native_fast_head=FAST_HEAD,
      replay_manifest_sha256=sha(repfile),
      production_manifest_sha256=sha(mf),
      G1a_ceiling_gpu_h=25,
      A12_campaign_ceiling_gpu_h=150,
      original_project_cap_gpu_h=400,
      cumulative_project_ledger_required_before_GPU_admission=True,
      historical_two_root_replay_required_before_GPU_production=True,
      D4_outcomes_sealed_until="theory_v4_register_or_decline_OR_2026-11-02T12:00:00-05:00",
      G1a_deadline_et="2026-11-03T12:00:00-05:00",
      no_new_gpu_work_launched=True,no_GT_read=True,
      no_D4_outcome_scored=True))
    print("XP_A12_G1A_CPU_PREP_PASS",
          "n_replay",2,"n_new",1600,flush=True)
    print("G1A_PREP_DIR="+str(out),flush=True)
    print("G1A_REPLAY2_SHA="+sha(repfile),flush=True)
    print("G1A_1600_MANIFEST_SHA="+sha(mf),flush=True)
    print("G1A_FIRST_A12_GPU_RECEIPT_AWAITS_PROJECT_LEDGER=true",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A12_G1A_PREP "+str(e),flush=True)
        sys.exit(8)
