#!/usr/bin/env python3
"""A12 G4/G5 GT-free DAPS-4K historical continuation and two-root replays.

G4: EPRIME FFHQ 180x4, reps4..7. Historical A6 EPRIME0..3.
G5: D2 ImageNet 100x4, reps4..7. Historical night D2 reps0..3.
Only frozen measurements and native source/settings; no GT/outcomes.
A replay of two historical roots for EACH stage is mandatory before
production. This script ONLY freezes input manifests and receipts.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
 DAPS4K,DAPS_HEAD,MODEL_SHA,IMAGENET_MODEL_SHA,
 NATIVE,verify_clean_sparse)
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as img_seed
A6=BASE/"xp_a6_20261027T165112Z"
NIGHT=BASE/"xp_overnight_20261027T045215Z"
STAGES={
 "G4":("EPRIME",A6,180,720,"ffhq",30),
 "G5":("D2",NIGHT,100,400,"imagenet",40)}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()
def write(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def a6_seed(image,rep):
    key=f"XP|DAPS|ffhq_daps4k|{image}|{rep}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")%(2**31)
def seed(stage,iid,rep):
    if stage=="G4":return a6_seed(iid,rep)
    if stage=="G5":return img_seed("XP|DAPS|imagenet",iid,rep,2**31)
    raise ValueError(stage)
def prep(stage):
    name,old,n_images,n_new,dataset,cap=STAGES[stage]
    mfpath=old/(name+".json")
    source=json.loads(mfpath.read_text())
    gate(source["stage"]==name and source["n_roots"]==n_images*4
         and source["status"] in ("XP_EPRIME_TASKS_FROZEN",
                                 "XP_DAPS4K_TASKS_FROZEN")
         and source["gt_paths_supplied"] is False
         and source["settings"]==DAPS4K
         and source["source_head"]==DAPS_HEAD,
         "HISTORICAL_MANIFEST_"+stage)
    oldrows=source["rows"]
    grouped=defaultdict(list)
    for r in oldrows:
        grouped[r["image_id"]].append(r)
    gate(len(grouped)==n_images and all(
        sorted(r["rep"] for r in v)==[0,1,2,3] for v in grouped.values()),
        "HISTORICAL_4RUN_PANEL_"+stage)
    rows=[];seedsets={}
    for iid in sorted(grouped):
        old=grouped[iid]
        y={(x["measurement_path"],x["measurement_sha256"]) for x in old}
        gate(len(y)==1,"Y_UNIQUE_"+stage)
        ypath,ydigest=next(iter(y))
        gate(sha(ypath)==ydigest,"Y_SOURCE_DRIFT_"+stage)
        seedsets[iid]={r["seed"] for r in old}
        gate(len(seedsets[iid])==4 and all(
             r["seed"]==seed(stage,iid,r["rep"]) for r in old),
             "HISTORICAL_SEED_DOMAIN_"+stage)
        for rep in range(4,8):
            z=seed(stage,iid,rep)
            gate(z not in seedsets[iid],"NEW_SEED_COLLISION_"+stage)
            seedsets[iid].add(z)
            uid=f"XP|DAPS|ffhq_daps4k|{iid}|rep{rep}" if stage=="G4" else (
                f"XP|DAPS|imagenet|{iid}|rep{rep}")
            rows.append(dict(uid=uid,family="DAPS",dataset=dataset,
              image_id=iid,measurement_source=old[0]["measurement_source"],
              measurement_path=ypath,measurement_sha256=ydigest,
              rep=rep,seed=z))
    gate(len(rows)==n_new and len({r["uid"] for r in rows})==n_new,
         "NEW_ROOT_GRID_"+stage)
    replay=[sorted(grouped[iid],key=lambda x:x["rep"])[0]
            for iid in sorted(grouped)[:2]]
    gate(len(replay)==2 and replay[0]["image_id"]!=replay[1]["image_id"],
         "REPLAY2_DISTINCT_IMAGES_"+stage)
    return source,mfpath,rows,replay,cap
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                  text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"WORKTREE")
    gate(verify_clean_sparse(NATIVE),"DAPS4K_PINNED_PRISTINE_SOURCE")
    stages={};out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a12_g45_prepare_")
         and not out.exists() and not out.is_symlink(),"FRESH_OUTPUT")
    out.mkdir(mode=0o700)
    for stage in ("G4","G5"):
        old,oldpath,new,replay,cap=prep(stage)
        for flavor,rows in (("PROD",new),("REPLAY",replay)):
            doc=dict(schema="prdiff.track_r.xp.daps4k.a12.v1",
                status="XP_A12_DAPS4K_TASKS_FROZEN",stage=stage,
                phase=flavor,head=a.expected_head,
                source_head=DAPS_HEAD,n_roots=len(rows),
                n_images=len({r["image_id"] for r in rows}),
                dataset="ffhq" if stage=="G4" else "imagenet",
                method="DAPS-4K",
                settings=DAPS4K,model_presets=old["model_presets"],
                historical_manifest_sha256=sha(oldpath),
                original_historical_run=str(oldpath.parent),
                gt_paths_supplied=False,outcome_embargo=False,
                source_native_pristine=True,
                rows=rows)
            dest=out/f"XP_A12_{stage}_{flavor}_TASKS.json"
            write(dest,doc)
            stages[f"{stage}_{flavor}"]=dict(
                 manifest=str(dest),sha256=sha(dest),n=len(rows),
                 cap_gpu_h=cap,first2_replay_required=True)
    write(out/"XP_A12_G45_PREP_RECEIPT.json",dict(
        status="XP_A12_G45_INPUTS_FROZEN_CPU_PASS",head=a.expected_head,
        all_replays_pending=True,
        n_new=1120,n_replay=4,no_GPU=True,no_GT_read=True,
        stages=stages))
    print("XP_A12_G45_CPU_PREP_PASS G4=720 G5=400 replay=2+2",flush=True)
    for k,v in stages.items():print(k,"SHA",v["sha256"],flush=True)
    print("G45_PREP_DIR",out,flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    try:main(parser.parse_args())
    except Exception as e:
        traceback.print_exc();print("ACTION NEEDED: A12_G45_PREP",e,flush=True);sys.exit(8)
