#!/usr/bin/env python3
"""A12 G1b CPU-only prospective FAST ImageNet K64 manifest: runs32..63.

This is a *prepared* conditional stage, not permission to generate.
Admission requires G1a-G5 launched, the campaign projection fitting
the remaining 150 GPU-hour cap, and the planner's fixed A12 priority.
XP-D4 run16+ outcomes stay sealed until prereg v4 embargo release.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import (
 ROOT,BASE,FAST_HEAD,HIST_SHA,COMMON_Y_SHA,
 IMAGENET,IMAGENET_MODEL_SHA,gate)
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as native_seed
from scripts.track_r.xp_a12_g1a_fast_gpu import sha,A12_G1A_FROZEN_PRODUCER
from scripts.track_r.xp_a12_g1a_prepare_cpu import source_contract
FIRST=BASE/"xp_a12_g1a_prepare_20261029T170316Z"
PROD=BASE/"xp_a12_g1a_prod_20261029T180624Z"
def write(path,doc):
    with path.open("x") as h:json.dump(doc,h,indent=2,sort_keys=True);h.write("\n")
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *q:subprocess.check_output(
       ["git","-C",str(wt),*q],text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"G1B_SOURCE")
    p=FIRST/"XP_A12_G1A_1600_TASKS.json"
    first=json.loads(p.read_text())
    gate(first.get("head")==A12_G1A_FROZEN_PRODUCER and
         len(first["rows"])==1600 and
         sha(p)=="acb38367dcda4e1bd657e03b5f365be65cad725413b4965c6e42ab86d241cfd0",
         "G1A_FROZEN_SOURCE")
    result=json.loads((PROD/"XP_A12_G1A_FINAL_STATUS.json").read_text())
    gate(result["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
         and result["completed"]==1600 and result["failed_attempts"]==0
         and result["outcome_sealed"] is True,"G1A_GENERATION_PASS")
    _,_,common,history=source_contract()
    rows=[];seeds=set()
    for iid in sorted(common):
        y=common[iid]
        old=history[iid]
        previously={int(x["seed"]) for x in old}
        # More G1a seeds are fixed by the original frozen manifest.
        g1a=[x for x in first["rows"] if x["image_id"]==iid]
        gate(len(g1a)==16 and
             [x["rep"] for x in g1a]==list(range(16,32)),
             "G1A_IMAGE_16")
        previously.update(x["seed"] for x in g1a)
        for rep in range(32,64):
            seed=native_seed("XP|FAST|imagenet",iid,rep,2**31)
            key=(iid,seed)
            gate(seed not in previously and key not in seeds,"SEED_COLLISION")
            seeds.add(key)
            rows.append(dict(uid=f"XP|FAST|imagenet|{iid}|rep{rep}",
                source="XP_D1",measurement_source="XP_IMAGENET100",
                image_id=iid,rep=rep,seed=seed,
                measurement_path=y["measurement_path"],
                measurement_sha256=y["measurement_sha256"]))
    gate(len(rows)==3200 and len(seeds)==3200,"G1B_3200")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a12_g1b_prepare_")
         and not out.exists() and not out.is_symlink(),"G1B_NEW_OUTPUT")
    out.mkdir(mode=0o700)
    manifest=dict(schema="prdiff.track_r.xp.fast-tasks.v1",
      status="XP_FAST_TASKS_FROZEN",stage="A12_G1B_FAST_IMAGENET_D4",
      head=a.expected_head,fast_head=FAST_HEAD,
      dataset="imagenet",oversample=2.0,model_sha256=IMAGENET_MODEL_SHA,
      n_roots=3200,n_images=100,gt_paths_supplied=False,
      outcome_embargo=True,output_scores_sealed=True,
      seed_rule="XP|FAST|imagenet|<basename>|<rep>, first8 SHA256 mod2**31",
      G1a_producer=A12_G1A_FROZEN_PRODUCER,
      G1a_source_sha256=sha(p),rows=rows)
    mf=out/"XP_A12_G1B_3200_TASKS.json";write(mf,manifest)
    receipt=dict(status="XP_A12_G1B_CONDITIONAL_PREPARED_CPU_PASS",
       head=a.expected_head,manifest_sha256=sha(mf),n_roots=3200,
       no_GPU=True,stage_ceiling_gpu_h=40,
       G1a_complete_sha256=sha(PROD/"XP_A12_G1A_FINAL_STATUS.json"),
       G2_G3_G4_G5_must_have_launched=True,
       A12_campaign_projection_must_fit=True,
       G1b_not_yet_admitted=True,
       no_GT_read=True,no_D4_outcomes_scored=True)
    write(out/"XP_A12_G1B_PREP_RECEIPT.json",receipt)
    print("XP_A12_G1B_3200_CONDITIONAL_PREP_PASS",flush=True)
    print("G1B_MANIFEST_SHA",sha(mf),flush=True)
    print("G1B_PREP_DIR",out,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc();print("ACTION NEEDED: A12_G1B_PREP",e,flush=True);sys.exit(8)
