#!/usr/bin/env python3
"""CPU-only freeze of the 2026-10-27 overnight GT-free Tier A/B bundle.

Stages: D1-P1 100*4 FAST, D1-P2 100*12 FAST (sealed),
A64 180*48 FAST (sealed), D2 100*4 DAPS-4K.
No GT paths supplied, no scoring, no historical outputs modified.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_t_sitcom_prepare_cpu import (
    COMMON_Y_SHA,IMAGENET,ROOT,WT,BASE,IMAGENET_MODEL_SHA)
from scripts.track_r.xp_t_fast_prepare_cpu import sha as file_sha,R05,R05_SHA
from scripts.track_r.r0_5_fast_gpu import (
    frozen_roots as frozen_historical_roots,CANARY_HEAD,SOURCE_HEAD)
from scripts.track_r.xp_fast_gpu import frozen_roots as frozen_xp_roots
from scripts.track_r.r0_5_prepare_cpu import native_seed
from scripts.track_r.xp_a16_contract import (
    RUN as A16_RUN,TASKS as A16_TASKS,MANIFEST_SHA as A16_SHA,
    PREP_HEAD as A16_HEAD)
from scripts.track_r.xp_t_daps4k_prepare_cpu import DAPS4K,DAPS_HEAD,MODEL_SHA
from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,RECEIPT_DIR,verify_clean_sparse
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as imagenet_seed

NAMESPACE="xp_overnight_"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
STAGES=("D1_P1","D1_P2","A64","D2")
COUNTS={"D1_P1":400,"D1_P2":1200,"A64":8640,"D2":400}

def safely(path):
    p=Path(path)
    real=p.resolve(strict=False)
    if (not p.is_absolute() or p.is_symlink() or not p.is_file()
        or any(x in str(real).lower() or x in str(p).lower() for x in BAD)
        or ROOT.resolve() not in real.parents):
        raise RuntimeError("STOP_NIGHT_FILE_OR_PROTECTED_PATH")
    return p

def write_x(path,value):
    with Path(path).open("x") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def unique_signed_y(common):
    if (common.get("status")!="XP_IMAGENET100_Y_FROZEN"
        or common.get("n_images")!=100
        or common.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_NIGHT_IMAGENET_COMMON")
    source=common.get("rows")
    if (not isinstance(source,list) or len(source)!=100
        or [r["basename"] for r in source]!=
            sorted(r["basename"] for r in source)
        or len({r["basename"] for r in source})!=100):
        raise RuntimeError("STOP_NIGHT_IMAGENET_ORDER")
    for row in source:
        if file_sha(safely(row["measurement_path"]))!=row["measurement_sha256"]:
            raise RuntimeError("STOP_NIGHT_IMAGENET_Y_SHA")
    return source

def construct(imagenet,old,a16):
    old_rows=frozen_historical_roots(old,CANARY_HEAD)
    a16_rows=frozen_xp_roots(a16,A16_HEAD)
    aliases=defaultdict(set)
    historical_seeds=defaultdict(set)
    for r in old_rows+a16_rows:
        key=(r["measurement_source"],str(r["image_id"]))
        aliases[key].add((r["measurement_path"],r["measurement_sha256"]))
        historical_seeds[key].add(r["seed"])
    if (len(old_rows)!=720 or len(a16_rows)!=2160
        or len(aliases)!=180
        or any(len(v)!=1 for v in aliases.values())
        or any(len(historical_seeds[k])!=16 for k in aliases)):
        raise RuntimeError("STOP_NIGHT_A64_HISTORICAL_LINEAGE")
    all_rows={}
    for stage, reps in (("D1_P1",range(0,4)),("D1_P2",range(4,16))):
        rows=[]
        for r in imagenet:
            iid=r["basename"]
            for rep in reps:
                seed=imagenet_seed("XP|FAST|imagenet",iid,rep,2**31)
                rows.append(dict(
                    uid=f"XP|FAST|imagenet|{iid}|rep{rep}",
                    source="XP_D1",measurement_source="XP_IMAGENET100",
                    image_id=iid,rep=rep,seed=seed,
                    measurement_path=r["measurement_path"],
                    measurement_sha256=r["measurement_sha256"]))
        if len(rows)!=COUNTS[stage] or len({r["uid"] for r in rows})!=len(rows):
            raise RuntimeError("STOP_NIGHT_D1_SEEDS_OR_ROWS")
        all_rows[stage]=rows
    rows=[]
    for key in sorted(aliases):
        cohort,iid=key
        path,digest=next(iter(aliases[key]))
        if file_sha(safely(path))!=digest:
            raise RuntimeError("STOP_NIGHT_A64_Y_SHA")
        for rep in range(16,64):
            seed=native_seed(cohort,iid,rep)
            if seed in historical_seeds[key]:
                raise RuntimeError("STOP_NIGHT_A64_SEED_COLLISION")
            rows.append(dict(
                uid=f"S5_r0_fast|FAST|{cohort}_{iid}|rep{rep}",
                source="S5_r0_fast",measurement_source=cohort,
                image_id=iid,rep=rep,seed=seed,
                measurement_path=path,measurement_sha256=digest))
    if len(rows)!=COUNTS["A64"] or len({r["uid"] for r in rows})!=8640:
        raise RuntimeError("STOP_NIGHT_A64_ROOT_COUNT")
    all_rows["A64"]=rows
    rows=[]
    for r in imagenet:
        iid=r["basename"]
        for rep in range(4):
            # Frozen generic new-family ImageNet seed rule:
            # SHA256('XP|DAPS|imagenet|<basename>|<rep>') first8 BE % 2**31.
            seed=imagenet_seed("XP|DAPS|imagenet",iid,rep,2**31)
            rows.append(dict(
                uid=f"XP|DAPS|imagenet|{iid}|rep{rep}",
                family="DAPS",dataset="imagenet",image_id=iid,
                measurement_source="XP_IMAGENET100",rep=rep,seed=seed,
                measurement_path=r["measurement_path"],
                measurement_sha256=r["measurement_sha256"]))
    if len(rows)!=400 or len({r["uid"] for r in rows})!=400:
        raise RuntimeError("STOP_NIGHT_D2_ROOTS")
    all_rows["D2"]=rows
    return all_rows

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_NIGHT_PREP_MUST_BE_CPU")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
            text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
            text=True,timeout=30).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_NIGHT_PREP_GIT")
    if (file_sha(safely(IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"))!=COMMON_Y_SHA
        or file_sha(safely(A16_TASKS))!=A16_SHA
        or file_sha(safely(R05))!=R05_SHA
        or file_sha(safely(ROOT/"models/imagenet256.pt"))!=IMAGENET_MODEL_SHA
        or file_sha(safely(ROOT/"models/ffhq_10m.pt"))!=MODEL_SHA):
        raise RuntimeError("STOP_NIGHT_INPUTS_OR_MODELS_SHA")
    if not verify_clean_sparse(NATIVE):
        raise RuntimeError("STOP_NIGHT_DAPS_NATIVE_SOURCE")
    common=json.loads(safely(IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json").read_text())
    a16=json.loads(safely(A16_TASKS).read_text())
    old=json.loads(safely(R05).read_text())
    rows=construct(unique_signed_y(common),old,a16)
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith(NAMESPACE)
        or out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_NIGHT_PREP_NAMESPACE")
    out.mkdir(parents=True,exist_ok=False)
    stages={}
    for name in STAGES:
        p=out/(name+".json")
        dataset="imagenet" if name in ("D1_P1","D1_P2","D2") else "ffhq"
        meta=dict(
           schema="prdiff.track_r.xp.fast-tasks.v1" if name!="D2" else
                   "prdiff.track_r.xp.daps4k.night.v1",
           status="XP_FAST_TASKS_FROZEN" if name!="D2" else
                  "XP_DAPS4K_TASKS_FROZEN",
           head=head,stage=name,
           fast_head=SOURCE_HEAD if name!="D2" else None,
           source_head=DAPS_HEAD if name=="D2" else None,
           n_roots=COUNTS[name],dataset=dataset,
           gt_paths_supplied=False,oversample=2.0,
           model_sha256=IMAGENET_MODEL_SHA if dataset=="imagenet" else MODEL_SHA,
           outcome_embargo=name in ("D1_P2","A64"),
           n_images=100 if dataset=="imagenet" else 180,
           source_common_y_sha256=COMMON_Y_SHA,
           historical_A16_sha256=A16_SHA,
           rows=rows[name])
        if name=="D2":
            meta.update(family="DAPS",method="DAPS-4K",settings=DAPS4K,
              model_presets={
               "imagenet":{"checkpoint":str(ROOT/"models/imagenet256.pt"),
                           "checkpoint_sha256":IMAGENET_MODEL_SHA,
                           "num_channels":256,"num_res_blocks":2,
                           "attention_resolutions":"32,16,8"}},
              native_source_receipt_sha256=
                 file_sha(safely(RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json")))
        write_x(p,meta)
        stages[name]={"path":str(p),"sha256":file_sha(safely(p)),
                      "rows":COUNTS[name],"outcomes_embargoed":
                          name in ("D1_P2","A64")}
    frozen={
       "status":"XP_NIGHT_INPUTS_FROZEN",
       "head":head,"stages":stages,"total_new_roots":10640,
       "expected_n_images":{"D1_P1":100,"D1_P2":100,"A64":180,"D2":100},
       "shared_imagenet_y_sha256":COMMON_Y_SHA,
       "historical_ffhq_A16_sha256":A16_SHA,
       "native_daps_head":DAPS_HEAD,
       "gt_paths_supplied":False,
       "new_quality_scores_generated":False,
       "prediction_freeze_required_for_scoring_D1_P2_A64":True,
       "gpu_work_performed":False,
    }
    write_x(out/"XP_NIGHT_PREP.json",frozen)
    print("XP_NIGHT_PREP_PASS D1_P1=400 D1_P2=1200 A64=8640 D2=400")
    print("XP_NIGHT_ROOT="+str(out))
    print("XP_NIGHT_PREP_SHA="+file_sha(safely(out/"XP_NIGHT_PREP.json")))
    for key in STAGES:print(key+"_SHA="+stages[key]["sha256"])
    print("GPU_WORK_PERFORMED=false")
    print("SEALED_OUTCOMES_UNSCORED=true")
    print("OK")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:main(ap.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: NIGHT_PREP "+str(e),flush=True)
        sys.exit(8)
