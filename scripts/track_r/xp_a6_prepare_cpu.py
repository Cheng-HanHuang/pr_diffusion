#!/usr/bin/env python3
"""A6 next-batch CPU-only GT-free D1 phase-2 / SITCOM D3 / DAPS E' freeze.

Uses the *original* D1_P2 manifest by SHA (no regeneration of inputs).
D3 gets 100 shared ImageNet y x 4 independent native SITCOM seeds.
E' gets 180 original FFHQ signed y x 4 new-family DAPS-4K seeds;
different population than legacy DAPS 400x5; no replay extension claim.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_a6_d1p2_contract import (
    ROOT,BASE,D1P2,SHA as D1_SHA,SOURCE_HEAD as D1_HEAD)
from scripts.track_r.xp_t_sitcom_prepare_cpu import (
    WT,IMAGENET,COMMON_Y_SHA,IMAGENET_MODEL_SHA,MODEL_SHA,
    SITCOM,SITCOM_HEAD,seed as sitcom_seed,check_native_checkout)
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
    DAPS4K,DAPS_HEAD)
from scripts.track_r.xp_t_daps4k_native_cpu import (
    NATIVE,RECEIPT_DIR,verify_clean_sparse)
from scripts.track_r.xp_a16_contract import (
    TASKS as A16_TASKS,MANIFEST_SHA as A16_SHA,PREP_HEAD as A16_HEAD)
from scripts.track_r.xp_fast_gpu import frozen_roots
from scripts.track_r.xp_t_fast_prepare_cpu import sha

SITCOM_TIMING=BASE/"xp_t_sitcom_20261026T233451Z"/"XP_T_SITCOM_TASKS.json"
SOURCE_R07=BASE/"r0_5_canary_20261023T004927Z/prepare/R0_5_GT_FREE_ROOTS.json"
FAST_SOURCE_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"
COUNTS={"D1_P2":1200,"D3":400,"EPRIME":720}
SITCOM_PROVENANCE="XP_T_SITCOM_TASKS_FROZEN"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def check_file(p):
    p=Path(p)
    if (p.is_symlink() or not p.is_file()
        or ROOT.resolve() not in p.resolve(strict=False).parents
        or any(b in str(p.resolve(strict=False)).lower() for b in BAD)):
        raise RuntimeError("STOP_A6_MISSING_OR_PROTECTED_SOURCE")
    return p

def write_x(p,data):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def build(common,a16):
    images=common.get("rows",[])
    if (common.get("status")!="XP_IMAGENET100_Y_FROZEN"
        or common.get("n_images")!=100 or
        common.get("gt_paths_supplied") is not False or
        len(images)!=100 or
        [x["basename"] for x in images]!=sorted(x["basename"] for x in images)):
        raise RuntimeError("STOP_A6_IMAGENET_SOURCE")
    image_ids=set()
    d3=[]
    for item in images:
        iid=item["basename"]
        if iid in image_ids:raise RuntimeError("STOP_A6_DUPLICATE_IMAGENET_ID")
        image_ids.add(iid)
        if sha(check_file(item["measurement_path"]))!=item["measurement_sha256"]:
            raise RuntimeError("STOP_A6_IMAGENET_Y_SHA")
        seen=set()
        for rep in range(4):
            seed=sitcom_seed("imagenet",iid,rep)
            if seed in seen:raise RuntimeError("STOP_A6_D3_SEED_COLLISION")
            seen.add(seed)
            d3.append(dict(uid=f"XP|SITCOM|imagenet|{iid}|rep{rep}",
                family="SITCOM",dataset="imagenet",image_id=iid,rep=rep,
                seed=seed,measurement_source="XP_IMAGENET100",
                measurement_path=item["measurement_path"],
                measurement_sha256=item["measurement_sha256"]))
    old=frozen_roots(a16,A16_HEAD)
    ff=defaultdict(list)
    for row in old:
        ff[(row["measurement_source"],str(row["image_id"]))].append(row)
    if (len(ff)!=180 or sum(k[0]=="S1_b22" for k in ff)!=100
        or sum(k[0]=="S2_b24_dev80" for k in ff)!=80):
        raise RuntimeError("STOP_A6_EPRIME_FFHQ_IDS")
    e=[]
    for (cohort,iid),group in sorted(ff.items()):
        if ([r["rep"] for r in group]!=list(range(4,16))
            or len({(r["measurement_path"],r["measurement_sha256"]) for r in group})!=1):
            raise RuntimeError("STOP_A6_EPRIME_FROZEN_Y_IDENTITY")
        y,digest=group[0]["measurement_path"],group[0]["measurement_sha256"]
        if sha(check_file(y))!=digest:
            raise RuntimeError("STOP_A6_EPRIME_Y_SHA")
        # New population, seed domain *explicitly* ffhq_daps4k,
        # UID stable on B22/DEV80 as an original measurement cohort.
        uid=f"{cohort}_{iid}"
        seeds=set()
        for rep in range(4):
            key=f"XP|DAPS|ffhq_daps4k|{uid}|{rep}"
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")%(2**31)
            if seed in seeds:raise RuntimeError("STOP_A6_EPRIME_SEED_COLLISION")
            seeds.add(seed)
            e.append(dict(uid=f"XP|DAPS|ffhq_daps4k|{uid}|rep{rep}",
                          family="DAPS",dataset="ffhq",image_id=uid,
                          rep=rep,seed=seed,measurement_source=cohort,
                          measurement_path=y,measurement_sha256=digest))
    if len(d3)!=400 or len(e)!=720 or len({r["uid"] for r in e+d3})!=1120:
        raise RuntimeError("STOP_A6_FROZEN_COUNTS")
    return d3,e

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A6_PREP_GPU_VISIBLE")
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if actual!=a.expected_head or dirty:
        raise RuntimeError("STOP_A6_PREP_GIT")
    if (sha(check_file(D1P2))!=D1_SHA or sha(check_file(A16_TASKS))!=A16_SHA
        or sha(check_file(IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"))!=COMMON_Y_SHA
        or sha(check_file(ROOT/"models/imagenet256.pt"))!=IMAGENET_MODEL_SHA
        or sha(check_file(ROOT/"models/ffhq_10m.pt"))!=MODEL_SHA):
        raise RuntimeError("STOP_A6_PREP_MODEL_OR_MANIFEST_SHA")
    common=json.loads((IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json").read_text())
    a16=json.loads(A16_TASKS.read_text())
    d3,e=build(common,a16)
    frozen_sitcom=json.loads(check_file(SITCOM_TIMING).read_text())
    if (frozen_sitcom.get("status")!=SITCOM_PROVENANCE
        or frozen_sitcom.get("native_source_head")!=SITCOM_HEAD
        or frozen_sitcom.get("n_roots")!=18
        or frozen_sitcom.get("sampler_config",{}).get("diff_steps")!=5
        or frozen_sitcom.get("sampler_config",{}).get("lgvd_steps")!=100):
        raise RuntimeError("STOP_A6_SITCOM_TIMING_SOURCE")
    native_head=subprocess.check_output(["git","-C",str(SITCOM),"rev-parse","HEAD"],text=True).strip()
    state=check_native_checkout(native_head,
         subprocess.check_output(["git","-C",str(SITCOM),"status","--porcelain=v1",
             "--untracked-files=no"],text=True),
         subprocess.check_output(["git","-C",str(SITCOM),"diff","--numstat","HEAD",
                                 "--","posterior_sample.py"],text=True))
    if (state["tracked_modified"]!=frozen_sitcom["external_tracked_diff_allowlist"]
        or sha(check_file(SITCOM/"posterior_sample.py"))!=
           frozen_sitcom["external_unused_cli_sha256"]):
        raise RuntimeError("STOP_A6_SITCOM_NATIVE_SOURCE_DRIFT")
    daps_files=verify_clean_sparse(NATIVE)
    if not daps_files or not isinstance(daps_files,dict):
        raise RuntimeError("STOP_A6_DAPS_NATIVE_SOURCE")
    d1=json.loads(D1P2.read_text())
    if (d1.get("status")!="XP_FAST_TASKS_FROZEN"
        or d1.get("head")!=D1_HEAD
        or d1.get("stage")!="D1_P2"
        or d1.get("outcome_embargo") is not True
        or d1.get("n_roots")!=1200):
        raise RuntimeError("STOP_A6_D1P2_SOURCE")
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith("xp_a6_")
        or out.is_symlink() or out.exists()):
        raise RuntimeError("STOP_A6_OUTPUT_NAMESPACE")
    out.mkdir(parents=True,exist_ok=False)
    d3_mf=dict(schema="prdiff.track_r.xp.sitcom.a6.v1",
       status="XP_D3_TASKS_FROZEN",head=actual,stage="D3",
       family="SITCOM",dataset="imagenet",n_roots=400,n_images=100,
       runs_per_image=4,native_source_head=SITCOM_HEAD,
       sampler_config=frozen_sitcom["sampler_config"],
       model_presets=frozen_sitcom["model_presets"],
       external_tracked_diff_allowlist=frozen_sitcom["external_tracked_diff_allowlist"],
       external_unused_cli_sha256=frozen_sitcom["external_unused_cli_sha256"],
       gt_paths_supplied=False,historical_paths_supplied=False,
       outcomes_embargoed=False,
       seed_rule="xp_t_sitcom_prepare_cpu.seed(imagenet,image,rep)",
       rows=d3)
    e_mf=dict(schema="prdiff.track_r.xp.daps4k.a6.v1",
       status="XP_EPRIME_TASKS_FROZEN",head=actual,stage="EPRIME",
       family="DAPS",dataset="ffhq",n_roots=720,n_images=180,
       runs_per_image=4,source_head=DAPS_HEAD,
       native_source_receipt_sha256=sha(check_file(RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json")),
       settings=DAPS4K,model_presets={
        "ffhq":{"checkpoint":str(ROOT/"models/ffhq_10m.pt"),
                "checkpoint_sha256":MODEL_SHA,"num_channels":128,
                "num_res_blocks":1,"attention_resolutions":16}},
       gt_paths_supplied=False,historical_paths_supplied=False,
       outcomes_embargoed=True,seed_dataset="ffhq_daps4k",
       seed_rule="SHA256('XP|DAPS|ffhq_daps4k|<cohort>_<image_id>|rep') first8 BE mod2**31",
       rows=e)
    stages={}
    for stage,rows,mf in (("D3",d3,d3_mf),("EPRIME",e,e_mf)):
        dest=out/(stage+".json")
        write_x(dest,mf)
        stages[stage]=dict(path=str(dest),sha256=sha(dest),
                           n_roots=len(rows),
                           outcomes_embargoed=stage=="EPRIME")
    stages["D1_P2"]=dict(path=str(D1P2),sha256=D1_SHA,
                n_roots=1200,outcomes_embargoed=True,
                producer_head=D1_HEAD)
    prep=dict(status="XP_A6_NEXT_INPUTS_FROZEN",head=actual,
       n_total_new_roots=2320,n_new_d1p2=1200,
       n_new_d3=400,n_new_eprime=720,
       stages=stages,source_a16_sha256=A16_SHA,
       source_shared_imagenet_y_sha256=COMMON_Y_SHA,
       source_d1p2_manifest_sha256=D1_SHA,
       original_fast_imagenet_generation_head=D1_HEAD,
       gt_paths_supplied=False,no_gpu_or_gt_scoring_performed=True,
       outcome_embargo_active=True)
    write_x(out/"XP_A6_PREP.json",prep)
    print("XP_A6_PREP_PASS D1_P2=1200 D3=400 EPRIME=720")
    print("XP_A6_RUN_ROOT="+str(out))
    print("XP_A6_PREP_SHA="+sha(out/"XP_A6_PREP.json"))
    for k in ("D1_P2","D3","EPRIME"):print(k+"_MANIFEST_SHA="+stages[k]["sha256"])
    print("D1_P2_EPRIME_OUTCOMES_SEALED=true")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A6_PREP "+str(e),flush=True)
        sys.exit(8)
