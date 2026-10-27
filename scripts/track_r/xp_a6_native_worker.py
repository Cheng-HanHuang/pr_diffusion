#!/usr/bin/env python3
"""Single-root A6 SITCOM D3 / pristine DAPS E' GPU adapter.

Only delegates to previously replay-verified/calibrated native workers.
Does not modify model/sampler math or use any external bundled dataset.
All input manifest/checkpoint/source hashes are checked before GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a6_prepare_cpu import (
    ROOT,BASE,WT,DAPS4K,DAPS_HEAD,SITCOM_HEAD,check_native_checkout,
    SITCOM,NATIVE,verify_clean_sparse,sha,sitcom_seed,MODEL_SHA,
    IMAGENET_MODEL_SHA)
from scripts.track_r import xp_t_sitcom_worker_gpu as ss
from scripts.track_r import xp_t_daps4k_worker_gpu as dd

def validated(run,head,stage,index):
    run=Path(run)
    if (run.parent!=BASE or not run.name.startswith("xp_a6_")
        or run.is_symlink() or not run.is_dir() or stage not in ("D3","EPRIME")):
        raise RuntimeError("STOP_A6_NATIVE_SOURCE_NAMESPACE")
    x=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
        text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30).strip()
    if x!=head or dirty or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_A6_NATIVE_CODE_HEAD")
    prep=json.loads((run/"XP_A6_PREP.json").read_text())
    if (prep.get("status")!="XP_A6_NEXT_INPUTS_FROZEN"
        or prep.get("head")!=head
        or prep.get("no_gpu_or_gt_scoring_performed") is not True):
        raise RuntimeError("STOP_A6_NATIVE_PREP")
    info=prep["stages"][stage]
    path=run/(stage+".json")
    if info["path"]!=str(path) or sha(path)!=info["sha256"]:
        raise RuntimeError("STOP_A6_NATIVE_MANIFEST_HASH")
    mf=json.loads(path.read_text())
    rows=mf.get("rows",[])
    if (mf.get("head")!=head or
        mf.get("gt_paths_supplied") is not False
        or mf.get("historical_paths_supplied") is not False
        or mf.get("n_roots")!=len(rows)
        or index not in range(len(rows))
        or len({r["uid"] for r in rows})!=len(rows)):
        raise RuntimeError("STOP_A6_NATIVE_MANIFEST_OR_INDEX")
    row=rows[index]
    if (row["rep"]!=index%4
        or sha(row["measurement_path"])!=row["measurement_sha256"]):
        raise RuntimeError("STOP_A6_NATIVE_OBSERVATION")
    if stage=="D3":
        if (len(rows)!=400 or mf.get("status")!="XP_D3_TASKS_FROZEN"
            or mf.get("dataset")!="imagenet" or
            mf.get("native_source_head")!=SITCOM_HEAD
            or mf.get("outcomes_embargoed") is not False
            or row.get("dataset")!="imagenet"
            or row.get("uid")!=f"XP|SITCOM|imagenet|{row['image_id']}|rep{row['rep']}"
            or row.get("seed")!=sitcom_seed("imagenet",row["image_id"],row["rep"])
            or row.get("measurement_source")!="XP_IMAGENET100"):
            raise RuntimeError("STOP_A6_NATIVE_D3_IDENTITY")
        if mf["model_presets"]["imagenet"]["checkpoint_sha256"]!=IMAGENET_MODEL_SHA:
            raise RuntimeError("STOP_A6_D3_CHECKPOINT_REFERENCE")
        for k in ("imagenet","ffhq"):
            p=mf["model_presets"][k]
            if sha(p["checkpoint_path"])!=p["checkpoint_sha256"]:
                raise RuntimeError("STOP_A6_D3_MODEL_HASH")
        native_head=subprocess.check_output(["git","-C",str(SITCOM),"rev-parse","HEAD"],text=True).strip()
        state=check_native_checkout(native_head,
            subprocess.check_output(["git","-C",str(SITCOM),"status","--porcelain=v1",
                 "--untracked-files=no"],text=True),
            subprocess.check_output(["git","-C",str(SITCOM),"diff","--numstat","HEAD",
                 "--","posterior_sample.py"],text=True))
        if (state["tracked_modified"]!=mf["external_tracked_diff_allowlist"]
            or sha(SITCOM/"posterior_sample.py")!=mf["external_unused_cli_sha256"]):
            raise RuntimeError("STOP_A6_D3_NATIVE_DIFF")
        if os.environ.get("PYTHONPYCACHEPREFIX","").startswith(str(run/"pycache")+"/") is False:
            raise RuntimeError("STOP_A6_D3_BYTECODE")
    else:
        if (len(rows)!=720 or mf.get("status")!="XP_EPRIME_TASKS_FROZEN"
            or mf.get("dataset")!="ffhq"
            or mf.get("source_head")!=DAPS_HEAD
            or mf.get("settings")!=DAPS4K
            or mf.get("outcomes_embargoed") is not True
            or row.get("dataset")!="ffhq"
            or row.get("family")!="DAPS"
            or row.get("uid")!=f"XP|DAPS|ffhq_daps4k|{row['image_id']}|rep{row['rep']}"
            or row.get("measurement_source") not in ("S1_b22","S2_b24_dev80")):
            raise RuntimeError("STOP_A6_EPRIME_IDENTITY")
        seed_key=f"XP|DAPS|ffhq_daps4k|{row['image_id']}|{row['rep']}"
        expected=int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8],"big")%(2**31)
        if row["seed"]!=expected:
            raise RuntimeError("STOP_A6_EPRIME_SEED")
        p=mf["model_presets"]["ffhq"]
        if p["checkpoint_sha256"]!=MODEL_SHA or sha(p["checkpoint"])!=MODEL_SHA:
            raise RuntimeError("STOP_A6_EPRIME_MODEL_HASH")
        if not verify_clean_sparse(NATIVE):
            raise RuntimeError("STOP_A6_EPRIME_NATIVE_SOURCE")
        if os.environ.get("PYTHONPYCACHEPREFIX","").startswith(str(run/"pycache")+"/") is False:
            raise RuntimeError("STOP_A6_EPRIME_BYTECODE")
    return mf,row,info["sha256"]

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--run",required=True)
    a.add_argument("--expected-head",required=True)
    a.add_argument("--stage",choices=("D3","EPRIME"),required=True)
    a.add_argument("--index",type=int,required=True)
    a.add_argument("--gpu",type=int,required=True)
    a.add_argument("--preflight-only",action="store_true")
    args=a.parse_args()
    mf,row,digest=validated(args.run,args.expected_head,args.stage,args.index)
    if args.preflight_only:
        print("XP_A6_NATIVE_PREFLIGHT_PASS",args.stage,args.index,flush=True)
        return 0
    if (args.gpu not in range(4)
        or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu)):
        raise RuntimeError("STOP_A6_NATIVE_PHYSICAL_GPU")
    if args.stage=="D3":
        ss.RUN=Path(args.run)
        ss.PREP_HEAD=args.expected_head
        ss.MANIFEST_SHA=digest
        return ss.generate(args,mf,row)
    dd.RUN=Path(args.run)
    dd.PREP_HEAD=args.expected_head
    dd.MANIFEST_SHA=digest
    return dd.generate(args,mf,row)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A6_NATIVE "+str(e),flush=True)
        sys.exit(8)
