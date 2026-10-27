#!/usr/bin/env python3
"""Single-root published DAPS-4K ImageNet XP-D2 using calibrated worker.

Only changes immutable input manifest/output namespace (not DAPS algorithm).
Native code, checkpoint and measurement verified before GPU operation.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_night_prepare_cpu import ROOT,WT,BASE,COUNTS,DAPS_HEAD
from scripts.track_r.xp_t_fast_prepare_cpu import sha as file_sha
from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
from scripts.track_r.xp_t_daps4k_worker_gpu import (
    validate_weight_trace,model_config)
from scripts.track_r import xp_t_daps4k_worker_gpu as native_worker

def manifest_checked(root,head,index):
    root=Path(root)
    if root.parent!=BASE or not root.name.startswith("xp_overnight_"):
        raise RuntimeError("STOP_D2_BAD_RUN_ROOT")
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_D2_PYCACHE")
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
        text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30).strip()
    if actual!=head or dirty:
        raise RuntimeError("STOP_D2_GIT")
    prep_path=root/"XP_NIGHT_PREP.json"
    prep=json.loads(prep_path.read_text())
    if (prep.get("status")!="XP_NIGHT_INPUTS_FROZEN" or prep.get("head")!=head
        or prep.get("gpu_work_performed") is not False
        or prep.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_D2_PREP")
    source=root/"D2.json"
    if file_sha(source)!=prep["stages"]["D2"]["sha256"]:
        raise RuntimeError("STOP_D2_SOURCE_SHA")
    manifest=json.loads(source.read_text())
    if (manifest.get("status")!="XP_DAPS4K_TASKS_FROZEN"
        or manifest.get("stage")!="D2"
        or manifest.get("head")!=head
        or manifest.get("dataset")!="imagenet"
        or manifest.get("source_head")!=DAPS_HEAD
        or manifest.get("n_roots")!=400
        or manifest.get("n_images")!=100
        or manifest.get("gt_paths_supplied") is not False
        or manifest.get("outcome_embargo") is not False
        or manifest.get("settings")!=native_worker.DAPS4K):
        raise RuntimeError("STOP_D2_MANIFEST")
    rows=manifest.get("rows",[])
    if (len(rows)!=400 or len({r["uid"] for r in rows})!=400
        or index not in range(400)):
        raise RuntimeError("STOP_D2_ROWS")
    row=rows[index]
    if (row.get("uid")!=f"XP|DAPS|imagenet|{row['image_id']}|rep{index%4}"
        or row.get("family")!="DAPS"
        or row.get("rep")!=index%4
        or row.get("dataset")!="imagenet"
        or row.get("measurement_source")!="XP_IMAGENET100"
        or file_sha(row["measurement_path"])!=row["measurement_sha256"]):
        raise RuntimeError("STOP_D2_ROW_Y")
    if not verify_clean_sparse(NATIVE):
        raise RuntimeError("STOP_D2_PINNED_SOURCE_DRIFT")
    model=manifest["model_presets"]["imagenet"]
    if (file_sha(model["checkpoint"])!=model["checkpoint_sha256"]
        or model["checkpoint_sha256"]!=
           "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"):
        raise RuntimeError("STOP_D2_MODEL")
    return manifest,row,prep["stages"]["D2"]["sha256"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--index",required=True,type=int)
    p.add_argument("--gpu",required=True,type=int)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    manifest,row,digest=manifest_checked(a.root,a.expected_head,a.index)
    if a.preflight_only:
        print("XP_D2_PREFLIGHT_PASS",a.index,row["image_id"],flush=True)
        return 0
    if (not 0<=a.gpu<4 or
        os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.gpu)):
        raise RuntimeError("STOP_D2_GPU_ENV")
    # Reuse exact calibrated 400x10/100-Langevin native worker; patch
    # *only* module-local manifest receipt constants for this process.
    native_worker.RUN=Path(a.root)
    native_worker.PREP_HEAD=a.expected_head
    native_worker.MANIFEST_SHA=digest
    return native_worker.generate(a,manifest,row)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_D2_WORKER "+str(exc),flush=True)
        sys.exit(8)
