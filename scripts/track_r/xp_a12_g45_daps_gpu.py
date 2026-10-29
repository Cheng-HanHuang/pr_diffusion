#!/usr/bin/env python3
"""A12 G4/G5 native DAPS-4K one-root stage adapter.

Shares the rigorously audited XP-T DAPS-4K native generate() without
changing its numerical algorithm, native source, model, or settings.
The only changes are frozen new run indices and isolated output path.
Historical two-root replay inputs must run and match before production.
GT never appears in worker inputs.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
 DAPS4K,DAPS_HEAD,MODEL_SHA,IMAGENET_MODEL_SHA,DAPS as NATIVE,verify_clean_sparse,
 PY_DAPS)
from scripts.track_r.xp_a12_g45_prepare_cpu import sha,seed
from scripts.track_r import xp_t_daps4k_worker_gpu as native
def check(a):
    root=Path(a.run)
    gate(root.parent==BASE and root.name.startswith(("xp_a12_g45_prod_","xp_a12_campaign_prod_"))
         and root.is_dir() and not root.is_symlink(),"G45_RUN")
    gate(a.stage in ("G4","G5") and a.phase in ("REPLAY","PROD"),
         "G45_STAGE_PHASE")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *q:subprocess.check_output(
      ["git","-C",str(wt),*q],text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"G45_EXECUTOR_HEAD")
    gate(Path(sys.executable).resolve()==PY_DAPS.resolve()
         and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
         and os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
         and a.gpu in range(4)
         and os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
              str(root/"pycache")+"/"),
         "NATIVE_DAPS_PROCESS_CONTRACT")
    gate(verify_clean_sparse(NATIVE),"DAPS_PRISTINE_SOURCE")
    prep=Path(a.prep)
    gate(prep.parent==BASE and prep.name.startswith("xp_a12_g45_prepare_")
         and prep.is_dir(),"G45_PREP")
    receipt=json.loads((prep/"XP_A12_G45_PREP_RECEIPT.json").read_text())
    gate(receipt["status"]=="XP_A12_G45_INPUTS_FROZEN_CPU_PASS"
         and receipt["head"]==a.roots_source_head
         and receipt["all_replays_pending"] is True,
         "G45_PREP_RECEIPT")
    name=f"{a.stage}_{a.phase}"
    mpath=prep/f"XP_A12_{name}_TASKS.json"
    gate(sha(mpath)==receipt["stages"][name]["sha256"],
         "G45_MANIFEST_SHA")
    mf=json.loads(mpath.read_text())
    n=720 if a.stage=="G4" else 400
    expected_count=2 if a.phase=="REPLAY" else n
    gate(mf["status"]=="XP_A12_DAPS4K_TASKS_FROZEN"
         and mf["schema"]=="prdiff.track_r.xp.daps4k.a12.v1"
         and mf["head"]==a.roots_source_head
         and mf["source_head"]==DAPS_HEAD
         and mf["settings"]==DAPS4K
         and mf["gt_paths_supplied"] is False
         and mf["n_roots"]==expected_count
         and 0<=a.index<expected_count,
         "G45_MANIFEST")
    row=mf["rows"][a.index]
    ds="ffhq" if a.stage=="G4" else "imagenet"
    gate(row["dataset"]==ds and
         row["seed"]==seed(a.stage,row["image_id"],row["rep"]) and
         row["rep"] in (range(4,8) if a.phase=="PROD" else range(0,4))
         and sha(row["measurement_path"])==row["measurement_sha256"],
         "G45_NONSEALED_Y_OR_SEED")
    model=mf["model_presets"][ds]
    chk=MODEL_SHA if ds=="ffhq" else IMAGENET_MODEL_SHA
    gate(model["checkpoint_sha256"]==chk and
         sha(model["checkpoint"])==chk and
         mf["method"]=="DAPS-4K","MODEL_SHA")
    return root,mf,row,sha(mpath)
def main():
    p=argparse.ArgumentParser()
    for a in ("expected-head","roots-source-head","run","prep","stage","phase"):
        p.add_argument("--"+a,required=True)
    p.add_argument("--index",required=True,type=int)
    p.add_argument("--gpu",required=True,type=int)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    root,mf,row,digest=check(a)
    if a.preflight_only:
        print("A12_G45_NATIVE_PREFLIGHT_PASS",a.stage,a.phase,a.index,flush=True)
        return 0
    # The legacy native worker writes into RUN/gpu/root_<UID-SHA>.
    # Keep replay/production separated, and never overwrite old roots.
    dest=root/a.stage/a.phase
    gate((dest/"gpu").is_dir(),"G45_OUTPUT_NAMESPACE")
    native.RUN=dest
    native.PREP_HEAD=a.roots_source_head
    native.MANIFEST_SHA=digest
    return native.generate(a,mf,row)
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        traceback.print_exc();print("ACTION NEEDED: A12_G45_GPU",e,flush=True);sys.exit(8)
