#!/usr/bin/env python3
"""A12 G6 native DAPS-4K clean timing 1-root adapter.

Exact published sampler and XP-T FFHQ first-3x3 sources, not a new
method or noisy y. The clean-only supervisor controls all GPU admission.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g45_prepare_cpu import sha
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
 NATIVE,PY_DAPS,DAPS_HEAD,DAPS4K,MODEL_SHA,verify_clean_sparse)
from scripts.track_r import xp_t_daps4k_worker_gpu as native
def main():
 p=argparse.ArgumentParser()
 for x in ("run","prep","expected-head","roots-source-head"):
  p.add_argument("--"+x,required=True)
 p.add_argument("--index",type=int,required=True)
 p.add_argument("--gpu",type=int,required=True)
 p.add_argument("--preflight-only",action="store_true")
 a=p.parse_args()
 run=Path(a.run);prep=Path(a.prep)
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(["git","-C",str(wt),*args],
                       text=True).strip()
 gate(run.parent==BASE and run.name.startswith("xp_a12_g6_prod_")
      and run.is_dir() and not run.is_symlink(),"RUN")
 gate(prep.parent==BASE and prep.name.startswith("xp_a12_g6_prepare_"),"PREP")
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"WORKTREE")
 gate(Path(sys.executable).resolve()==PY_DAPS.resolve() and
      os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu) and
      os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
           str(run/"pycache")+"/"),"GPU_ENV")
 gate(verify_clean_sparse(NATIVE),"NATIVE_SOURCE")
 rec=json.loads((prep/"XP_A12_G6_PREP_RECEIPT.json").read_text())
 path=prep/"XP_A12_G6_DAPS_FFHQ_TASKS.json"
 gate(rec["head"]==a.roots_source_head and
      rec["status"]=="XP_A12_G6_CONDITIONAL_CLEAN_TIMING_PREP_PASS"
      and rec["daps_manifest_sha256"]==sha(path),"FROZEN_INPUT")
 mf=json.loads(path.read_text())
 gate(mf["head"]==a.roots_source_head and
      mf["status"]=="XP_A12_DAPS4K_TASKS_FROZEN" and
      mf["stage"]=="G6" and
      mf["source_head"]==DAPS_HEAD and
      mf["settings"]==DAPS4K and
      len(mf["rows"])==9 and 0<=a.index<9,"MANIFEST")
 r=mf["rows"][a.index]
 gate(r["dataset"]=="ffhq" and
      sha(r["measurement_path"])==r["measurement_sha256"]
      and sha(ROOT/"models/ffhq_10m.pt")==MODEL_SHA,"Y_AND_MODEL_SHA")
 if a.preflight_only:
  print("A12_G6_DAPS_NATIVE_PREFLIGHT_PASS",a.index,flush=True);return 0
 native.RUN=run
 native.PREP_HEAD=a.roots_source_head
 native.MANIFEST_SHA=rec["daps_manifest_sha256"]
 return native.generate(a,mf,r)
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G6_DAPS",e,flush=True)
  sys.exit(8)
