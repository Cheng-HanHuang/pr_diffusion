#!/usr/bin/env python3
"""A12 G6 conditional clean timing inputs (3 images x 3 per family).

CPU freeze of original XP-T DAPS-4K FFHQ and FAST ImageNet first-batch
measurements/seeds, with no GT. NEVER launch if both old batches have
valid clean timing; G6 exists only to repair missing clean comparisons.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,FAST_HEAD,gate
from scripts.track_r.xp_a12_g45_prepare_cpu import sha,write
from scripts.track_r.xp_t_daps4k_prepare_cpu import DAPS_HEAD,DAPS4K
DAPS=BASE/"xp_t_daps4k_20261027T011832Z"
FAST=BASE/"xp_t_fast_imagenet_20261026T212512Z"
def timing_check():
 ff=json.loads((FAST/"XP_T_FAST_COMPLETE.json").read_text())
 dd=json.loads((DAPS/"XP_T_DAPS4K_COMPLETE.json").read_text())
 # Distinguish clean timing validity from successful reconstruction.
 return dict(fast_clean=ff.get("timing_clean_eligible") is True,
             daps_clean=dd.get("clean_timing_eligible") is True,
             fast_receipt_sha=sha(FAST/"XP_T_FAST_COMPLETE.json"),
             daps_receipt_sha=sha(DAPS/"XP_T_DAPS4K_COMPLETE.json"))
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                  text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"WORKTREE")
 report=timing_check()
 if report["fast_clean"] and report["daps_clean"]:
  print("XP_A12_G6_EXISTING_CLEAN_TIMING_ALREADY_SATISFIED",flush=True)
  return
 oldf=json.loads((FAST/"XP_T_FAST_ROOTS.json").read_text())
 oldd=json.loads((DAPS/"XP_T_DAPS4K_TASKS.json").read_text())
 fr=oldf["rows"]
 dr=[r for r in oldd["rows"] if r["dataset"]=="ffhq"]
 gate(len(fr)==len(dr)==9 and oldf["dataset"]=="imagenet"
      and oldf["fast_head"]==FAST_HEAD
      and oldd["settings"]==DAPS4K
      and oldd["source_head"]==DAPS_HEAD,"ORIGINAL_XPT_SETTINGS")
 for r in fr+dr:
  gate(sha(r["measurement_path"])==r["measurement_sha256"],
       "FROZEN_Y_SHA")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a12_g6_prepare_")
      and not out.exists() and not out.is_symlink(),"FRESH_OUTPUT")
 out.mkdir(mode=0o700)
 fast= dict(schema="prdiff.track_r.xp.fast-tasks.v1",
     status="XP_FAST_TASKS_FROZEN",
     stage="A12_G6_FAST_IN_CLEAN",
     head=a.expected_head,dataset="imagenet",oversample=2.0,
     fast_head=FAST_HEAD,n_roots=9,n_images=3,
     model_sha256=oldf["model_sha256"],
     gt_paths_supplied=False,rows=fr,
     original_timing_task_sha256=sha(FAST/"XP_T_FAST_ROOTS.json"))
 daps=dict(schema="prdiff.track_r.xp.daps4k.a12.v1",
     status="XP_A12_DAPS4K_TASKS_FROZEN",stage="G6",
     phase="CLEAN_TIMING",head=a.expected_head,
     source_head=DAPS_HEAD,n_roots=9,n_images=3,
     dataset="ffhq",settings=DAPS4K,
     model_presets=oldd["model_presets"],
     gt_paths_supplied=False,method="DAPS-4K",rows=dr)
 fpath=out/"XP_A12_G6_FAST_IN_TASKS.json"
 dpath=out/"XP_A12_G6_DAPS_FFHQ_TASKS.json"
 write(fpath,fast);write(dpath,daps)
 write(out/"XP_A12_G6_PREP_RECEIPT.json",dict(
      status="XP_A12_G6_CONDITIONAL_CLEAN_TIMING_PREP_PASS",
      head=a.expected_head,
      n_fast=9,n_daps=9,no_GT=True,no_GPU=True,
      fast_manifest_sha256=sha(fpath),daps_manifest_sha256=sha(dpath),
      prior_clean_receipts=report,
      clean_GPUs_ONLY=True,stage_cap_gpu_h=1))
 print("XP_A12_G6_CLEAN_TIMING_PREP_PASS 9+9",flush=True)
 print("G6_PREP_DIR",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G6_PREP",e,flush=True);sys.exit(8)
