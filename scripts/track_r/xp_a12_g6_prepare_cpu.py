#!/usr/bin/env python3
"""A13 G6 clean timing: FAST ImageNet and *legacy DAPS 400x5 FFHQ*.

Do NOT substitute the published DAPS-4K 400x10. G6 specifically asks
for native F8/DAPS 400x5 FFHQ first 3 B22 images x 3 new timing runs.
Both groups use SHA-pinned existing y and independent original seeds.
The original F8 patched native adapter, not pristine DAPS-4K, is used.
No GPU, GT, scoring or checkpoint loaded by this preflight.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,FAST_HEAD,gate
from scripts.track_r.xp_a12_g45_prepare_cpu import sha,write
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
 ORIG_DAPS,DAPS_HEAD,PATCHED_SAMPLER_BLOB,MODEL,MODEL_SHA,git)
FAST=BASE/"xp_t_fast_imagenet_20261026T212512Z"
F8=BASE/"xp_f8_prod_20261029T071352Z"
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 cmd=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
          text=True).strip()
 gate(cmd("rev-parse","HEAD")==a.expected_head and
      not cmd("status","--porcelain"),"WORKTREE")
 gate(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD
      and git(ORIG_DAPS,"hash-object","sampler.py")==PATCHED_SAMPLER_BLOB
      and sha(MODEL)==MODEL_SHA,"NATIVE_DAPS400X5_SHA")
 fpath=FAST/"XP_T_FAST_ROOTS.json"
 fast=json.loads(fpath.read_text())
 gate(fast["status"]=="XP_FAST_TASKS_FROZEN"
      and fast["dataset"]=="imagenet"
      and fast["fast_head"]==FAST_HEAD
      and fast["n_roots"]==9,"FAST_IMAGE3X3")
 fr=fast["rows"]
 old=F8/"XP_F8_PROD_TASKS.json"
 f8=json.loads(old.read_text())
 rec=json.loads((F8/"XP_F8_PROD_LAST_STATUS.json").read_text())
 gate(f8["n_total"]==920
      and rec["status"]=="XP_F8_920_GENERATION_COMPLETE_EMBARGOED"
      and rec["complete"]==920,"F8_SOURCE_COMPLETE")
 groups=defaultdict(list)
 for r in f8["rows"]:
  if r["cohort"]=="S1_b22":
   groups[r["image_id"]].append(r)
 ids=sorted(groups)[:3]
 gate(len(ids)==3,"B22_FIRST3")
 dr=[]
 for iid in ids:
  allr={int(r["rep"]):r for r in groups[iid]}
  gate(set(allr)==set(range(2,8)),"B22_SIX_F8_PREFIX")
  for rep in (2,3,4):
   r=allr[rep]
   dr.append(dict(uid=r["uid"],family="DAPS",
      dataset="ffhq",cohort=r["cohort"],image_id=iid,
      rep=rep,native_seed=r["native_seed"],measurement_source="S1_b22",
      measurement_path=r["measurement_path"],
      measurement_sha256=r["measurement_sha256"]))
 gate(len(fr)==len(dr)==9 and
      len({r["uid"] for r in dr})==9,"G6_9X2")
 for row in fr+dr:
  gate(sha(row["measurement_path"])==row["measurement_sha256"],
       "FROZEN_MEASUREMENT_SHA")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a12_g6_prepare_")
      and not out.exists() and not out.is_symlink(),"OUT")
 out.mkdir(mode=0o700)
 ff=dict(schema="prdiff.track_r.xp.fast-tasks.v1",
   status="XP_FAST_TASKS_FROZEN",
   stage="A12_G6_FAST_IN_CLEAN",
   head=a.expected_head,dataset="imagenet",oversample=2.0,
   fast_head=FAST_HEAD,n_roots=9,n_images=3,
   model_sha256=fast["model_sha256"],
   gt_paths_supplied=False,rows=fr,
   original_timing_task_sha256=sha(fpath))
 dd=dict(schema="prdiff.track_r.a13.g6-daps400x5.v1",
   status="XP_A13_G6_DAPS400X5_FFHQ_TASKS_FROZEN",
   stage="G6",head=a.expected_head,
   native_head=DAPS_HEAD,patched_sampler_blob=PATCHED_SAMPLER_BLOB,
   n_roots=9,n_images=3,dataset="ffhq",
   native_config="annealing400_diffusion5_langevin100",
   checkpoint_sha256=MODEL_SHA,
   source_F8_manifest_sha256=sha(old),
   gt_paths_supplied=False,rows=dr)
 fp=out/"XP_A12_G6_FAST_IN_TASKS.json"
 dp=out/"XP_A13_G6_DAPS400X5_TASKS.json"
 write(fp,ff);write(dp,dd)
 write(out/"XP_A13_G6_PREP_RECEIPT.json",dict(
   status="XP_A13_G6_CLEAN_TIMING_TWO_FAMILIES_PREP_PASS",
   head=a.expected_head,n_fast=9,n_daps=9,no_GT=True,
   no_GPU=True,fast_manifest_sha256=sha(fp),
   daps_manifest_sha256=sha(dp),
   native_DAPS_config="400x5_NOT_400x10",
   clean_GPUs_ONLY=True,stage_cap_gpu_h=1))
 print("XP_A13_G6_400X5_FAST_IN_PREP_PASS 9+9",flush=True)
 print("G6_PREP_DIR",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_G6_PREP",e,flush=True)
  sys.exit(8)
