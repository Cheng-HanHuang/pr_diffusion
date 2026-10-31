#!/usr/bin/env python3
"""A16.5 XP-N two-root D1 replay gate (native FAST, no new outcomes).

Run only after the independent XP-N GT/y identity audit. Reproduce the
original D1 A6 root 4 from two different images, with D1's pinned
measurements, checkpoint, sampler and seeds. Compare reconstructed RGB
pixels against frozen historical PNGs. Under A10, co-tenancy is allowed
within 80% GPU VRAM, with no foreign-process termination.

This controller intentionally DOES NOT start any XP-N production.
"""
from __future__ import annotations
import argparse,datetime,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from PIL import Image,ImageChops
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g1a_fast_gpu import (
  A12_G1A_FROZEN_PRODUCER,A12_G1A_FROZEN_SHAS,
  root_folder,validate_complete,sha)
from scripts.track_r.xp_a15_cp2_d4_cpu import jread,write
from scripts.track_r.xp_a18_xpn_image_inventory_cpu import check_d1
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

PREP=BASE/"xp_a12_g1a_prepare_20261029T170316Z"
FROZEN=PREP/"XP_A12_G1A_REPLAY2_TASKS.json"
HIST=BASE/"xp_a6_20261027T165112Z"/"gpu/D1_P2/chunk_00000_024"
OLD_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
Y_PREFIX="xp_a18_xpn_measurements_"
PROJECT_CAP=400.
XPN_CAP=20.
REPLAY_CAP=1.
A12_STATE=BASE/"xp_a12_campaign_prod_20261030T022725Z"/"XP_A12_CAMPAIGN_STATE.json"
A17_TIMING=BASE/"xp_a17_cp3_timing_20261031T165802Z"/"A17_CP3_CLEAN_TIMING_RECEIPT.json"

def source(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_REPLAY_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(
  ["git","-C",str(wt),*z],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"A18_REPLAY_CLEAN_HEAD")
 check_d1()
 ydir=Path(a.measurements)
 gate(ydir.parent==BASE and ydir.name.startswith(Y_PREFIX)
      and ydir.is_dir() and not ydir.is_symlink(),
      "A18_REPLAY_XPN_Y_SOURCE")
 yrec=jread(ydir/"XP_N_GT_Y_PREP_RECEIPT.json")
 auditfile=ydir/"XP_N_GT_Y_INDEPENDENT_AUDIT.json"
 audit=jread(auditfile)
 gate(yrec["status"]=="A18_XPN_GT_Y_ISOLATED_CPU_PREP_PASS_IDENTITY_REAUDIT_PENDING"
      and audit["status"]=="A18_XPN_GT_Y_INDEPENDENT_ALL100_PASS"
      and audit["n_images"]==100
      and audit["GT_tensor_sha256_verified_all100"] is True
      and audit["signed_y_reconstituted_all100"] is True
      and audit["worst_identity_rms"]<=1e-4
      and audit["source_y_manifest_sha256"]==yrec["y_manifest_sha256"]
      and audit["gt_only_manifest_sha256"]==yrec["gt_only_manifest_sha256"]
      and sha(ydir/"XP_N_IMAGENET100_Y_MANIFEST.json")==yrec["y_manifest_sha256"],
      "A18_REPLAY_INDEPENDENT_XPN_Y_GT_IDENTITY")
 gate(FAST_PY.is_file(),"A18_REPLAY_FAST_NATIVE_ENV")
 gate(FROZEN.is_file() and sha(FROZEN)==
      A12_G1A_FROZEN_SHAS["A12_G1A_HISTORICAL_REPLAY2"],
      "A18_REPLAY_FROZEN_D1_REPLAY2_MANIFEST")
 manifest=jread(FROZEN);rows=manifest["rows"]
 gate(manifest["head"]==A12_G1A_FROZEN_PRODUCER
      and manifest["stage"]=="A12_G1A_HISTORICAL_REPLAY2"
      and manifest["dataset"]=="imagenet"
      and manifest["gt_paths_supplied"] is False
      and manifest["outcome_embargo"] is True
      and len(rows)==2 and rows[0]["image_id"]!=rows[1]["image_id"]
      and all(int(r["rep"])==4 for r in rows),
      "A18_REPLAY_EXACT_TWO_D1_ROOTS")
 old=jread(HIST/"R0_5_CHUNK_COMPLETE.json")
 gate(old["status"]=="R0_5_CHUNK_PASS" and old["head"]==OLD_HEAD
      and old["start"]==0 and old["count"]==24,
      "A18_REPLAY_PINNED_HISTORICAL_CHUNK")
 refs=[]
 for r in rows:
  olddir=root_folder(HIST,r)
  rec=validate_complete(olddir,r,OLD_HEAD,
      "3107d40c495886c52ea2a407f4d8f74079278fffffaa93f4f55eff1edb5272f7")
  gate(rec["gt_access"] is False and rec["dataset"]=="imagenet"
       and rec["rep"]==4 and rec["fast_source_head"]==
           "a9758b38906b7c7950fa04491f37f6634948db26",
       "A18_REPLAY_REFERENCE_NATIVE_SOURCE")
  refs.append((r,olddir,rec))
 base=jread(A12_STATE)
 gate(base["status"]=="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
      and not base.get("active"),"A18_REPLAY_A12_LEDGER_TERMINAL")
 t=jread(A17_TIMING)
 gate(t["status"]=="A17_CP3_CLEAN_PAIRED_TIMING_PASS",
      "A18_REPLAY_A17_TIMING_CHARGES")
 project=float(base["project_charged_gpu_h"])+float(
    t["total_A17_charged_gpu_hours"])
 prior=0.
 for p in BASE.glob("xp_a18_xpn_replay_*/XP_N_D1_REPLAY_RECEIPT.json"):
  rec=jread(p)
  gate(rec["status"] in ("A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS",
                        "A18_XPN_D1_REPLAY_FAILED_OR_PARTIAL"),
       "A18_REPLAY_PRIOR_RECEIPT_STATUS")
  gate(rec["charged_gpu_h"]>=0,"A18_REPLAY_PRIOR_GPU_CHARGE")
  prior+=float(rec["charged_gpu_h"])
  gate(rec["status"]!="A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS",
       "A18_REPLAY_ALREADY_SUCCEEDED")
 gate(project+prior+XPN_CAP<=PROJECT_CAP
      and prior+REPLAY_CAP<=XPN_CAP,
      "A18_REPLAY_PROJECT_AND_XPN_BUDGET")
 return wt,ydir,refs,project,prior

def admissible_cards():
 cards=collect()
 # Exclude other Track R workers on the same GPU, but preserve the
 # A10 allowance for unrelated co-tenant jobs.
 candidates=[]
 for c in cards:
  if c.visibility!="known":continue
  own_track=False
  for pid in c.process_pids:
   cmd=subprocess.run(["ps","-o","args=","-p",str(pid)],
      capture_output=True,text=True,timeout=8,check=False)
   if cmd.returncode==0 and (
       "scripts/track_r/" in cmd.stdout or
       "scripts.track_r." in cmd.stdout):
    own_track=True
    break
  if not own_track:candidates.append(c)
 return candidates

def main(a):
 wt,ydir,refs,project,prior=source(a)
 cards=admissible_cards()
 chosen,reasons=choose_gpu(cards,8192,
    safety_margin_mib=2048,fraction=.80,clean_only=False)
 if a.preflight_only:
  print("A18_XPN_D1_REPLAY_PREFLIGHT_PASS",
        "SOURCE_GT_Y_SHA",sha(ydir/"XP_N_GT_Y_INDEPENDENT_AUDIT.json"),
        "ADMISSIBLE_GPU",chosen,"REASONS",reasons,
        "BUDGET_PROJECT_H",round(project,6),flush=True)
  return
 gate(chosen is not None,"A18_REPLAY_NO_ADMISSIBLE_GPU")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a18_xpn_replay_")
      and not out.exists() and not out.is_symlink(),
      "A18_REPLAY_NEW_OUTPUT")
 out.mkdir(mode=0o700)
 (out/"logs").mkdir()
 dest=out/"replay"/"chunk_00000_002"
 cmd=[str(FAST_PY),str(wt/"scripts/track_r/xp_a12_g1a_fast_gpu.py"),
   "--expected-head",a.expected_head,
   "--roots",str(FROZEN),
   "--roots-sha",sha(FROZEN),
   "--out",str(dest),"--start","0","--count","2",
   "--physical-gpu",str(chosen),"--dataset","imagenet",
   "--oversample","2.0",
   "--roots-source-head",A12_G1A_FROZEN_PRODUCER]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(chosen),
    PYTHONPATH=str(wt),PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
    CUBLAS_WORKSPACE_CONFIG=":4096:8")
 logfile=out/"logs"/"D1_REPLAY2_NATIVE_FAST.log"
 seen=set();unknown=False;max_used=0;nsamples=0
 initial_card=next(c for c in cards if c.index==chosen)
 print("A18_XPN_D1_TWO_ROOT_REPLAY_START",
       "GPU",chosen,"COTENANT",bool(initial_card.foreign_pids),flush=True)
 start=time.monotonic()
 with logfile.open("x") as log:
  proc=subprocess.Popen(cmd,cwd=str(wt),env=env,
      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   time.sleep(2)
   try:
    card=next((c for c in collect(own_roots=[proc.pid]) if c.index==chosen),None)
    if card is None or card.visibility!="known":unknown=True
    else:
     seen.update(card.foreign_pids)
     max_used=max(max_used,card.used_mib)
     nsamples+=1
     if (card.used_mib>.8*card.total_mib or
         (time.monotonic()-start)>REPLAY_CAP*3600):
      # Only terminate our own native worker; preserve foreign jobs.
      proc.terminate()
   except (Exception,StopIteration):
    unknown=True
  rc=proc.wait()
 cost=(time.monotonic()-start)/3600
 correct=False;pixel_match=[];verify_error=None
 try:
  if rc==0:
   valid=jread(dest/"R0_5_CHUNK_COMPLETE.json")
   gate(valid["status"]=="R0_5_CHUNK_PASS"
        and valid["count"]==2 and valid["new"]==2
        and valid["head"]==a.expected_head,
        "A18_REPLAY_TWO_NATIVE_COMPLETES")
   correct=True
   for row,oldpath,oldrec in refs:
    path=root_folder(dest,row)
    newer=validate_complete(path,row,a.expected_head,sha(FROZEN))
    with (
        Image.open(oldpath/"reconstruction.png") as im0,
        Image.open(path/"reconstruction.png") as im1,
    ):
     same=im0.mode==im1.mode and im0.size==im1.size and (
          ImageChops.difference(im0,im1).getbbox() is None)
    pixel_match.append(dict(uid=row["uid"],original_png_sha256=oldrec[
        "reconstruction_png_sha256"],replayed_png_sha256=newer[
        "reconstruction_png_sha256"],pixels_exact=same))
    correct=correct and same
 except Exception as e:
  correct=False
  verify_error=type(e).__name__+":"+str(e)[:400]
 status=("A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS"
          if rc==0 and correct and len(pixel_match)==2 and
             max_used<=.8*initial_card.total_mib and
             not unknown and nsamples>0
          else "A18_XPN_D1_REPLAY_FAILED_OR_PARTIAL")
 receipt=dict(status=status,
    expected_head=a.expected_head,XP_N_independent_GT_Y_audit_sha256=sha(
        ydir/"XP_N_GT_Y_INDEPENDENT_AUDIT.json"),
    historical_D1_manifest_SHA256=sha(FROZEN),
    historical_D1_A6_head=OLD_HEAD,
    native_fast_source_head="a9758b38906b7c7950fa04491f37f6634948db26",
    n_roots=len(pixel_match),pixel_comparisons=pixel_match,
    gpu_physical=chosen,co_tenant_on_entry=bool(initial_card.foreign_pids),
    foreign_pids_seen=sorted(seen),visibility_unknown=unknown,
    max_memory_used_mib=max_used,n_telemetry_samples=nsamples,
    native_worker_rc=rc,native_log_sha256=sha(logfile),
    validation_error=verify_error,
    charged_gpu_h=cost,prior_XPN_replay_gpu_h=prior,
    project_gpu_h_before_XPN=project,
    project_cap_gpu_h=PROJECT_CAP,XP_N_GPU_h_cap=XPN_CAP,
    no_GT_or_candidate_scores_read=True,no_XPN_candidates_generated=True)
 write(out/"XP_N_D1_REPLAY_RECEIPT.json",receipt)
 print("A18_XPN_D1_REPLAY_TERMINAL",status,
       "GPU_H",round(cost,6),"GPU",chosen,"RC",rc,flush=True)
 gate(status=="A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS",
      "A18_REPLAY_PIXEL_EXACT_REQUIRED")

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--measurements",required=True)
 p.add_argument("--out")
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_TWO_ROOT_REPLAY",e,flush=True)
  sys.exit(8)
