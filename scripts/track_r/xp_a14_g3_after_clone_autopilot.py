#!/usr/bin/env python3
"""A14 G3 PC-DAPS after Cheng's exact native-source clone.

Uses only frozen FFHQ original signed y; CPU-certified G3 input manifest,
2-image native smoke, then 180 or authorized cohort-A 100 bags within
45 occupied GPU-hours. Never repeats G2/G4/G5, and bills against the
completed A14 shared A12/project accounting receipt.
"""
from __future__ import annotations
import argparse,datetime,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_autopilot import (
 CAP,PROJECT_CAP,CAMPAIGN_CAP,G1B_EXPECTED_STATUS,
 sha,jread,write,emit,run_queue,smoke_projection)
from scripts.track_r.xp_a12_g367_readiness_cpu import PC,PC_HEAD
A14_OLD_HEAD="0acb0081d992405f034f246318c3c5dbc6e3282f"
# The G3 native/measurement manifest passed CPU certification before the
# infrastructure-only controller fix. Retain exact original provenance.
A14_CERTIFIED_PREP_HEAD="f7681abfa6722d9c8c996da7a92eff73dbed98bb"
A14_CERTIFIED_PREP_SHA="3ad52d84216ccd57c14d8b59c3a643a9b89f59adfce908dac2f43d4d1639ff84"

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(
  ["git","-C",str(wt),*q],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"WORKTREE")
 prior=Path(a.prior_campaign)
 gate(prior.parent.parent==BASE and prior.parent.name.startswith(
      "xp_a12_campaign_prod_") and prior.name==
      "XP_A12_CAMPAIGN_STATE.json" and prior.is_file()
      and not prior.is_symlink() and not prior.parent.is_symlink(),
      "PRIOR_CAMPAIGN_PATH")
 p=jread(prior)
 gate(p["status"]=="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
      and p["head"]==A14_OLD_HEAD and p.get("active")==[]
      and p["G3_status"].startswith("A14_SOURCE_ABSENT")
      and float(p["stage_seconds"]["G3"])==0,
      "A14_PRIOR_GENERATION_TERMINAL")
 for stage,n in (("G4",720),("G5",400)):
  q=jread(prior.parent/f"XP_A12_{stage}_FINAL_STATUS.json")
  gate(q["status"]==f"XP_A12_{stage}_GENERATION_COMPLETE"
       and q["n_roots"]==n and q["head"]==A14_OLD_HEAD,
       "A14_PRIOR_"+stage)
 q=jread(prior.parent/"XP_A14_G2_PROGRESS.json")
 gate(q["status"]=="XP_A14_G2_ALL_2880_COMPLETE"
      and q["coverage"]["paired_full8_images"]==180
      and q["old_smoke_billed_once"] is True,
      "A14_PRIOR_G2")
 project0=float(p["project_charged_gpu_h"])
 campaign0=float(p["A12_charged_gpu_h"])
 gate(project0+CAP["G3"]<=PROJECT_CAP
      and campaign0+CAP["G3"]<=CAMPAIGN_CAP,
      "A14_SHARED_BUDGET_G3")
 prep=Path(a.prep)
 gate(prep.parent==BASE and prep.name.startswith(
      "xp_a12_g3_prepare_") and prep.is_dir() and not prep.is_symlink(),
      "G3_PREP_PATH")
 rec=jread(prep/"XP_A12_G3_PREP_RECEIPT.json")
 mf=prep/"XP_A12_G3_180_BAGS.json"
 gate(rec["status"]=="XP_A12_G3_CPU_PREP_PASS"
      and (rec["head"]==a.expected_head or
           (rec["head"]==A14_CERTIFIED_PREP_HEAD and
            rec["manifest_sha256"]==A14_CERTIFIED_PREP_SHA))
      and rec["n_bags"]==180
      and rec["two_id_smoke_required"] is True
      and rec["cap_gpu_h"]==45
      and sha(mf)==rec["manifest_sha256"],
      "G3_PREP_FROZEN")
 native=subprocess.check_output(
  ["git","-C",str(PC),"rev-parse","HEAD"],text=True).strip()
 gate(native.startswith(PC_HEAD)
      and native==rec["source_head"],"G3_NATIVE_SOURCE_IDENTITY")
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith(
      "xp_a12_g3_prod_") and not run.exists()
      and not run.is_symlink(),"NEW_G3_RUN")
 if a.preflight_only:
  print("A14_G3_SHARED_LEDGER_AND_PREP_PREFLIGHT_PASS",
        "project_prior_gpu_h",project0,
        "A12_prior_gpu_h",campaign0,
        "G3_cap_gpu_h",CAP["G3"],flush=True)
  return
 # The common run_queue queries a.a14 for G2-only timing telemetry.
 # G3 is a separately admitted continuation, not an A14 G2 pilot.
 a.a14=False
 with (BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for sub in ("roots","cost","logs","pycache"):
   (run/sub).mkdir()
  state=dict(status="G3_ADMITTED",head=a.expected_head,
    stage_seconds={k:0. for k in CAP},
    project_initial_gpu_h=project0,campaign_initial_gpu_h=campaign0,
    parent_A14_campaign_sha256=sha(prior),
    G3_status="SMOKE_REQUIRED",done={},active=[],verified_jobs={},
    GT_not_read=True,D4_outcomes_not_opened=True,
    A14_prereg_head="f31901c808954f2bee724ab8a3834e2b65ef25c9")
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
  P={"G3":rec};roots={"G3":prep}
  smoke=[dict(stage="G3",kind="SMOKE",index=i) for i in (0,1)]
  print("A14_G3_NATIVE_TWO_ID_SMOKE_BEGIN",flush=True)
  if not run_queue(a,run,"G3",smoke,roots,P,state,
                   project0,campaign0):
   emit(run,"G3_SMOKE_PARTIAL_NO_PRODUCTION")
   write(run/"XP_A14_G3_FINAL_STATUS.json",dict(
    status="XP_A14_G3_SMOKE_INCOMPLETE_STOP",
    stage_charged_gpu_h=state["stage_seconds"]["G3"]/3600,
    prior_campaign_sha256=sha(prior)))
   return
  # This is the frozen two-bag native timing projection, not a
  # surrogate model; no threshold, inference or denoiser settings change.
  try:
   projection=smoke_projection(run,"G3",smoke,P,roots,
                              state["stage_seconds"]["G3"])
  except RuntimeError as e:
   if "STAGE_SMOKE_PROJECTS_OVER_CEILING_G3" not in str(e):
    raise
   projection=dict(projection_pass=False,
       cohort_A_only=True,stage_cap_gpu_h=45,
       reason=str(e),continue_with_ceiling_limited_partial=True)
  cohort_A=bool(projection.get("cohort_A_only"))
  target=100 if cohort_A else 180
  state["G3_status"]=("COHORT_A_REDUCTION_A14" if cohort_A
                      else "FULL180_A14")
  emit(run,"G3_NATIVE_TWO_ID_SMOKE_PASS",projection=projection,
       target_bags=target)
  write(run/"XP_A14_G3_SMOKE_PROJECTION.json",dict(
      status="XP_A14_G3_TWO_NATIVE_BAGS_VERIFIED",
      projection=projection,n_smoke_bags=2,target_bags=target,
      prior_campaign_sha256=sha(prior),source_head=a.expected_head,
      native_PC_head=native))
  print("A14_G3_SMOKE_PASS_TARGET",target,flush=True)
  state["done"]={}
  production=[dict(stage="G3",kind="PROD",index=i)
              for i in range(2,target)]
  complete=run_queue(a,run,"G3",production,roots,P,state,
                     project0,campaign0)
  bags=len(state["verified_jobs"]["G3"])
  charge=state["stage_seconds"]["G3"]/3600
  gate(0<charge<=45.05 and 2<=bags<=target,
       "G3_CHARGE_AND_COMPLETENESS")
  final=dict(status=(
      "XP_A14_G3_NATIVE_FULL180_GENERATION_COMPLETE"
      if complete and target==180 else
      "XP_A14_G3_NATIVE_COHORT_A100_GENERATION_COMPLETE"
      if complete and target==100 else
      "XP_A14_G3_PARTIAL_AT_CEILING"),
      source_head=a.expected_head,PC_head=native,
      verified_bags=bags,target_bags=target,
      charged_gpu_h=charge,stage_cap_gpu_h=45.,
      project_charged_gpu_h=project0+charge,
      A12_charged_gpu_h=campaign0+charge,
      two_id_smoke_pass=True,no_GT_scoring=True,
      prior_campaign_sha256=sha(prior),
      prep_manifest_sha256=rec["manifest_sha256"])
  state["status"]=final["status"]
  state["active"]=[]
  state["project_charged_gpu_h"]=final["project_charged_gpu_h"]
  state["A12_charged_gpu_h"]=final["A12_charged_gpu_h"]
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
  write(run/"XP_A14_G3_FINAL_STATUS.json",final)
  print("A14_G3_TERMINAL",json.dumps(final,sort_keys=True),
        flush=True)

def entry():
 p=argparse.ArgumentParser()
 for name in ("expected-head","prior-campaign","prep","run"):
  p.add_argument("--"+name,required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:
  main(p.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A14_G3_NATIVE",exc,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
