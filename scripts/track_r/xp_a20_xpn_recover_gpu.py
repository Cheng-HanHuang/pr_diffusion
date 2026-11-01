#!/usr/bin/env python3
"""A20: receipt-preserving A17 XP-N recovery from 80%-VRAM co-tenancy stop.

This RECOVERS the original production outputs IN PLACE. All native workers
run at the exact original execution commit in a separate clean detached
worktree, with unchanged task manifests, seeds, solver and checkpoint.
Existing valid root receipts are independently SHA-verified; partial
chunks use the frozen native worker --resume option on the SAME GPU as
their immutable chunk metadata. Any unreceipted root dir fails closed.
No GT, labels, PSNR, CLIP, prediction or >=4 scoring access.

A pressure stop signals ONLY our worker process group, charges its whole
process wall time, and returns it to pending. Never touches foreign PIDs.
A retry does not delete files or reset costs. Project lock protects
other Track-R controller ledgers. Final and recovery receipts are separate.
"""
from __future__ import annotations
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a18_xpn_contract import (
    PHASES,Y_SHA,AUDIT_SHA,MODEL_SHA,validate_xpn_source)
from scripts.track_r.xp_a18_xpn_autopilot import (
    FAST,PHASE_BATCH,PHASE_FILES,STAGE_CAP_H,PROJECT_CAP_H,
    REQUEST_MIB,RESERVE_MIB,RESERVATION_PER_NEW_WORKER_H,
    available,atomic,now,emit,git)
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a15_cp2_d4_cpu import jread,sha
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

EXPECTED_ORIGINAL=BASE/"xp_a18_xpn_prod_20261031T225748Z"
EXPECTED_TASKS=BASE/"xp_a18_xpn_tasks_20261031T224645Z"
EXPECTED_PILOT_SHA="93ac0c8e683f95fed72fd9c834995b25f23947a511db31c239cc1ac021caded0"
ORIGINAL_STATUS="STOP_XPN_CONTROLLER_OR_NATIVE_EXCEPTION"
PROJECT_LOCK=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
RESUME_LIMIT_PER_JOB=4
SAMPLE_SECONDS=5
POLL_SECONDS=25
STOP=False
def sigstop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGINT,sigstop)
signal.signal(signal.SIGTERM,sigstop)

def path_for(run,job):
 return run/"production"/job["phase"]/f"chunk_{job['start']:05d}_{job['count']:03d}"

def job_id(job):return f"{job['phase']}_{job['start']:05d}"

def check_chunk(run,job,manifest,head,require_complete=False):
 """Validate EVERY existing result SHA/metadata and no unreceipted root dirs."""
 path,digest,rows=manifest
 dest=path_for(run,job)
 if not dest.exists():
  gate(not require_complete,"A20_MISSING_EXPECTED_CHUNK_"+job_id(job))
  return dict(status="NEW",reused=0,original_gpu=None)
 gate(dest.is_dir() and not dest.is_symlink(),
      "A20_CHUNK_DIRECTORY_"+job_id(job))
 meta=jread(dest/"R0_5_CHUNK_META.json")
 gate(meta["schema"]=="prdiff.track_r.r0.5.fast-chunk.v1"
      and meta["status"]=="R0_5_CHUNK_CONFIG"
      and meta["head"]==head and meta["root_sha256"]==digest
      and meta["start"]==job["start"] and meta["count"]==job["count"]
      and meta["gpu"] in range(4)
      and meta["uids"]==[r["uid"] for r in rows[job["start"]:job["start"]+job["count"]]],
      "A20_FROZEN_CHUNK_METADATA_"+job_id(job))
 selected=rows[job["start"]:job["start"]+job["count"]]
 got=0;expected=set()
 for row in selected:
  folder=root_folder(dest,row)
  expected.add(folder.name)
  if folder.exists():
   gate(folder.is_dir() and not folder.is_symlink(),
        "A20_PARTIAL_ROOT_DIRECTORY_"+job_id(job))
   rec=validate_complete(folder,row,head,digest)
   gate(rec["gt_access"] is False and
        rec["method"]=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
        and rec["fast_source_head"]==
            "a9758b38906b7c7950fa04491f37f6634948db26",
        "A20_PARTIAL_ROOT_RECEIPT_"+job_id(job))
   got+=1
 others={d.name for d in dest.glob("root_*")}-expected
 gate(not others,"A20_UNEXPECTED_ROOT_FOLDERS_"+job_id(job))
 complete=dest/"R0_5_CHUNK_COMPLETE.json"
 if complete.exists():
  c=jread(complete)
  gate(c["status"]=="R0_5_CHUNK_PASS"
       and c["head"]==head and c["start"]==job["start"]
       and c["count"]==job["count"]
       and c["new"]+c["reused"]==job["count"]
       and got==job["count"]
       and c["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
       "A20_CHUNK_COMPLETE_RECEIPT_"+job_id(job))
  return dict(status="COMPLETE",reused=got,original_gpu=meta["gpu"],
              chunk_sha256=sha(complete))
 gate(not require_complete,"A20_MISSING_CHUNK_COMPLETE_"+job_id(job))
 gate(got<job["count"],"A20_COMPLETE_ROOTS_MISSING_CHUNK_FINAL_"+job_id(job))
 return dict(status="PARTIAL",reused=got,original_gpu=meta["gpu"])

def reconstruct(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A20_RECOVERY_CPU_ONLY")
 gate(git(wt,"rev-parse","HEAD")==a.expected_head
      and not git(wt,"status","--porcelain"),
      "A20_RECOVERY_CLEAN_EXECUTOR_HEAD")
 run=EXPECTED_ORIGINAL;tasks=EXPECTED_TASKS
 gate(run.is_dir() and tasks.is_dir()
      and not run.is_symlink() and not tasks.is_symlink(),
      "A20_FROZEN_RUN_AND_TASKS")
 statefile=run/"XP_N_CONTROLLER_STATE.json"
 finalfile=run/"XP_N_FINAL_STATUS.json"
 old=jread(statefile);final=jread(finalfile)
 head=old["expected_head"]
 gate(len(head)==40 and head==final["expected_head"]
      and old["status"]==ORIGINAL_STATUS
      and final["status"]==ORIGINAL_STATUS
      and not old["active"]
      and len(old["done"])==6
      and final["n_completed_roots"]==120
      and old["XP_N_4_to_63_outcomes_sealed"] is True
      and old["GT_not_read"] is True
      and old["no_GT_scoring_or_prediction_access"] is True,
      "A20_ORIGINAL_TERMINAL_AND_120")
 # Verify original failure was exactly one memory-pressure stop; the
 # other interrupted workers were stopped by the controller teardown.
 failures=old["failures"]
 events=run/"XP_N_EVENTS.jsonl"
 event_rows=[json.loads(line) for line in events.read_text().splitlines()]
 pressure={e["job"] for e in event_rows
           if e.get("kind")=="ABOVE_80PCT_TERMINATE_OWN_WORKER"}
 gate(pressure=={"A18_XPN_PILOT_0_3_00080"} and
      any(x.get("job")=="A18_XPN_PILOT_0_3_00080"
          and x.get("rc")==-15 for x in failures),
      "A20_ORIGINAL_PRESSURE_STOP_PROVENANCE")
 receipts=list(sorted((run/"attempt_cost").glob("XP_N_GPU_ATTEMPT_*.json")))
 gate(len(receipts)==10,"A20_ORIGINAL_10_ATTEMPTS")
 attempts={p.name.removeprefix("XP_N_GPU_ATTEMPT_").removesuffix(".json"):jread(p)
          for p in receipts}
 gate(len(attempts)==len(receipts),"A20_UNIQUE_ORIGINAL_ATTEMPTS")
 charged=sum(float(x["elapsed_process_gpu_h"]) for x in attempts.values())
 gate(abs(charged-float(old["attempt_cost_gpu_h"]))<1e-8
      and abs(charged-float(final["charged_stage_gpu_h"]))<1e-8
      and all(x["rc"] in (0,-15) and 0<x["elapsed_process_gpu_h"]<1.
              for x in attempts.values()),
      "A20_ORIGINAL_LEDGER_RECONCILED")
 manifests={};jobs=[]
 prep=jread(tasks/"XP_N_TASKS_PREP_RECEIPT.json")
 gate(prep["status"]=="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
      and prep["n_roots"]==6400 and prep["n_images"]==100
      and old["manifest_producer_head"]==prep["expected_head"]
      and prep["source_gt_y_audit_sha256"]==AUDIT_SHA
      and prep["source_y_manifest_sha256"]==Y_SHA
      and old["phase_manifests_sha256"]["A18_XPN_PILOT_0_3"]==EXPECTED_PILOT_SHA,
      "A20_FROZEN_PREP_OR_SOURCE_SHA")
 for phase in PHASES:
  path=tasks/PHASE_FILES[phase];man=jread(path);digest=sha(path)
  gate(prep["phase_files"][phase]["sha256"]==digest
       and old["phase_manifests_sha256"][phase]==digest
       and man["head"]==prep["expected_head"],
       "A20_FROZEN_MANIFEST_"+phase)
  manifests[phase]=(path,digest,man["rows"])
  for start in range(0,len(man["rows"]),PHASE_BATCH[phase]):
   job=dict(phase=phase,start=start,count=PHASE_BATCH[phase])
   validate_xpn_source(path,digest,man,man["rows"],prep["expected_head"],
                        start,job["count"])
   jobs.append(job)
 gate(len(jobs)==90,"A20_90_JOBS")
 snapshot={};num_roots=0;complete=0;partial=0
 for job in jobs:
  info=check_chunk(run,job,manifests[job["phase"]],head)
  key=job_id(job);snapshot[key]=info
  num_roots+=info["reused"]
  complete+=info["status"]=="COMPLETE"
  partial+=info["status"]=="PARTIAL"
  if key in old["done"]:
   gate(info["status"]=="COMPLETE" and
        info["chunk_sha256"]==old["done"][key],
        "A20_REPLAY_ORIGINAL_COMPLETED_CHUNK_SHA_"+key)
  else:
   gate(info["status"]!="COMPLETE",
        "A20_UNCHARGED_ORIGINAL_COMPLETE_CHUNK_"+key)
 gate(num_roots==163 and complete==6 and partial==4
      and set(old["done"])=={j for j,info in snapshot.items()
                            if info["status"]=="COMPLETE"},
      "A20_ORIGINAL_EXACT_163_SIX_COMPLETE_FOUR_PARTIAL")
 gate(any(x["v5_gate"] for x in [old])
      and old["v5_gate"]["status"]=="V5_CODE_SHA_FROZEN_PRE_FIRST_XPN_ROOT",
      "A20_PROSPECTIVE_CODE_FREEZE_NOT_LAPSE")
 baseline=float(final["project_total_gpu_h"])
 gate(abs(baseline-float(old["project_baseline_gpu_h"])-
            float(old["replay_gpu_h"])-charged)<1e-7,
      "A20_PROJECT_CHARGE_RECONCILED")
 gate(0<charged<STAGE_CAP_H and baseline+STAGE_CAP_H-charged<PROJECT_CAP_H,
      "A20_20H_AND_PROJECT_CAP")
 gate(sha(ROOT/"models/imagenet256.pt")==MODEL_SHA,
      "A20_IMAGENET_CHECKPOINT")
 return wt,run,tasks,head,manifests,jobs,snapshot,old,final,charged,baseline

def validate_native_checkout(wt,head,checkout):
 gate(checkout.parent==ROOT and checkout.name.startswith(
        "pr_diffusion_track_r_xpn_native_"),
      "A20_PINNED_WORKTREE_PATH")
 if not checkout.exists():
  subprocess.run(["git","-C",str(wt),"worktree","add","--detach",
         str(checkout),head],check=True,timeout=120)
 gate(git(checkout,"rev-parse","HEAD")==head and
      not git(checkout,"status","--porcelain"),
      "A20_NATIVE_SOURCE_EXECUTION_COMMIT_EXACT")
 return checkout

def main(a):
 wt,run,tasks,head,manifests,jobs,snapshot,old,original,prior,project=reconstruct(a)
 if a.preflight_only:
  cards=collect()
  eligible=[c for c in available(cards) if choose_gpu(
      [c],REQUEST_MIB,safety_margin_mib=RESERVE_MIB,
      fraction=.80,clean_only=False)[0] is not None]
  print("A20_XPN_RECOVERY_PREFLIGHT_PASS",
        "ROOTS_REUSABLE",163,"COMPLETE_CHUNKS",6,
        "PARTIAL_CHUNKS",4,"PRIOR_CHARGED_H",round(prior,9),
        "80PCT_ELIGIBLE_GPUS",[c.index for c in eligible],
        "STATUS",("READY" if eligible else "WAIT_FOREIGN_GPU_VRAM"),
        flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a20_xpn_recovery_")
      and not out.exists() and not out.is_symlink(),
      "A20_NEW_RECOVERY_CONTROLLER_ROOT")
 # No competing recovery controller: a prior state, even after
 # failure, must be explicitly reconciled before another submission.
 priorruns=list(BASE.glob("xp_a20_xpn_recovery_*/XP_N_RECOVERY_STATE.json"))
 gate(not priorruns,"A20_EXISTING_RECOVERY_MUST_RECONCILE_FIRST")
 out.mkdir(mode=0o700)
 for name in ("logs","attempt_cost","pycache"):
  (out/name).mkdir(mode=0o700)
 checkout=ROOT/("pr_diffusion_track_r_xpn_native_"+out.name.rsplit("_",1)[-1])
 checkout=validate_native_checkout(wt,head,checkout)
 state=dict(status="A20_XPN_RECOVERY_WAIT_GPU_VRAM",
   original_run=str(run),original_execution_head=head,
   recovery_controller_head=a.expected_head,
   original_state_sha256=sha(run/"XP_N_CONTROLLER_STATE.json"),
   original_final_sha256=sha(run/"XP_N_FINAL_STATUS.json"),
   original_event_sha256=sha(run/"XP_N_EVENTS.jsonl"),
   original_charged_gpu_h=prior,project_total_before_recovery_gpu_h=project,
   recovered_from_root_receipts=163,
   phase_manifest_sha256={p:m[1] for p,m in manifests.items()},
   done={k:v["chunk_sha256"] for k,v in snapshot.items()
         if v["status"]=="COMPLETE"},
   n_native_pressure_stops=0,charged_new_gpu_h=0.,active=[],
   attempts_per_chunk={},failures=[],v5_gate=old["v5_gate"],
   no_GT_scoring=True,outcomes_ge4_sealed=True)
 def save():
  atomic(out/"XP_N_RECOVERY_STATE.json",state)
 def record_active(active):
  state["active"]=[dict(job=k,gpu=x["gpu"],pid=x["proc"].pid)
                   for k,x in active.items()]
  save()
 save()
 # Lock across A12/other Track R controllers; no foreign process killed.
 with PROJECT_LOCK.open("a+") as lock:
  while True:
   try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if STOP or (out/"STOP_XPN_RECOVERY").exists():
     state["status"]="STOP_RECOVERY_BEFORE_GPU_LOCK";save();return
    time.sleep(30)
  active={};stop_reason=None
  def final_charge(key,v):
   elapsed=(time.monotonic()-v["started"])/3600.
   proc=v["proc"];rc=proc.returncode
   previous=state["charged_new_gpu_h"]
   state["charged_new_gpu_h"]=previous+elapsed
   atomic(out/"attempt_cost"/
       f"{key}_try{v['attempt']:02d}.json",dict(
     status="A20_NATIVE_ATTEMPT_CHARGED_UNMODIFIED_HEAD",
     job=key,attempt=v["attempt"],physical_gpu=v["gpu"],
     original_execution_head=head,rc=rc,
     charged_process_gpu_h=elapsed,foreign_pids_seen=sorted(v["foreign"]),
     memory_pressure_own_sigterm=v["pressure"],
     telemetry_unknown=v["unknown"],peak_used_mib=v["peak"],
     samples=v["samples"],native_log=str(v["log"]),utc=now()))
   if v["pressure"]:state["n_native_pressure_stops"]+=1
   return rc
  def kill_ours():
   for v in active.values():
    if v["proc"].poll() is None:
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
  def used_h():
   return prior+state["charged_new_gpu_h"]+sum(
       (time.monotonic()-v["started"])/3600 for v in active.values())
  def safe_budget(extra=0):
   committed=used_h()+(len(active)+extra)*RESERVATION_PER_NEW_WORKER_H
   return committed<=STAGE_CAP_H and (
     project+committed-prior<=PROJECT_CAP_H)
  status="RUNNING"
  try:
   for phase in PHASES:
    phasejobs=[j for j in jobs if j["phase"]==phase]
    while not all(job_id(j) in state["done"] for j in phasejobs):
     cards=collect(own_roots=[v["proc"].pid for v in active.values()])
     for key,v in list(active.items()):
      card=next((c for c in cards if c.index==v["gpu"]),None)
      if card is None or card.visibility!="known":
       v["unknown"]=True
      else:
       v["foreign"].update(card.foreign_pids)
       v["peak"]=max(v["peak"],card.used_mib)
       v["samples"]+=1
       if card.used_mib>int(.80*card.total_mib) and not v["pressure"]:
        v["pressure"]=True
        emit(out,"GPU_PRESSURE_PAUSE_OWN_ONLY",job=key,gpu=v["gpu"],
             used_mib=card.used_mib,total_mib=card.total_mib)
        try:os.killpg(v["proc"].pid,signal.SIGTERM)
        except ProcessLookupError:pass
      rc=v["proc"].poll()
      if rc is None:continue
      v["proc"].wait()
      v["handle"].close()
      final_charge(key,v)
      del active[key]
      if v["unknown"]:
       raise RuntimeError("A20_TELEMETRY_UNKNOWN_"+key)
      if v["pressure"] and rc==-15:
       # Exactly the existing A10 contract: own worker interrupted,
       # valid partial roots reused on the same native GPU later.
       info=check_chunk(run,v["job"],manifests[phase],head)
       gate(info["status"]!="COMPLETE","A20_PRESSURE_BUT_COMPLETE_"+key)
       emit(out,"PRESSURE_PARTIAL_PRESERVED",job=key,
            reused_roots=info["reused"])
       continue
      if rc!=0:
       raise RuntimeError("A20_NATIVE_RC_"+str(rc)+"_"+key)
      info=check_chunk(run,v["job"],manifests[phase],head,
                       require_complete=True)
      state["done"][key]=info["chunk_sha256"]
      emit(out,"NATIVE_CHUNK_COMPLETED_SHA",job=key,
           chunk_sha256=info["chunk_sha256"],reused_roots=info["reused"])
      if len(state["done"])%10==0:
       print("A20_RECOVERY_CHUNKS",len(state["done"]),"/90",
             "STAGE_CHARGED_H",round(used_h(),6),flush=True)
      record_active(active)
     if STOP or (out/"STOP_XPN_RECOVERY").exists():
      stop_reason="STOP_RECOVERY_OPERATOR";break
     if not safe_budget(1):
      stop_reason="STOP_RECOVERY_20H_OR_PROJECT_CAP";break
     occupied={v["gpu"] for v in active.values()}
     candidates=[c for c in available(cards) if c.index not in occupied]
     while len(active)<4 and safe_budget(1):
      j=None;gpu=None;info=None
      # Finish original partial chunks on their IMMUTABLE GPU. New
      # chunks may use other admissible GPUs while partials are blocked.
      for p in (True,False):
       for candidate in phasejobs:
        key=job_id(candidate)
        if key in state["done"] or key in active:continue
        chk=check_chunk(run,candidate,manifests[phase],head)
        if (chk["status"]=="PARTIAL")!=p:continue
        if p:
         ix=chk["original_gpu"]
         card=next((c for c in candidates if c.index==ix),None)
         pick=(choose_gpu([card],REQUEST_MIB,
                safety_margin_mib=RESERVE_MIB,fraction=.80,
                clean_only=False)[0] if card else None)
        else:
         pick,_=choose_gpu(candidates,REQUEST_MIB,
             safety_margin_mib=RESERVE_MIB,fraction=.80,clean_only=False)
        if pick is not None:
         j=candidate;gpu=pick;info=chk;break
       if j:break
      if j is None:break
      key=job_id(j)
      attempt=state["attempts_per_chunk"].get(key,0)+1
      gate(attempt<=RESUME_LIMIT_PER_JOB,
           "A20_EXCESS_PRESSURE_RETRIES_"+key)
      dest=path_for(run,j);mf,digest,_=manifests[phase]
      cmd=[str(FAST),str(checkout/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
           "--expected-head",head,
           "--roots",str(mf),"--roots-sha",digest,
           "--roots-source-head",old["manifest_producer_head"],
           "--out",str(dest),"--start",str(j["start"]),
           "--count",str(j["count"]),"--physical-gpu",str(gpu),
           "--dataset","imagenet","--oversample","2.0"]
      if info["status"]=="PARTIAL":cmd.append("--resume")
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
           PYTHONPATH=str(checkout),PYTHONDONTWRITEBYTECODE="1",
           PYTHONNOUSERSITE="1",OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
           NUMEXPR_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",
           TMPDIR=str(ROOT/"tmp"))
      log=out/"logs"/f"{key}_try{attempt:02d}.log"
      handle=log.open("x")
      proc=subprocess.Popen(cmd,cwd=str(checkout),env=env,
           stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
      active[key]=dict(job=j,gpu=gpu,proc=proc,handle=handle,
           log=log,attempt=attempt,started=time.monotonic(),
           pressure=False,foreign=set(),unknown=False,peak=0,samples=0)
      state["attempts_per_chunk"][key]=attempt
      emit(out,"ORIGINAL_HEAD_NATIVE_RESUME_OR_NEW",job=key,
           gpu=gpu,resume=info["status"]=="PARTIAL",
           already_valid_roots=info["reused"],pid=proc.pid)
      candidates=[c for c in candidates if c.index!=gpu]
      record_active(active)
     if not active:
      state["status"]="A20_XPN_WAIT_FOREIGN_GPU_VRAM"
      save()
      if int(time.monotonic())%300<30:
       print("A20_XPN_WAIT_FOREIGN_GPU_VRAM",
             "DONE",len(state["done"]),"/90",
             "ROOTS_CHARGED_H",round(used_h(),6),flush=True)
      time.sleep(POLL_SECONDS)
     else:
      state["status"]="A20_XPN_RUNNING_NATIVE"
      record_active(active)
      time.sleep(SAMPLE_SECONDS)
    if stop_reason:break
    emit(out,"PHASE_COMPLETE_GT_FREE_SEALED",phase=phase)
    print("A20_XPN_PHASE_COMPLETE",phase,flush=True)
   if len(state["done"])==90:
    status="A20_XPN_ALL6400_NATIVE_GENERATION_COMPLETE_SEALED"
   elif stop_reason:
    status=stop_reason
   else:
    status="STOP_A20_INCOMPLETE_UNEXPECTED"
  except BaseException as e:
   status="STOP_A20_RECOVERY_ERROR"
   state["failures"].append(type(e).__name__+":"+str(e)[:1000])
   raise
  finally:
   kill_ours()
   for key,v in list(active.items()):
    v["proc"].wait()
    v["handle"].close()
    final_charge(key,v)
    del active[key]
   state["status"]=status
   record_active(active)
   atomic(out/"XP_N_RECOVERY_FINAL_STATUS.json",dict(
      status=status,expected_recovery_head=a.expected_head,
      original_execution_head=head,original_run=str(run),
      initial_120_complete_roots_plus_43_partial=163,
      completed_roots=sum(j["count"] for j in jobs
                          if job_id(j) in state["done"]),
      n_chunks_complete=len(state["done"]),
      prior_charged_GPU_h=prior,
      new_charged_GPU_h=state["charged_new_gpu_h"],
      stage_total_GPU_h=prior+state["charged_new_gpu_h"],
      project_total_GPU_h=project+state["charged_new_gpu_h"],
      stage_GPU_cap_h=STAGE_CAP_H,project_cap_GPU_h=PROJECT_CAP_H,
      all_manifests_scientifically_unchanged=True,
      frozen_native_execution_head=head,
      scores_ge4_still_sealed=True,GT_not_read=True,
      initial_state_sha256=state["original_state_sha256"],
      failures=state["failures"]))
   print("A20_XPN_RECOVERY_TERMINAL",status,
         "CHUNKS",len(state["done"]),"/90",
         "GPU_H_TOTAL",round(prior+state["charged_new_gpu_h"],6),
         flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A20_XPN_RECOVERY",e,flush=True)
  sys.exit(8)
