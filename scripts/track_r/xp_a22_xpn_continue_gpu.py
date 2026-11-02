#!/usr/bin/env python3
"""A18 28-GPU-hour XP-N continuation after accepted A20 terminal stop.

One-chunk smoke, then up to four workers. Full SHA reconciliation of
original A18 attempts, A20 attempts, 90 chunks and every existing native
root BEFORE any GPU is admitted. Resume partials only with the exact old
GPU and immutable native metadata. All scientific inputs and execution
commit unchanged. No GT, scoring, CLIP, predictions or protocol edits.

Reservation is for NEW admissions ONLY, never a reason to kill workers.
The hard watchdog checks CHARGED + LIVE process-wall GPU-hours and
terminates ONLY our workers if 28 GPU-h total is about to be reached.
All attempts, including failures and SIGTERM, are charged permanently.
"""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a20_xpn_recover_gpu import (
    EXPECTED_ORIGINAL,EXPECTED_TASKS,PROJECT_LOCK,
    check_chunk,path_for,job_id,validate_native_checkout)
from scripts.track_r.xp_a18_xpn_autopilot import (
    FAST,PHASE_BATCH,PHASE_FILES,REQUEST_MIB,RESERVE_MIB,
    RESERVATION_PER_NEW_WORKER_H,available,atomic,emit,now,git)
from scripts.track_r.xp_a18_xpn_contract import (
    PHASES,Y_SHA,AUDIT_SHA,MODEL_SHA,validate_xpn_source)
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

STAGE_CAP_H=28.
PROJECT_CAP_H=400.
SAMPLE_SECONDS=2.
POLL_SECONDS=20.
MAX_RETRIES_PER_JOB=4
STAGE_HARD_STOP_MARGIN_H=.005
STOP=False

def on_stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,on_stop)
signal.signal(signal.SIGINT,on_stop)

def verify_sources(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A22_XPN_CPU_ORCHESTRATOR_ONLY")
 gate(git(wt,"rev-parse","HEAD")==a.expected_head
      and not git(wt,"status","--porcelain"),
      "A22_XPN_CLEAN_HEAD")
 prior_states=sorted(BASE.glob("xp_a20_xpn_recovery_*/XP_N_RECOVERY_STATE.json"))
 gate(len(prior_states)==1,"A22_XPN_EXACT_ONE_A20_TERMINAL")
 a20root=prior_states[0].parent
 a20=jread(prior_states[0])
 a20final=jread(a20root/"XP_N_RECOVERY_FINAL_STATUS.json")
 a18=jread(EXPECTED_ORIGINAL/"XP_N_CONTROLLER_STATE.json")
 a18final=jread(EXPECTED_ORIGINAL/"XP_N_FINAL_STATUS.json")
 gate(a20["status"]==a20final["status"]==
      "STOP_RECOVERY_20H_OR_PROJECT_CAP"
      and a18["status"]==a18final["status"]==
          "STOP_XPN_CONTROLLER_OR_NATIVE_EXCEPTION"
      and not a20["active"] and not a18["active"]
      and not a20["failures"] and
      len(a20["done"])==72 and len(a18["done"])==6
      and a20["no_GT_scoring"] is True
      and a20["outcomes_ge4_sealed"] is True,
      "A22_XPN_PINNED_TERMINAL_A20")
 head=a20["original_execution_head"]
 gate(head==a18["expected_head"]
      and head==a20final["original_execution_head"]
      and a20["original_charged_gpu_h"]==a18["attempt_cost_gpu_h"],
      "A22_XPN_EXECUTION_HEAD_AND_A18_CHARGE")
 # The original ten receipts and all A20 retry receipts must be
 # reconciled against their independent terminal ledger values.
 oldreceipts=list((EXPECTED_ORIGINAL/"attempt_cost").glob("*.json"))
 gate(len(oldreceipts)==10,"A22_XPN_A18_ATTEMPT_COUNT")
 oldsum=sum(float(jread(p)["elapsed_process_gpu_h"]) for p in oldreceipts)
 gate(abs(oldsum-float(a18["attempt_cost_gpu_h"]))<1e-8,
      "A22_XPN_A18_ALL_GPU_ATTEMPT_COST")
 a20receipts=list((a20root/"attempt_cost").glob("*.json"))
 gate(len(a20receipts)>0 and
      len({p.name for p in a20receipts})==len(a20receipts),
      "A22_XPN_A20_ATTEMPT_RECEIPT_UNIVERSE")
 new_sum=0.
 for p in a20receipts:
  t=jread(p)
  gate(t["status"]=="A20_NATIVE_ATTEMPT_CHARGED_UNMODIFIED_HEAD"
       and t["original_execution_head"]==head
       and t["charged_process_gpu_h"]>0
       and t["rc"] in (0,-15),
       "A22_XPN_A20_ATTEMPT_RECEIPT_"+p.name)
  new_sum+=float(t["charged_process_gpu_h"])
 gate(abs(new_sum-float(a20["charged_new_gpu_h"]))<1e-8
      and abs(new_sum-float(a20final["new_charged_GPU_h"]))<1e-8
      and abs(oldsum-float(a20final["prior_charged_GPU_h"]))<1e-8,
      "A22_XPN_A20_ALL_GPU_ATTEMPT_COST")
 stage_used=oldsum+new_sum
 project_used=float(a20final["project_total_GPU_h"])
 gate(abs(stage_used-float(a20final["stage_total_GPU_h"]))<1e-8
      and abs(project_used-(float(a18final["project_total_gpu_h"])+new_sum))<1e-8
      and 18.7<stage_used<20.
      and project_used<PROJECT_CAP_H-10.,
      "A22_XPN_CHARGED_INITIAL_RECONCILIATION")
 prep=jread(EXPECTED_TASKS/"XP_N_TASKS_PREP_RECEIPT.json")
 gate(prep["status"]=="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
      and prep["n_roots"]==6400
      and prep["source_y_manifest_sha256"]==Y_SHA
      and prep["source_gt_y_audit_sha256"]==AUDIT_SHA
      and prep["native_d1_replay_pixel_exact"] is True
      and a18["manifest_producer_head"]==prep["expected_head"],
      "A22_XPN_FROZEN_SOURCE")
 gate(sha(ROOT/"models/imagenet256.pt")==MODEL_SHA and FAST.is_file(),
      "A22_XPN_NATIVE_IMAGE_MODEL")
 manifests={};jobs=[];snapshot={}
 for phase in PHASES:
  filename=PHASE_FILES[phase]
  p=EXPECTED_TASKS/filename
  m=jread(p);digest=sha(p)
  gate(prep["phase_files"][phase]["sha256"]==digest
       and a20["phase_manifest_sha256"][phase]==digest
       and a18["phase_manifests_sha256"][phase]==digest
       and m["head"]==prep["expected_head"],
       "A22_XPN_MANIFEST_SHA_"+phase)
  manifests[phase]=(p,digest,m["rows"])
  step=PHASE_BATCH[phase]
  for start in range(0,len(m["rows"]),step):
   j=dict(phase=phase,start=start,count=step)
   validate_xpn_source(p,digest,m,m["rows"],
                       prep["expected_head"],start,step)
   info=check_chunk(EXPECTED_ORIGINAL,j,manifests[phase],head)
   key=job_id(j);snapshot[key]=info
   if key in a20["done"]:
    gate(info["status"]=="COMPLETE"
         and info["chunk_sha256"]==a20["done"][key],
         "A22_XPN_A20_DONE_SHA_"+key)
   else:
    gate(info["status"]!="COMPLETE","A22_XPN_UNCHARGED_COMPLETION_"+key)
   jobs.append(j)
 gate(len(jobs)==90 and len(a20["done"])==72
      and sum(x["count"] for x in jobs if
              snapshot[job_id(x)]["status"]=="COMPLETE")==4672
      and all(snapshot[k]["status"]=="COMPLETE" for k in a20["done"]),
      "A22_XPN_ALL_90_CHUNKS_SHA_INVENTORY")
 # v5 forecast successfully registered and independently reproduced by
 # planner A18 before any held-run scoring; no rerun/read needed here.
 gate(a20["v5_gate"]["status"]=="V5_CODE_SHA_FROZEN_PRE_FIRST_XPN_ROOT",
      "A22_XPN_FROZEN_PROSPECTIVE_V5")
 # One prior A22 controller, even if terminated without receipt,
 # must be explicitly reconciled. Never create a fresh ledger silently.
 gate(not list(BASE.glob("xp_a22_xpn_continue_*/XP_N_CONTINUATION_STATE.json")),
      "A22_XPN_PRIOR_CONTROLLER_REQUIRES_RECONCILIATION")
 return wt,head,a20root,manifests,jobs,snapshot,oldsum,new_sum,stage_used,project_used

def main(a):
 wt,head,a20root,manifests,jobs,snapshot,oldsum,newsum,prior,project=verify_sources(a)
 if a.preflight_only:
  candidates=[c.index for c in available(collect()) if
      choose_gpu([c],REQUEST_MIB,safety_margin_mib=RESERVE_MIB,
                 fraction=.80,clean_only=False)[0] is not None]
  partial=[(key,v["reused"],v["original_gpu"]) for key,v in snapshot.items()
           if v["status"]=="PARTIAL"]
  print("A22_XPN_28H_CONTINUATION_PREFLIGHT_PASS",
       "COMPLETE",72,"REMAINING",18,
       "ORIGINAL_GPU_H",round(oldsum,9),
       "A20_GPU_H",round(newsum,9),
       "STAGE_GPU_H",round(prior,9),
       "H_REMAINING",round(STAGE_CAP_H-prior,9),
       "PROJECT_GPU_H",round(project,7),
       "ELIGIBLE_GPU",candidates,"PARTIAL",partial,flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a22_xpn_continue_")
      and not out.exists() and not out.is_symlink(),
      "A22_XPN_NEW_ROOT")
 out.mkdir(mode=0o700)
 for sub in ("logs","attempt_cost"):
  (out/sub).mkdir(mode=0o700)
 original_checkout=ROOT/("pr_diffusion_track_r_xpn_native_"+out.name.rsplit("_",1)[-1])
 checkout=validate_native_checkout(wt,head,original_checkout)
 state=dict(status="A22_WAIT_GPU",expected_head=a.expected_head,
    original_execution_head=head,original_run=str(EXPECTED_ORIGINAL),
    previous_A20_run=str(a20root),
    initial_A18_charged_gpu_h=oldsum,initial_A20_charged_gpu_h=newsum,
    initial_stage_charged_gpu_h=prior,
    project_charged_before_A22_gpu_h=project,
    stage_cap_gpu_h=STAGE_CAP_H,project_cap_gpu_h=PROJECT_CAP_H,
    n_initial_complete_chunks=72,
    initial_receipt_roots_reused=sum(v["reused"] for v in snapshot.values()),
    original_manifest_sha256={k:v[1] for k,v in manifests.items()},
    done={k:v["chunk_sha256"] for k,v in snapshot.items()
          if v["status"]=="COMPLETE"},
    smoke_chunk=None,smoke_pass=False,
    active=[],attempts={},charged_gpu_h=0.,
    n_pressure_pauses=0,failures=[],all_held_GT_outcomes_unread=True,
    v5_registered_pre_scoring=True)
 def persist():atomic(out/"XP_N_CONTINUATION_STATE.json",state)
 def emit_state(active):
  state["active"]=[dict(key=k,pid=v["proc"].pid,gpu=v["gpu"])
                   for k,v in active.items()]
  persist()
 persist()
 with PROJECT_LOCK.open("a+") as lk:
  while True:
   try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if STOP or (out/"STOP_XPN_CONTINUATION").exists():
     state["status"]="STOP_OPERATOR_WAITING_PROJECT_LOCK"
     persist();return
    time.sleep(20)
  active={}
  def used():
   return prior+state["charged_gpu_h"]+sum(
       (time.monotonic()-v["start"])/3600 for v in active.values())
  def project_total():return project+used()-prior
  def can_admit():
   # Reservation applies ONLY to new workers, never stops an active one.
   projected=used()+(len(active)+1)*RESERVATION_PER_NEW_WORKER_H
   return projected<=STAGE_CAP_H and (
       project+projected-prior<=PROJECT_CAP_H)
  def kill_ours():
   for v in active.values():
    if v["proc"].poll() is None:
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
  def charge(key,v):
   cost=(time.monotonic()-v["start"])/3600
   state["charged_gpu_h"]+=cost
   atomic(out/"attempt_cost"/f"{key}_try{v['attempt']:02d}.json",dict(
       status="A22_28H_ATTEMPT_CHARGED",key=key,
       attempt=v["attempt"],rc=v["proc"].returncode,
       physical_gpu=v["gpu"],process_wall_gpu_h=cost,
       original_execution_head=head,
       memory_pressure_termination=v["pressure"],
       telemetry_unknown=v["unknown"],
       peak_memory_mib=v["peak"],
       foreign_pids_seen=sorted(v["foreign"]),
       n_samples=v["samples"],native_log=str(v["log"]),
       utc=now()))
   return cost
  stopped=None;smoke=next(j for j in jobs if job_id(j) not in state["done"])
  state["smoke_chunk"]=job_id(smoke)
  emit_state(active)
  try:
   while len(state["done"])<90:
    cards=collect(own_roots=[v["proc"].pid for v in active.values()])
    # HARD watchdog: must stop all own workers at cap, even if
    # co-tenancy makes a single chunk take much longer than expected.
    if (used()>=STAGE_CAP_H-STAGE_HARD_STOP_MARGIN_H or
        project_total()>=PROJECT_CAP_H-STAGE_HARD_STOP_MARGIN_H):
     stopped="STOP_A22_28H_HARD_WATCHDOG"
     emit(out,"HARD_28H_WATCHDOG_STOP_OWN",charged_live_h=used())
     break
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
       emit(out,"PRESSURE_STOP_OWN",key=key,gpu=v["gpu"])
       try:os.killpg(v["proc"].pid,signal.SIGTERM)
       except ProcessLookupError:pass
     rc=v["proc"].poll()
     if rc is None:continue
     v["proc"].wait();v["handle"].close()
     charge(key,v)
     del active[key]
     if v["unknown"]:
      raise RuntimeError("A22_NATIVE_TELEMETRY_LOSS_"+key)
     if v["pressure"] and rc==-15:
      info=check_chunk(EXPECTED_ORIGINAL,v["job"],manifests[v["job"]["phase"]],head)
      gate(info["status"]!="COMPLETE","A22_PRESSURE_AFTER_COMPLETE_"+key)
      emit(out,"PRESSURE_PARTIAL_SHA_PRESERVED",key=key,
           valid_receipts=info["reused"])
      emit_state(active)
      continue
     if rc!=0:
      raise RuntimeError("A22_NATIVE_RC_"+str(rc)+"_"+key)
     info=check_chunk(EXPECTED_ORIGINAL,v["job"],manifests[v["job"]["phase"]],
                      head,require_complete=True)
     state["done"][key]=info["chunk_sha256"]
     if key==state["smoke_chunk"]:
      state["smoke_pass"]=True
      emit(out,"ONE_CHUNK_NATIVE_SMOKE_SHA_PASS",key=key,
           complete_sha256=info["chunk_sha256"])
      print("A22_XPN_ONE_CHUNK_SMOKE_PASS",key,flush=True)
     emit(out,"NATIVE_96_CHUNK_SHA_PASS",key=key,
          complete_sha256=info["chunk_sha256"])
     if len(state["done"])%4==0:
      print("A22_XPN_PROGRESS_CHUNKS",len(state["done"]),"/90",
            "STAGE_GPU_H",round(used(),6),flush=True)
     emit_state(active)
    if STOP or (out/"STOP_XPN_CONTINUATION").exists():
     stopped="STOP_OPERATOR";break
    # No new launch at reserve threshold, but active workers CONTINUE.
    # Once they finish, if reservation still prevents admission, stop
    # with a distinct receipt and unchanged remaining hard-cap budget.
    if not active and not can_admit():
     stopped="STOP_A22_NEW_ADMISSION_RESERVATION";break
    free=[c for c in available(cards)
          if c.index not in {v["gpu"] for v in active.values()}]
    limit=1 if not state["smoke_pass"] else 4
    while len(active)<limit and can_admit():
     chosen=None;gpu=None;info=None
     for partial in (True,False):
      for j in jobs:
       key=job_id(j)
       if key in state["done"] or key in active:continue
       v=check_chunk(EXPECTED_ORIGINAL,j,manifests[j["phase"]],head)
       if (v["status"]=="PARTIAL")!=partial:continue
       if partial:
        card=next((c for c in free if c.index==v["original_gpu"]),None)
        pick=(choose_gpu([card],REQUEST_MIB,safety_margin_mib=RESERVE_MIB,
               fraction=.80,clean_only=False)[0] if card else None)
       else:
        pick,_=choose_gpu(free,REQUEST_MIB,safety_margin_mib=RESERVE_MIB,
                fraction=.80,clean_only=False)
       if pick is not None:
        chosen=j;gpu=pick;info=v;break
      if chosen is not None:break
     if chosen is None:break
     key=job_id(chosen)
     count=state["attempts"].get(key,0)+1
     gate(count<=MAX_RETRIES_PER_JOB,"A22_TOO_MANY_PRESSURE_RETRIES_"+key)
     mf,digest,_=manifests[chosen["phase"]]
     dest=path_for(EXPECTED_ORIGINAL,chosen)
     cmd=[str(FAST),str(checkout/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
       "--expected-head",head,"--roots",str(mf),"--roots-sha",digest,
       "--roots-source-head",
          jread(EXPECTED_TASKS/"XP_N_TASKS_PREP_RECEIPT.json")["expected_head"],
       "--out",str(dest),"--start",str(chosen["start"]),
       "--count",str(chosen["count"]),"--physical-gpu",str(gpu),
       "--dataset","imagenet","--oversample","2.0"]
     if info["status"]=="PARTIAL":cmd.append("--resume")
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
       PYTHONPATH=str(checkout),PYTHONDONTWRITEBYTECODE="1",
       PYTHONNOUSERSITE="1",OMP_NUM_THREADS="2",
       MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
       NUMEXPR_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",
       TMPDIR=str(ROOT/"tmp"))
     log=out/"logs"/f"{key}_try{count:02d}.log"
     handle=log.open("x")
     proc=subprocess.Popen(cmd,cwd=str(checkout),env=env,
       stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
     active[key]=dict(proc=proc,job=chosen,gpu=gpu,start=time.monotonic(),
       handle=handle,log=log,attempt=count,pressure=False,unknown=False,
       foreign=set(),peak=0,samples=0)
     state["attempts"][key]=count
     emit(out,"NEW_NATIVE_WORKER_ADMITTED",key=key,gpu=gpu,
          resume=info["status"]=="PARTIAL",existing_roots=info["reused"],
          native_execution_head=head)
     free=[c for c in free if c.index!=gpu]
     emit_state(active)
    if not active:
     state["status"]="A22_WAIT_FOREIGN_GPU_VRAM"
     emit_state(active)
     time.sleep(POLL_SECONDS)
    else:
     state["status"]="A22_RUNNING_NATIVE"
     emit_state(active)
     time.sleep(SAMPLE_SECONDS)
   status=("A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE"
            if len(state["done"])==90 else stopped or "STOP_A22_UNKNOWN")
  except BaseException as exc:
   status="STOP_A22_NATIVE_OR_LEDGER_ERROR"
   state["failures"].append(type(exc).__name__+":"+str(exc)[:900])
   raise
  finally:
   kill_ours()
   for key,v in list(active.items()):
    v["proc"].wait();v["handle"].close()
    charge(key,v)
    del active[key]
   state["status"]=status
   emit_state(active)
   final=dict(status=status,expected_head=a.expected_head,
     original_execution_head=head,original_run=str(EXPECTED_ORIGINAL),
     a20_terminal_receipt_sha256=sha(a20root/"XP_N_RECOVERY_FINAL_STATUS.json"),
     complete_chunks=len(state["done"]),
     accepted_full_chunk_roots=sum(j["count"] for j in jobs if
                                 job_id(j) in state["done"]),
     initial_stage_gpu_h=prior,A22_charged_gpu_h=state["charged_gpu_h"],
     stage_total_gpu_h=prior+state["charged_gpu_h"],
     project_total_gpu_h=project+state["charged_gpu_h"],
     stage_cap_gpu_h=STAGE_CAP_H,project_cap_gpu_h=PROJECT_CAP_H,
     native_science_seeds_manifests_unchanged=True,
     no_GT_or_scoring_or_predictions_read=True,
     prospective_v5_already_registered=True,
     all_individual_native_root_hashes_verified=True,
     A18_3_complete_case_only_if_incomplete=(len(state["done"])<90),
     failures=state["failures"])
   atomic(out/"XP_N_CONTINUATION_FINAL_STATUS.json",final)
   print("A22_XPN_TERMINAL",json.dumps(dict(
      status=status,done=len(state["done"]),stage_gpu_h=final["stage_total_gpu_h"],
      project_gpu_h=final["project_total_gpu_h"]),sort_keys=True),
      flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A22_XPN_28H_CONTINUATION",str(exc),flush=True)
  sys.exit(8)
