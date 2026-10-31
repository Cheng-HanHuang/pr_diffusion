#!/usr/bin/env python3
"""A16.5 XP-N 100x64 native FAST-DIPS GPU generation autopilot.

Mandatory prospective guard: no new XP-N root may start before a
SHA-verifiable v5 CODE freeze recorded in the planner registry, OR
Nov 2 2026 12:00 ET (optional forecast lapses). No v5 forecast
is claimed merely from a code freeze. Runs 4..63 stay LOCAL, UNSCORED
and SEALED until registered/declined v5 OUTPUT or Nov 4 noon ET.
No GT, scoring, previews or predictions are opened by this controller.

Phases 0..3, 4..15, 16..63, in that order. Up to four GPUs under
A10.1 co-tenancy; 80% admission cap, 1 Track R process/GPU, no
foreign signaling. Every process-wall attempt is charged; stage
and 400 GPU-hour project caps. Crash with live workers fails closed.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a18_xpn_contract import PHASES
from scripts.track_r.xp_a18_xpn_contract import validate_xpn_source,Y_SHA,AUDIT_SHA,MODEL_SHA
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
PLANNER_BRANCH="claude/prdiff-track-r-planner-20261021"
REGISTRY="docs/track_r/theory/PREDICTIONS_REGISTRY.md"
V5_CODE_CUTOFF=datetime.datetime(2026,11,2,12,0,tzinfo=ZoneInfo("America/Detroit"))
V5_OUTCOME_UNSEAL=datetime.datetime(2026,11,4,12,0,tzinfo=ZoneInfo("America/Detroit"))
PHASE_FILES={
 "A18_XPN_PILOT_0_3":"XP_N_PILOT_0_3_TASKS.json",
 "A18_XPN_MAIN_4_15":"XP_N_MAIN_4_15_TASKS.json",
 "A18_XPN_LATE_16_63":"XP_N_LATE_16_63_TASKS.json",
}
PHASE_BATCH={"A18_XPN_PILOT_0_3":20,
             "A18_XPN_MAIN_4_15":60,
             "A18_XPN_LATE_16_63":96}
REQUEST_MIB=8192
RESERVE_MIB=2048
STAGE_CAP_H=20.
PROJECT_CAP_H=400.
RESERVATION_PER_NEW_WORKER_H=0.25
BASE_STATE=BASE/"xp_a12_campaign_prod_20261030T022725Z/XP_A12_CAMPAIGN_STATE.json"
A17_RECEIPT=BASE/"xp_a17_cp3_timing_20261031T165802Z/A17_CP3_CLEAN_TIMING_RECEIPT.json"
REPLAY_STATUS="A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS"
STOP=False

def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def localnow():return datetime.datetime.now(ZoneInfo("America/Detroit"))
def emit(run,kind,**kw):
 with (run/"XP_N_EVENTS.jsonl").open("a",encoding="utf-8") as f:
  f.write(json.dumps(dict(utc=now(),kind=kind,**kw),sort_keys=True)+"\n")
def atomic(path,obj):
 path=Path(path)
 tmp=path.with_name("."+path.name+".tmp-"+str(os.getpid()))
 with tmp.open("x") as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 os.replace(tmp,path)
def git(wt,*a):
 return subprocess.check_output(["git","-C",str(wt),*a],
     text=True,timeout=90).strip()
def prep(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_AUTOPILOT_CPU_ONLY")
 gate(git(wt,"rev-parse","HEAD")==a.expected_head and
      not git(wt,"status","--porcelain"),"A18_AUTOPILOT_CLEAN_PINNED_HEAD")
 tasks=Path(a.tasks)
 gate(tasks.parent==BASE and tasks.name.startswith("xp_a18_xpn_tasks_")
      and tasks.is_dir() and not tasks.is_symlink(),
      "A18_AUTOPILOT_IMMUTABLE_TASKS_DIR")
 r=jread(tasks/"XP_N_TASKS_PREP_RECEIPT.json")
 producer=r["expected_head"]
 gate(r["status"]=="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
      and r["cheng_go"] is True
      and r["native_d1_replay_pixel_exact"] is True
      and r["n_roots"]==6400 and r["n_images"]==100
      and r["no_GT_or_candidate_outcomes_read"] is True
      and r["source_y_manifest_sha256"]==Y_SHA
      and r["source_gt_y_audit_sha256"]==AUDIT_SHA
      and r["stage_GPU_cap_h"]==STAGE_CAP_H
      and r["project_cap_GPU_h"]==PROJECT_CAP_H
      and r["gpu_production_not_started"] is True,
      "A18_AUTOPILOT_FROZEN_TASK_PREP_RECEIPT")
 manifests={};jobs=[]
 for phase in PHASES:
  path=tasks/PHASE_FILES[phase]
  manifest=jread(path)
  digest=sha(path)
  rows=manifest["rows"]
  size=PHASE_BATCH[phase]
  gate(r["phase_files"][phase]["sha256"]==digest,
       "A18_AUTOPILOT_MANIFEST_SHA_"+phase)
  for start in range(0,len(rows),size):
   count=min(size,len(rows)-start)
   validate_xpn_source(path,digest,manifest,rows,producer,start,count)
   jobs.append(dict(phase=phase,start=start,count=count,
        key=f"{phase}_{start:05d}"))
  manifests[phase]=(path,digest,rows)
 gate(len(jobs)==90 and sum(x["count"] for x in jobs)==6400,
      "A18_AUTOPILOT_EXACT_PHASE_PARTITION")
 replay_path=Path(a.replay)
 gate(replay_path.parent==BASE and replay_path.name.startswith("xp_a18_xpn_replay_"),
      "A18_AUTOPILOT_REPLAY_ROOT")
 replay=jread(replay_path/"XP_N_D1_REPLAY_RECEIPT.json")
 gate(replay["status"]==REPLAY_STATUS
      and replay["expected_head"]=="97b8265685307115af2f8f5d39bf8336b38429d1"
      and replay["XP_N_independent_GT_Y_audit_sha256"]==AUDIT_SHA
      and replay["native_worker_rc"]==0
      and replay["visibility_unknown"] is False
      and replay["n_roots"]==2
      and all(z["pixels_exact"] for z in replay["pixel_comparisons"])
      and r["replay_receipt_sha256"]==sha(replay_path/"XP_N_D1_REPLAY_RECEIPT.json"),
      "A18_AUTOPILOT_REPLAY_SHA_AND_PIXELS")
 gate(FAST.is_file(),"A18_AUTOPILOT_NATIVE_FAST_ENV")
 gate(sha(ROOT/"models/imagenet256.pt")==MODEL_SHA,
      "A18_AUTOPILOT_NATIVE_CHECKPOINT_SHA")
 # Exercise the REAL native source/config/measurement integrity gate for
 # every phase before scheduling GPUs. --preflight-only does not load CUDA
 # or produce a candidate, and cannot invalidate a v5 forecast.
 for phase,(mf,digest,rows) in manifests.items():
  e=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONPATH=str(wt),
         PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
  cmd=[str(FAST),str(wt/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
    "--expected-head",a.expected_head,
    "--roots",str(mf),"--roots-sha",digest,
    "--roots-source-head",producer,
    "--out",str(BASE/"xp_a18_xpn_uncreated_native_preflight"),
    "--start","0","--count",str(PHASE_BATCH[phase]),
    "--physical-gpu","0","--dataset","imagenet","--oversample","2.0",
    "--preflight-only"]
  native=subprocess.run(cmd,cwd=str(wt),env=e,
      capture_output=True,text=True,timeout=180)
  gate(native.returncode==0 and "XP_FAST_PREFLIGHT_PASS" in native.stdout,
      "A18_AUTOPILOT_NATIVE_PHASE_PREFLIGHT_"+phase+"_"+
      (native.stderr[-260:]+native.stdout[-260:]))
 b=jread(BASE_STATE);t=jread(A17_RECEIPT)
 gate(b["status"]=="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
      and not b.get("active")
      and t["status"]=="A17_CP3_CLEAN_PAIRED_TIMING_PASS",
      "A18_AUTOPILOT_ACCEPTED_PROJECT_BASELINE")
 baseline=float(b["project_charged_gpu_h"])+float(t["total_A17_charged_gpu_hours"])
 charge_replay=float(replay["charged_gpu_h"])
 gate(0<charge_replay<1.
      and baseline+charge_replay+STAGE_CAP_H<=PROJECT_CAP_H,
      "A18_AUTOPILOT_PROJECT_CAP")
 # No parallel XP-N controller. Other live GPU jobs are checked at each
 # admission and cannot share a GPU with this one.
 for p in BASE.glob("xp_a18_xpn_prod_*/XP_N_FINAL_STATUS.json"):
  result=jread(p)
  gate(result.get("status")!="A18_XPN_ALL6400_NATIVE_GENERATION_COMPLETE_SEALED",
       "A18_AUTOPILOT_ALREADY_PRODUCED")
 return wt,tasks,producer,manifests,jobs,baseline,charge_replay

def v5_code_freeze(wt):
 """Return verified first-code-freeze evidence, or None.

 Parser accepts a planner registry row with freeze name exactly
 'v5 code' (not a v5 predictions row). SHA-256 of code artifact
 must be independently verified from the pinned git commit.
 """
 try:
  subprocess.run(["git","-C",str(wt),"fetch","--no-tags","--quiet",
    "origin","refs/heads/"+PLANNER_BRANCH],check=True,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
  text=git(wt,"show","FETCH_HEAD:"+REGISTRY)
  # The planner registry may point to a theory-branch commit whose
  # objects are not yet reachable in this detached execution worktree.
  # Fetch its branch without touching source code or working files.
  subprocess.run(["git","-C",str(wt),"fetch","--no-tags","--quiet",
    "origin","refs/heads/theory/prdiff-track-r-th-20261026"],
    check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    timeout=60)
 except (OSError,subprocess.SubprocessError):
  return None
 lines=[line for line in text.splitlines()
        if line.startswith("|") and
           len(line.split("|"))>=8 and
           line.split("|")[1].strip().lower()=="v5 code"]
 if len(lines)!=1:return None
 f=[x.strip() for x in lines[0].split("|")[1:-1]]
 if len(f)<8 or not ("FROZEN" in lines[0] or "REGISTERED" in lines[0]):
  return None
 commit=f[2].strip(" `")
 path=f[3].strip(" `")
 digestmatch=re.search(r"\b[a-f0-9]{64}\b",f[4])
 if not re.fullmatch(r"[a-f0-9]{40}",commit) or not path.startswith(
    "docs/track_r/theory/") or digestmatch is None:return None
 try:
  data=subprocess.check_output(["git","-C",str(wt),"show",
         commit+":"+path],timeout=90)
  stamp=git(wt,"show","-s","--format=%cI",commit)
  t=datetime.datetime.fromisoformat(stamp)
 except (ValueError,OSError,subprocess.SubprocessError):
  return None
 if hashlib.sha256(data).hexdigest()!=digestmatch.group(0) or t>V5_CODE_CUTOFF:
  return None
 return dict(commit=commit,path=path,sha256=digestmatch.group(0),
             commit_time=stamp,registry_line=lines[0])

def v5_gate(wt):
 evidence=v5_code_freeze(wt)
 if evidence:return dict(status="V5_CODE_SHA_FROZEN_PRE_FIRST_XPN_ROOT",
                         evidence=evidence)
 if localnow()>=V5_CODE_CUTOFF:
  return dict(status="V5_FORECAST_CODE_LAPSED_NO_OPTIONAL_V5_TEST",
              cutoff=V5_CODE_CUTOFF.isoformat())
 return None

def available(cards):
 result=[]
 for c in cards:
  if c.visibility!="known":continue
  track=False
  for pid in c.process_pids:
   p=subprocess.run(["ps","-o","args=","-p",str(pid)],
      capture_output=True,text=True,timeout=8)
   if p.returncode==0 and (
       "scripts/track_r/" in p.stdout
       or "scripts.track_r." in p.stdout
       or "pr_diffusion_track_r" in p.stdout):
    track=True
    break
  if not track:result.append(c)
 return result

def verify_chunk(a,job,run,manifests):
 phase,start,count=job["phase"],job["start"],job["count"]
 path,digest,rows=manifests[phase]
 dest=run/"production"/phase/f"chunk_{start:05d}_{count:03d}"
 m=jread(dest/"R0_5_CHUNK_META.json")
 c=jread(dest/"R0_5_CHUNK_COMPLETE.json")
 gate(m["status"]=="R0_5_CHUNK_CONFIG" and
      m["head"]==a.expected_head
      and m["root_sha256"]==digest
      and m["start"]==start and m["count"]==count
      and m["uids"]==[r["uid"] for r in rows[start:start+count]]
      and c["status"]=="R0_5_CHUNK_PASS"
      and c["new"]==count and c["start"]==start
      and c["count"]==count and c["head"]==a.expected_head
      and c["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
      "A18_AUTOPILOT_NATIVE_CHUNK_RECEIPTS")
 for row in rows[start:start+count]:
  x=validate_complete(root_folder(dest,row),row,a.expected_head,digest)
  gate(x["gt_access"] is False
       and x["method"]=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
       and x["measurement_sha256"]==row["measurement_sha256"]
       and x["fast_source_head"]==
         "a9758b38906b7c7950fa04491f37f6634948db26",
       "A18_AUTOPILOT_NATIVE_GT_FREE_RECEIPT")
 return sha(dest/"R0_5_CHUNK_COMPLETE.json")

def main(a):
 wt,tasks,producer,manifests,jobs,baseline,replaycharge=prep(a)
 if a.preflight_only:
  gate((Path(a.run).parent==BASE and
        Path(a.run).name.startswith("xp_a18_xpn_prod_")),
       "A18_AUTOPILOT_RUN_NAMESPACE")
  g=v5_gate(wt)
  print("A18_XPN_AUTOPILOT_PREFLIGHT_PASS",
        "JOBS",len(jobs),"ROOTS",6400,
        "V5_GATE",g["status"] if g else "WAIT_V5_CODE_OR_NOV2_NOON",
        "BUDGET_BASE",round(baseline+replaycharge,6),flush=True)
  return
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a18_xpn_prod_")
      and not run.exists() and not run.is_symlink(),
      "A18_AUTOPILOT_NEW_RUN_ROOT")
 # Stage lock prevents competing existing A12 controllers from double
 # allocating the project ledger. Do not lock while waiting for v5.
 run.mkdir(mode=0o700)
 for sub in ("production","logs","attempt_cost","pycache"):
  (run/sub).mkdir(mode=0o700)
 state=dict(status="WAIT_V5_CODE_FREEZE_OR_NOV2_CUTOFF",
     expected_head=a.expected_head,manifest_producer_head=producer,
     tasks_preparation_receipt_sha256=sha(tasks/"XP_N_TASKS_PREP_RECEIPT.json"),
     GT_Y_independent_sha256=AUDIT_SHA,
     two_root_replay_receipt_sha256=sha(Path(a.replay)/"XP_N_D1_REPLAY_RECEIPT.json"),
     project_baseline_gpu_h=baseline,replay_gpu_h=replaycharge,
     stage_cap_gpu_h=STAGE_CAP_H,project_cap_gpu_h=PROJECT_CAP_H,
     phase_manifests_sha256={p:x[1] for p,x in manifests.items()},
     done={},attempt_cost_gpu_h=0.,active=[],failures=[],
     phase_counts={},v5_gate=None,
     XP_N_4_to_63_outcomes_sealed=True,GT_not_read=True,
     no_GT_scoring_or_prediction_access=True)
 def persist():
  atomic(run/"XP_N_CONTROLLER_STATE.json",state)
 persist()
 wait_cycles=0
 while True:
  if STOP or (run/"STOP_XPN").exists():
   state["status"]="STOP_BEFORE_GPU_V5_GATE"
   persist()
   print("A18_XPN_AUTOPILOT_STOP_BEFORE_GPU",flush=True)
   return
  evidence=v5_gate(wt)
  if evidence:
   state["v5_gate"]=evidence
   break
  wait_cycles+=1
  if wait_cycles%12==1:
   print("A18_XPN_WAITING_V5_CODE_FREEZE_OR_2026_11_02_12_ET",
         "PROJECT_GPU_H_USED",0,flush=True)
  time.sleep(300)
 # Commit the gate evidence BEFORE allocating any new XP-N GPU worker.
 state["status"]="V5_PROSPECTIVE_GATE_PASS"
 persist()
 emit(run,"V5_PROSPECTIVE_GATE_PASS",v5_gate=state["v5_gate"])
 lock=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
 with lock.open("a+") as h:
  # An existing A12/Tail controller may own the lock when v5 becomes
  # available. Wait without spawning anything; never preempt the other
  # controller or silently bypass the shared project budget.
  while True:
   try:
    fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    break
   except BlockingIOError:
    if STOP or (run/"STOP_XPN").exists():
     state["status"]="STOP_WHILE_WAITING_PROJECT_GPU_LOCK"
     persist()
     print("A18_XPN_OPERATOR_STOP_WHILE_WAITING_LOCK",flush=True)
     return
    time.sleep(30)
  active={}
  def charge(jobid,proc,rc):
   v=active[jobid]
   elapsed=(time.monotonic()-v["start"])/3600
   name=f"XP_N_GPU_ATTEMPT_{jobid}.json"
   receipt=dict(status="A18_XPN_NATIVE_GPU_ATTEMPT_CHARGED",
       job_id=jobid,phase=v["job"]["phase"],start=v["job"]["start"],
       count=v["job"]["count"],gpu=v["gpu"],pid=proc.pid,
       rc=rc,elapsed_process_gpu_h=elapsed,
       foreign_pids_seen=sorted(v["foreign"]),
       process_visibility_unknown=v["unknown"],
       peak_used_mib=v["peak_used"],n_samples=v["samples"],
       logfile=str(v["log"]),utc=now())
   atomic(run/"attempt_cost"/name,receipt)
   state["attempt_cost_gpu_h"]+=elapsed
   return elapsed
  def write_state(status):
   state["status"]=status
   state["active"]=[dict(job_id=k,gpu=v["gpu"],pid=v["proc"].pid)
                    for k,v in active.items()]
   persist()
  def stop_own():
   for v in active.values():
    if v["proc"].poll() is None:
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
  def live_charge():
   return sum((time.monotonic()-v["start"])/3600 for v in active.values())
  def budget_allows(n_extra):
   proposed=state["attempt_cost_gpu_h"]+live_charge()+(
       n_extra+len(active))*RESERVATION_PER_NEW_WORKER_H
   return (proposed<=STAGE_CAP_H
       and baseline+replaycharge+proposed<=PROJECT_CAP_H)
  write_state("A18_XPN_RUNNING")
  status="RUNNING";failed=False
  try:
   for phase in PHASES:
    these=[j for j in jobs if j["phase"]==phase]
    while not all(z["key"] in state["done"] for z in these):
     cards=collect(own_roots=[v["proc"].pid for v in active.values()])
     for jobid,v in list(active.items()):
      card=next((c for c in cards if c.index==v["gpu"]),None)
      if card is None or card.visibility!="known":
       v["unknown"]=True
      else:
       v["foreign"].update(card.foreign_pids)
       v["peak_used"]=max(v["peak_used"],card.used_mib)
       v["samples"]+=1
       if card.used_mib>0.80*card.total_mib:
        emit(run,"ABOVE_80PCT_TERMINATE_OWN_WORKER",job=jobid,gpu=v["gpu"])
        try:os.killpg(v["proc"].pid,signal.SIGTERM)
        except ProcessLookupError:pass
      rc=v["proc"].poll()
      if rc is None:continue
      v["handle"].close()
      charge(jobid,v["proc"],rc)
      del active[jobid]
      if rc!=0:
       state["failures"].append(dict(job=jobid,rc=rc,log=str(v["log"])))
       failed=True
       break
      try:
       receiptsha=verify_chunk(a,v["job"],run,manifests)
      except Exception as e:
       state["failures"].append(dict(job=jobid,reason=str(e)[:400]))
       failed=True
       break
      state["done"][jobid]=receiptsha
      state["phase_counts"][phase]=sum(
       z["phase"]==phase and z["key"] in state["done"] for z in these)
      if len(state["done"])%10==0:
       print("A18_XPN_GPU_CHUNKS",len(state["done"]),"/90",
        "CHARGED_GPU_H",round(state["attempt_cost_gpu_h"],5),flush=True)
      write_state("A18_XPN_RUNNING")
     if failed:raise RuntimeError("A18_XPN_NATIVE_WORKER_OR_RECEIPT_FAILED")
     if all(z["key"] in state["done"] for z in these):break
     if STOP or (run/"STOP_XPN").exists():
      status="STOP_OPERATOR"
      break
     if not budget_allows(1):
      status="STOP_XPN_OR_PROJECT_BUDGET_RESERVATION"
      break
     unused=[c for c in available(cards) if c.index not in
             {v["gpu"] for v in active.values()}]
     while len(active)<4 and budget_allows(1):
      gpu,_=choose_gpu(unused,REQUEST_MIB,safety_margin_mib=RESERVE_MIB,
                      fraction=.80,clean_only=False)
      if gpu is None:break
      j=next((z for z in these
              if z["key"] not in state["done"] and z["key"] not in active),
             None)
      if j is None:break
      jobid=j["key"]
      dest=(run/"production"/phase/
            f"chunk_{j['start']:05d}_{j['count']:03d}")
      gate(not dest.exists(),"A18_AUTOPILOT_CHUNK_REUSE_OR_ORPHAN")
      mpath,digest,_=manifests[phase]
      cmd=[str(FAST),str(wt/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
       "--expected-head",a.expected_head,
       "--roots",str(mpath),"--roots-sha",digest,
       "--roots-source-head",producer,"--out",str(dest),
       "--start",str(j["start"]),"--count",str(j["count"]),
       "--physical-gpu",str(gpu),
       "--dataset","imagenet","--oversample","2.0"]
      env=dict(os.environ)
      env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(wt),
       PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
       PYTHONPYCACHEPREFIX=str(run/"pycache"/jobid),
       CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",
       MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
       NUMEXPR_NUM_THREADS="2",B20_LF_ENABLE="0",B21_CONT_ENABLE="0",
       TMPDIR=str(ROOT/"tmp"))
      log=run/"logs"/f"{jobid}.log"
      handle=log.open("x")
      proc=subprocess.Popen(cmd,cwd=str(wt),env=env,stdout=handle,
          stderr=subprocess.STDOUT,start_new_session=True)
      active[jobid]=dict(proc=proc,gpu=gpu,start=time.monotonic(),
         job=j,handle=handle,log=log,foreign=set(),
         unknown=False,peak_used=0,samples=0)
      emit(run,"XP_N_NATIVE_CHUNK_LAUNCHED",job=jobid,gpu=gpu,
           pid=proc.pid,manifest_sha=digest)
      unused=[c for c in unused if c.index!=gpu]
      write_state("A18_XPN_RUNNING")
     time.sleep(5)
    if status!="RUNNING":break
    # No GPU job of the next phase starts until the entire earlier phase
    # has completed, preserving the preregistered sampling order.
    emit(run,"XP_N_PHASE_GENERATED_UNSCORED_SEALED",phase=phase,
         n_roots=sum(j["count"] for j in these))
    print("A18_XPN_PHASE_GPU_GENERATION_SEALED",phase,
          sum(j["count"] for j in these),flush=True)
   if len(state["done"])==len(jobs):
    status="A18_XPN_ALL6400_NATIVE_GENERATION_COMPLETE_SEALED"
  except BaseException as e:
   status="STOP_XPN_CONTROLLER_OR_NATIVE_EXCEPTION"
   state["failures"].append(dict(reason=type(e).__name__+":"+str(e)[:800]))
   raise
  finally:
   stop_own()
   for key,v in list(active.items()):
    v["proc"].wait()
    v["handle"].close()
    charge(key,v["proc"],v["proc"].returncode)
    del active[key]
   write_state(status)
   summary=dict(status=status,expected_head=a.expected_head,
      GT_Y_independent_sha256=AUDIT_SHA,
      source_tasks_receipt_sha256=state["tasks_preparation_receipt_sha256"],
      prospectivity_v5_gate=state["v5_gate"],
      total_roots=6400,n_completed_roots=sum(
        j["count"] for j in jobs if j["key"] in state["done"]),
      n_failed=len(state["failures"]),
      charged_stage_gpu_h=state["attempt_cost_gpu_h"],
      charged_historical_D1_replay_gpu_h=replaycharge,
      project_total_gpu_h=baseline+replaycharge+state["attempt_cost_gpu_h"],
      max_stage_gpu_h=STAGE_CAP_H,max_project_gpu_h=PROJECT_CAP_H,
      outcomes_4_to_63_still_sealed=True,
      all_candidate_outputs_local_not_published=True,
      no_GT_or_scoring_or_oracle_reads=True,
      native_fast_unchanged=True,
      independent_phases=list(PHASES),
      manifest_sha256={p:v[1] for p,v in manifests.items()},
      failures=state["failures"])
   atomic(run/"XP_N_FINAL_STATUS.json",summary)
   print("A18_XPN_AUTOPILOT_TERMINAL",
         json.dumps(dict(status=status,
           completed=summary["n_completed_roots"],
           charged_gpu_h=summary["charged_stage_gpu_h"],
           project_total_gpu_h=summary["project_total_gpu_h"]),
         sort_keys=True),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--tasks",required=True)
 ap.add_argument("--replay",required=True)
 ap.add_argument("--run",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_AUTOPILOT",e,flush=True)
  sys.exit(8)
