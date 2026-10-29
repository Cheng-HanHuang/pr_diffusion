#!/usr/bin/env python3
"""A13 immediate G1b: A12 FAST ImageNet runs32..63, 100x32 roots.

Independent, fully accounted four-GPU controller. G1a completed with
pixel-exact replay; no additional replay is mandated for G1b. A13
150 GPU-hour cap EXCLUDES A10. Project budget includes A10 and the
historical conservatively billed failed gate. No GT/scoring, no
mutation of original outputs. No foreign process signalling.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
LEDGER=BASE/"XP_A12_PROJECT_LEDGER_VERIFIED.json"
G1A=BASE/"xp_a12_g1a_prod_20261029T180624Z/XP_A12_G1A_FINAL_STATUS.json"
PROJECT_CAP=400.
A12_CAP=150.
STAGE_CAP=40.
V4_SHA="b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3"
STOP=False
def onstop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGINT,onstop)
signal.signal(signal.SIGTERM,onstop)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);temp=p.with_name("."+p.name+".tmp-"+str(os.getpid()))
 with temp.open("x") as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 os.replace(temp,p)
def event(run,msg,**kw):
 with (run/"XP_A13_G1B_EVENTS.jsonl").open("a") as f:
  f.write(json.dumps(dict(event=msg,utc=datetime.datetime.now(
     datetime.timezone.utc).isoformat(),**kw),sort_keys=True)+"\n")
def source_pin(path,head):
 actual=subprocess.check_output(
  ["git","-C",str(path),"rev-parse","HEAD"],text=True,timeout=60).strip()
 dirty=subprocess.check_output(["git","-C",str(path),"status",
             "--porcelain"],text=True,timeout=60).strip()
 gate(actual==head and not dirty,"PINNED_EXECUTOR_CLEAN")
def inputs(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_SUPERVISOR")
 source_pin(wt,a.expected_head)
 # A13 unseal: authenticate v4 as a forecast frozen BEFORE opening
 # any run16+ D4 outcomes. No analysis or GT on the GPU lane.
 registry=subprocess.check_output(["git","-C",str(wt),"show",
     "988aaed:docs/track_r/theory/PREDICTIONS_REGISTRY.md"],timeout=60)
 line=next((x for x in registry.decode().splitlines()
            if x.startswith("| v4 |")),None)
 gate(line is not None and
      "149331c319bb8697c1a0f40c04bf0e4016ec3420" in line and
      V4_SHA in line and "REGISTERED" in line and
      "2026-10-29T20:02:30Z" in line,"A13_V4_REGISTRY_UNSEAL")
 blob=subprocess.check_output(["git","-C",str(wt),"show",
    "149331c319bb8697c1a0f40c04bf0e4016ec3420:docs/track_r/theory/TH1_PREDICTIONS_v4.json"],
    timeout=60)
 gate(hashlib.sha256(blob).hexdigest()==V4_SHA,
      "A13_V4_FROZEN_FORECAST_CONTENT_SHA")
 ledger=read(LEDGER);g1a=read(G1A)
 gate(ledger["status"]=="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED"
      and ledger["has_every_stage_since_XP0"] is True
      and abs(ledger["project_total_before_A12_gpu_h"]-153.581581288)<1e-3
      and ledger["project_reported_total_is_conservative_upper_bound"] is True,
      "HISTORICAL_PROJECT_LEDGER")
 gate(g1a["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
      and g1a["completed"]==1600 and g1a["failed_attempts"]==0
      and g1a["replay_pass"] is True
      and g1a["ledger_sha256"]==sha(LEDGER)
      and abs(g1a["stage_gpu_h"]-3.572931348127313)<1e-5,
      "G1A_COMPLETED_PIXEL_REPLAY")
 project_before=float(g1a["project_total_gpu_h"])
 a12_before=float(g1a["stage_gpu_h"])
 gate(abs(project_before-(ledger["project_total_before_A12_gpu_h"]+
                           a12_before))<1e-6
      and project_before+STAGE_CAP<=PROJECT_CAP
      and a12_before+STAGE_CAP<=A12_CAP,
      "A13_CORRECTED_DOUBLE_CAPS")
 prep=Path(a.prep)
 gate(prep.parent==BASE and prep.name.startswith("xp_a12_g1b_prepare_")
      and prep.is_dir() and not prep.is_symlink(),"PREP_DIR")
 r=read(prep/"XP_A12_G1B_PREP_RECEIPT.json")
 mpath=prep/"XP_A12_G1B_3200_TASKS.json"
 gate(r["status"]=="XP_A13_G1B_IMMEDIATE_PREPARED_CPU_PASS"
      and r["head"]==a.expected_head
      and r["v4_registered_sha256"]==V4_SHA
      and r["n_roots"]==3200
      and r["stage_ceiling_gpu_h"]==40
      and r["G1b_not_yet_admitted"] is True
      and sha(mpath)==r["manifest_sha256"],"A13_G1B_FROZEN_PREP")
 m=read(mpath);rows=m["rows"]
 gate(m["status"]=="XP_FAST_TASKS_FROZEN"
      and m["stage"]=="A12_G1B_FAST_IMAGENET_D4"
      and m["head"]==a.expected_head
      and m["dataset"]=="imagenet"
      and m["gt_paths_supplied"] is False
      and m["n_roots"]==3200 and m["n_images"]==100
      and m["outcome_embargo"] is False
      and len(rows)==3200
      and all([x["rep"] for x in rows[i:i+32]]==
               list(range(32,64)) for i in range(0,3200,32))
      and len({x["uid"] for x in rows})==3200,
      "G1B_100X32_PINNED")
 return wt,prep,r,rows,project_before,a12_before
def verify(a,job,dest,receipt,rows):
 m=read(dest/"R0_5_CHUNK_META.json")
 d=read(dest/"R0_5_CHUNK_COMPLETE.json")
 rootsha=receipt["manifest_sha256"]
 start=job["start"];count=job["count"]
 gate(m["head"]==a.expected_head
      and m["root_sha256"]==rootsha
      and m["start"]==start and m["count"]==count
      and m["uids"]==[x["uid"] for x in rows[start:start+count]]
      and d["status"]=="R0_5_CHUNK_PASS"
      and d["head"]==a.expected_head
      and d["start"]==start and d["count"]==count
      and d["new"]==count
      and d["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
      "FAST_NATIVE_CHUNK")
 for x in rows[start:start+count]:
  r=validate_complete(root_folder(dest,x),x,a.expected_head,rootsha)
  gate(r["gt_access"] is False and
       r["method"]=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5",
       "NATIVE_FAST_METHOD_AND_GTFREE")
 return sha(dest/"R0_5_CHUNK_COMPLETE.json")
def main(a):
 wt,prep,receipt,rows,project_before,a12_before=inputs(a)
 if a.preflight_only:
  print("XP_A13_G1B_SOURCE_BUDGET_PRELIGHT_PASS",flush=True)
  print("G1B_PREP_SHA256",receipt["manifest_sha256"],flush=True)
  print("G1B_PRIOR_PROJECT_GPU_H",project_before,flush=True)
  print("G1B_PRIOR_A12_GPU_H",a12_before,flush=True)
  print("G1B_3200_NOROOT_INFERENCE=true",flush=True)
  return
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a13_g1b_prod_")
      and not run.exists() and not run.is_symlink(),"FRESH_RUN")
 lock=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
 with lock.open("a+") as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for sub in ("logs","cost","production","pycache"):(run/sub).mkdir()
  state=dict(status="A13_G1B_RUNNING",expected_head=a.expected_head,
     input_manifest_sha256=receipt["manifest_sha256"],
     project_ledger_sha256=sha(LEDGER),
     g1a_final_sha256=sha(G1A),
     project_before_gpu_h=project_before,a12_before_gpu_h=a12_before,
     v4_registered_sha256=V4_SHA,stage_ceiling_gpu_h=STAGE_CAP,
     A12_cap_gpu_h=A12_CAP,project_cap_gpu_h=PROJECT_CAP,
     spent_gpu_seconds=0.,done={},active=[],failures=[],GT_not_read=True)
  jobs=[dict(index=k,start=k*32,count=32) for k in range(100)]
  active={}
  def store(status):
   state["status"]=status
   state["active"]=[dict(index=i,gpu=v["gpu"],pid=v["proc"].pid,
                    attempt=v["attempt"]) for i,v in active.items()]
   write(run/"XP_A13_G1B_CONTROLLER_STATE.json",state)
  def charged_active():
   return sum(time.monotonic()-v["start"] for v in active.values())
  def total_charge():
   return (state["spent_gpu_seconds"]+charged_active())/3600
  def charge(i,v,rc):
   elapsed=time.monotonic()-v["start"]
   cost=run/"cost"/f"XP_A13_G1B_{i:03d}_A{v['attempt']}.json"
   write(cost,dict(status="A13_G1B_GPU_ATTEMPT_CHARGED",
       index=i,start=jobs[i]["start"],count=32,attempt=v["attempt"],
       gpu=v["gpu"],pid=v["proc"].pid,elapsed_seconds=elapsed,
       charged_gpu_h=elapsed/3600,returncode=rc,
       logfile=str(v["log"])))
   state["spent_gpu_seconds"]+=elapsed
  def shutdown():
   for v in active.values():
    if v["proc"].poll() is None:
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
  status="RUNNING"
  store(status)
  try:
   while len(state["done"])<100:
    cards=collect(own_roots=[v["proc"].pid for v in active.values()])
    for i,v in list(active.items()):
     card=next((c for c in cards if c.index==v["gpu"]),None)
     if card is not None and v["proc"].poll() is None:
      if card.used_mib>0.8*card.total_mib:
       event(run,"VRAM_80PCT_STOP_OWN_WORKER",index=i,gpu=v["gpu"])
       try:os.killpg(v["proc"].pid,signal.SIGTERM)
       except ProcessLookupError:pass
     rc=v["proc"].poll()
     if rc is None:continue
     v["handle"].close();charge(i,v,rc);del active[i]
     if rc!=0:
      state["failures"].append(dict(index=i,rc=rc,log=str(v["log"])))
      raise RuntimeError("STOP_A13_G1B_NATIVE_FAILED_CHUNK_"+str(i))
     chunk=verify(a,jobs[i],v["dest"],receipt,rows)
     state["done"][str(i)]=chunk
     if len(state["done"])%10==0:
      print("A13_G1B_CHUNKS",len(state["done"]),"/100",
           "GPU_H",round(state["spent_gpu_seconds"]/3600,4),flush=True)
     store("RUNNING")
    if len(state["done"])==100:break
    gross=total_charge()
    deadline=datetime.datetime(2026,11,4,12,0,
            tzinfo=ZoneInfo("America/Detroit"))
    reason=("OPERATOR_STOP" if STOP or (run/"STOP_G1B").exists()
         else "G1B_40H_CEILING" if gross>=STAGE_CAP-0.01
         else "A12_150H_CEILING" if a12_before+gross>=A12_CAP-0.01
         else "PROJECT_400H_CEILING" if project_before+gross>=PROJECT_CAP-0.01
         else "DEADLINE" if datetime.datetime.now(
                   ZoneInfo("America/Detroit"))>=deadline
         else None)
    if reason:
     status="PARTIAL_"+reason
     shutdown();break
    cards_free=[c for c in cards if c.index not in
         {v["gpu"] for v in active.values()} and c.visibility=="known"]
    while len(active)<4:
     # Charge a conservative reservation for ONE additional worker,
     # including currently active process intervals; no per-GPU concurrency
     # beyond one unless a separately validated native W* exists.
     reserve=(len(active)+1)*900/3600
     gross=total_charge()
     if (gross+reserve>STAGE_CAP or
         a12_before+gross+reserve>A12_CAP or
         project_before+gross+reserve>PROJECT_CAP):
      if not active:status="PARTIAL_RESERVATION_CAP"
      break
     gpu,_=choose_gpu(cards_free,8192,safety_margin_mib=2048,
         fraction=.80,clean_only=False)
     if gpu is None:break
     i=next((j["index"] for j in jobs if str(j["index"]) not in state["done"]
             and j["index"] not in active),None)
     if i is None:break
     dest=run/"production"/f"chunk_{i*32:05d}_032"
     gate(not dest.exists(),"NEW_OUTPUT_COLLISION")
     cmd=[str(FAST),str(wt/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
       "--expected-head",a.expected_head,
       "--roots",str(prep/"XP_A12_G1B_3200_TASKS.json"),
       "--roots-sha",receipt["manifest_sha256"],
       "--roots-source-head",receipt["head"],
       "--out",str(dest),"--start",str(i*32),"--count","32",
       "--physical-gpu",str(gpu),"--dataset","imagenet","--oversample","2.0"]
     env=dict(os.environ)
     env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(wt),
        PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        PYTHONPYCACHEPREFIX=str(run/"pycache"/f"g1b_{i:03d}"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",TMPDIR=str(ROOT/"tmp"),
        XDG_CACHE_HOME=str(ROOT/"outputs/.cache/track_r"),
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",
        B20_LF_ENABLE="0",B21_CONT_ENABLE="0")
     logfile=run/"logs"/f"G1B_{i:03d}.log"
     h=logfile.open("x")
     proc=subprocess.Popen(cmd,cwd=str(wt),env=env,stdout=h,
         stderr=subprocess.STDOUT,start_new_session=True)
     active[i]=dict(proc=proc,gpu=gpu,start=time.monotonic(),
                     attempt=0,handle=h,log=logfile,dest=dest)
     event(run,"CHUNK_LAUNCHED",index=i,start=i*32,gpu=gpu,pid=proc.pid)
     cards_free=[c for c in cards_free if c.index!=gpu]
     store("RUNNING")
    if status=="PARTIAL_RESERVATION_CAP":break
    time.sleep(4)
   if len(state["done"])==100:status="XP_A13_G1B_3200_GENERATION_COMPLETE"
  except BaseException:
   status="STOP_A13_G1B_EXCEPTION"
   raise
  finally:
   shutdown()
   for i,v in list(active.items()):
    v["proc"].wait();v["handle"].close()
    charge(i,v,v["proc"].returncode)
    del active[i]
   store(status)
   final=dict(status=status,completed=32*len(state["done"]),
     total=3200,failed_attempts=len(state["failures"]),
     stage_gpu_h=state["spent_gpu_seconds"]/3600,
     A12_total_gpu_h=a12_before+state["spent_gpu_seconds"]/3600,
     project_total_gpu_h=project_before+state["spent_gpu_seconds"]/3600,
     stage_cap_gpu_h=STAGE_CAP,A12_cap_gpu_h=A12_CAP,
     project_cap_gpu_h=PROJECT_CAP,
     source_head=a.expected_head,manifest_sha256=receipt["manifest_sha256"],
     G1a_pixel_exact_replay=True,v4_registered_sha256=V4_SHA,
     no_GT_scoring=True)
   write(run/"XP_A13_G1B_FINAL_STATUS.json",final)
   print("XP_A13_G1B_FINAL",json.dumps(final,sort_keys=True),flush=True)
def entry():
 p=argparse.ArgumentParser()
 for key in ("expected-head","prep","run"):
  p.add_argument("--"+key,required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: XP_A13_G1B_CONTROLLER",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
