#!/usr/bin/env python3
"""A12 automatic stage queue: gated G2, conditional G3, G4 and G5.

Single-controller shared 400 project / 150 A12 campaign GPU budget.
New generation has no GT or scoring. Never reads D4 run >=16 outcomes.
G3 requires a separately certified native PC-DAPS signed-y adapter;
when absent/skipped it publishes a blocked receipt. G1b is deliberately
not admitted unless G1a-G5 launched and remaining projected budget fits.
G6/G7 have their own readiness gates and are NOT invented here.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image,ImageChops
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import (
  sha as sha_fast,root_folder,validate_complete)
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
ROOT_FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
ROOT_DAPS_PY=ROOT/"conda-envs/daps/bin/python"
G1A=BASE/"xp_a12_g1a_prod_20261029T180624Z/XP_A12_G1A_FINAL_STATUS.json"
DEADLINE={"G2":(2026,11,3,12),"G4":(2026,11,4,12),"G5":(2026,11,4,12)}
CAP={"G2":10.,"G4":30.,"G5":40.}
N={"G2":2880,"G4":720,"G5":400}
RESERVE_MIB={"G2":8192,"G4":12288,"G5":12288}
A10=55.54578541457511
CAMPAIGN_CAP=150.
PROJECT_CAP=400.
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jread(p):
 return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+".tmp"+str(os.getpid()))
 with tmp.open("x") as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 os.replace(tmp,p)
def emit(run,tag,**fields):
 with (run/"XP_A12_CAMPAIGN_EVENTS.jsonl").open("a") as f:
  f.write(json.dumps(dict(event=tag,utc=datetime.datetime.now(
    datetime.timezone.utc).isoformat(),**fields),sort_keys=True)+"\n")
def head(wt):
 return subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],
                                text=True).strip()
def prepar(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_CONTROLLER")
 gate(head(wt)==a.expected_head and not subprocess.check_output(
  ["git","-C",str(wt),"status","--porcelain"],text=True).strip(),
  "SOURCE_PIN")
 ledger=Path(a.ledger)
 gate(ledger==BASE/"XP_A12_PROJECT_LEDGER_VERIFIED.json","LEDGER_PATH")
 L=jread(ledger)
 gate(L["status"]=="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED"
      and L["project_reported_total_is_conservative_upper_bound"] is True
      and L["has_every_stage_since_XP0"] is True
      and abs(L["project_total_before_A12_gpu_h"]-
       sum(z["gpu_h"] for z in L["stages"]))<1e-6,"PROJECT_LEDGER")
 g=jread(G1A)
 gate(g["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
      and g["completed"]==1600 and g["replay_pass"] is True
      and g["failed_attempts"]==0 and g["outcome_sealed"] is True
      and g["ledger_sha256"]==sha(ledger),"G1A_RECEIPT")
 start=L["project_total_before_A12_gpu_h"]+g["stage_gpu_h"]
 campaign=A10+g["stage_gpu_h"]
 gate(start+CAMPAIGN_CAP-campaign<=PROJECT_CAP and
      campaign<CAMPAIGN_CAP,"PROJECT_CAMPAIGN_CAP")
 P={}; roots={}
 for stage,p in (("G2",a.g2_prep),("G45",a.g45_prep)):
  q=Path(p)
  gate(q.parent==BASE and q.is_dir() and not q.is_symlink()
       and q.name.startswith("xp_a12_"+("g2" if stage=="G2" else "g45")+"_prepare_"),
       "PREP_PATH_"+stage)
  fname="XP_A12_G2_PREP_RECEIPT.json" if stage=="G2" else "XP_A12_G45_PREP_RECEIPT.json"
  x=jread(q/fname)
  gate(x["head"]==a.expected_head and x["no_GPU"] is True
       and x["status"]==("XP_A12_G2_INPUTS_FROZEN_CPU_PASS"
                         if stage=="G2" else "XP_A12_G45_INPUTS_FROZEN_CPU_PASS"),
       "FROZEN_PREP_"+stage)
  P[stage]=x;roots[stage]=q
 gate(P["G2"]["n_roots"]==2880 and P["G45"]["n_new"]==1120,"GRID")
 return wt,roots,P,L,g,start,campaign
def task_set(roots,P):
 jobs={}
 for sig in ("0.02","0.10"):
  mf=Path(P["G2"]["per_sigma"][sig]["path"])
  gate(mf.parent==roots["G2"] and sha(mf)==P["G2"]["per_sigma"][sig]["sha256"],
       "G2_SHA_"+sig)
  rows=jread(mf)["rows"]
  gate(len(rows)==1440 and all(rows[i]["image_id"]==
       rows[i+7]["image_id"] for i in range(0,1440,8)),"G2_GRID")
  # Two separate images, exactly one root each, first as smoke.
  smoke=[dict(stage="G2",kind="SMOKE",sig=sig,start=i,count=1)
         for i in (0,8)]
  prod=[dict(stage="G2",kind="PROD",sig=sig,start=i,count=n)
        for i,n in [(1,7),(9,7)]+[(k,8) for k in range(16,1440,8)]]
  jobs["G2_SIGMA_"+sig]=(smoke,prod)
 for stage in ("G4","G5"):
  def dm(phase):
   key=f"{stage}_{phase}";v=P["G45"]["stages"][key]
   mf=Path(v["manifest"])
   gate(mf.parent==roots["G45"] and sha(mf)==v["sha256"],
        "G45_MANIFEST_"+key)
   rows=jread(mf)["rows"]
   gate(len(rows)==v["n"],"G45_GRID_"+key)
   return rows
  rr=dm("REPLAY"); pp=dm("PROD")
  gate(len(rr)==2 and len(pp)==N[stage],"G45_COUNTS")
  jobs[stage]=([dict(stage=stage,kind="REPLAY",index=i) for i in range(2)],
               [dict(stage=stage,kind="PROD",index=i) for i in range(N[stage])])
 return jobs
def make_command(a,run,task,gpu,roots,P,attempt):
 stage=task["stage"];kind=task["kind"]
 if stage=="G2":
  sig=task["sig"];rec=P["G2"]["per_sigma"][sig]
  out=run/"G2"/("sigma_"+sig)/f"chunk_{task['start']:05d}_{task['count']:03d}"
  cmd=[str(ROOT_FAST_PY),str(Path(__file__).with_name("xp_a12_campaign_fast_gpu.py")),
    "--expected-head",a.expected_head,
    "--roots",rec["path"],"--roots-sha",rec["sha256"],
    "--roots-source-head",P["G2"]["head"],"--out",str(out),
    "--start",str(task["start"]),"--count",str(task["count"]),
    "--physical-gpu",str(gpu),"--dataset","ffhq","--oversample","2.0"]
  if attempt>0 and out.exists():cmd+=["--resume"]
  return cmd,out,rec["sha256"],str(ROOT_FAST_PY)
 phase="REPLAY" if kind=="REPLAY" else "PROD"
 rec=P["G45"]["stages"][stage+"_"+phase]
 m=jread(rec["manifest"]);row=m["rows"][task["index"]]
 dest=run/stage/phase/"gpu"/(
   "root_"+hashlib.sha256(row["uid"].encode()).hexdigest()[:24])
 cmd=[str(ROOT_DAPS_PY),
   str(Path(__file__).with_name("xp_a12_g45_daps_gpu.py")),
   "--expected-head",a.expected_head,
   "--roots-source-head",P["G45"]["head"],
   "--run",str(run),"--prep",str(roots["G45"]),
   "--stage",stage,"--phase",phase,
   "--index",str(task["index"]),"--gpu",str(gpu)]
 return cmd,dest,rec["sha256"],str(ROOT_DAPS_PY)
def verified(a,run,task,dest,source_sha,roots,P):
 if task["stage"]=="G2":
  rec=P["G2"]["per_sigma"][task["sig"]]
  mf=jread(rec["path"])
  chunk=jread(dest/"R0_5_CHUNK_COMPLETE.json")
  gate(chunk["status"]=="R0_5_CHUNK_PASS"
       and chunk["head"]==a.expected_head
       and chunk["new"]==task["count"]
       and chunk["start"]==task["start"]
       and chunk["count"]==task["count"]
       and chunk["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
       "G2_NATIVE_CHUNK")
  for row in mf["rows"][task["start"]:task["start"]+task["count"]]:
   r=validate_complete(root_folder(dest,row),row,
         a.expected_head,source_sha)
   gate(r["gt_access"] is False,"G2_NO_GT")
  return sha(dest/"R0_5_CHUNK_COMPLETE.json")
 stage=task["stage"];phase=task["kind"]
 m=jread(roots["G45"]/f"XP_A12_{stage}_{phase}_TASKS.json")
 row=m["rows"][task["index"]]
 receipt=dest/"XP_T_DAPS4K_ROOT_COMPLETE.json"
 d=jread(receipt)
 gate(d["status"]=="XP_T_DAPS4K_ROOT_PASS"
      and d["uid"]==row["uid"] and d["rep"]==row["rep"]
      and d["runner_head"]==a.expected_head
      and d["seed"]==row["seed"] and d["manifest_sha256"]==source_sha
      and d["method"]=="DAPS-4K" and d["model_load_strict"] is True
      and d["model_load_trace"]["unet_calls"]==1
      and d["gt_free_generation"] is True
      and d["measurement_sha256"]==row["measurement_sha256"]
      and sha(dest/"reconstruction.pt")==d["reconstruction_pt_sha256"]
      and sha(dest/"reconstruction.png")==d["reconstruction_png_sha256"],
      "G45_NATIVE_RECEIPT")
 if phase=="REPLAY":
  old=BASE/("xp_a6_20261027T165112Z" if stage=="G4"
           else "xp_overnight_20261027T045215Z")
  earlier=old/"gpu"/dest.name/"reconstruction.png"
  gate(earlier.is_file(),"G45_HISTORICAL_ROOT_MISSING")
  with Image.open(earlier) as x,Image.open(dest/"reconstruction.png") as y:
   gate(x.mode==y.mode and x.size==y.size and
        ImageChops.difference(x,y).getbbox() is None,
        "G45_REPLAY_PIXEL_MISMATCH")
 return sha(receipt)
def smoke_projection(run,stage,smoke,P,roots,stage_seconds):
 """Use native per-root sampling wall times, not CPU/model import overhead.

 Historical DAPS sampling often takes ~33-66 s, FAST ~3-15 s.
 A12 §1 requires a projection after TWO stage-native replay/smoke roots.
 Conservatively inflate native time 1.25x; never change the method.
 """
 times=[]
 for job in smoke:
  if stage=="G2":
   mf=jread(P["G2"]["per_sigma"][job["sig"]]["path"])
   row=mf["rows"][job["start"]]
   dest=run/"G2"/("sigma_"+job["sig"])/(
     f"chunk_{job['start']:05d}_{job['count']:03d}")
   root=root_folder(dest,row)
   rc=jread(root/"R0_FAST_ROOT_COMPLETE.json")
   gate(rc["status"]=="R0_FAST_ROOT_PASS","SMOKE_FAST_PASS")
   t=float(rc["native_sample_wall_seconds"])
  else:
   phase="REPLAY"
   mf=jread(roots["G45"]/f"XP_A12_{stage}_{phase}_TASKS.json")
   row=mf["rows"][job["index"]]
   dest=run/stage/phase/"gpu"/("root_"+
       hashlib.sha256(row["uid"].encode()).hexdigest()[:24])
   rc=jread(dest/"XP_T_DAPS4K_ROOT_COMPLETE.json")
   gate(rc["status"]=="XP_T_DAPS4K_ROOT_PASS","SMOKE_DAPS_PASS")
   t=float(rc["native_sampling_wall_s"])
  gate(math.isfinite(t) and 0<t<1800,"SMOKE_TIMING_FINITE")
  times.append(t)
 gate(len(times)==2,"TWO_ROOT_TIMING_GATE")
 remaining=(1440 if stage=="G2" else N[stage])-2
 # stage_seconds already includes the two sampled workers, including
 # their model-load overhead. Do not count those roots twice.
 projected=stage_seconds+1.25*max(times)*remaining
 gate(projected/3600<=CAP[stage],
      "STAGE_SMOKE_PROJECTS_OVER_CEILING_"+stage)
 return dict(smoke_native_seconds=times,projected_gpu_h=projected/3600,
    comparison_ceiling_gpu_h=CAP[stage],
    upper_estimate_multiplier=1.25,projection_pass=True)

def run_queue(a,run,stage,tasks,roots,P,state,project0,campaign0):
 active={}
 draining_reason=None
 cap=CAP[stage]*3600.
 def save(status):
  state["status"]=status
  state["active"]=[dict(job=j,gpu=v["gpu"],pid=v["proc"].pid)
                   for j,v in active.values()]
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
 def current_spend():
  return state["stage_seconds"][stage]+sum(
    time.monotonic()-v["start"] for j,v in active.values())
 def total_spend():
  return sum(state["stage_seconds"].values())+sum(
    time.monotonic()-v["start"] for j,v in active.values())
 def die_own():
  for _,v in active.values():
   if v["proc"].poll() is None:
    try:os.killpg(v["proc"].pid,signal.SIGTERM)
    except ProcessLookupError:pass
 def settle():
  for idx,(job,v) in list(active.items()):
   rc=v["proc"].poll()
   if rc is None:continue
   v["handle"].close()
   elapsed=time.monotonic()-v["start"]
   state["stage_seconds"][stage]+=elapsed
   name=f"{stage}_{job['kind']}_{idx}_A{v['attempt']}"
   write(run/"cost"/(name+".json"),dict(status="A12_ATTEMPT_GPU_COST",
     stage=stage,job=job,rc=rc,elapsed_seconds=elapsed,
     charged_gpu_h=elapsed/3600,
     source_head=a.expected_head,gpu=v["gpu"],pid=v["proc"].pid,
     log=str(v["log"]),utc=datetime.datetime.now(
         datetime.timezone.utc).isoformat()))
   del active[idx]
   if rc!=0 and draining_reason is not None:
    emit(run,"INTERRUPTED_OWN_WORKER_CHARGED",name=name,reason=draining_reason,
         charged_seconds=elapsed)
    save("DRAINING_"+stage+"_"+draining_reason)
    continue
   if rc!=0:
    raise RuntimeError("STOP_A12_NATIVE_RC_"+name+"_"+str(rc))
   digest=verified(a,run,job,v["dest"],v["root_sha"],roots,P)
   state["done"][str(idx)]=digest
   emit(run,"ROOT_VERIFIED",job=job,digest=digest)
   save("RUN_"+stage)
 def loop():
  nonlocal draining_reason
  save("RUN_"+stage)
  while len(state["done"])<len(tasks):
   cards=collect(own_roots=[v["proc"].pid for j,v in active.values()])
   settle()
   remaining=[(idx,j) for idx,j in enumerate(tasks)
              if str(idx) not in state["done"] and idx not in active]
   deadline=datetime.datetime(*DEADLINE[stage],tzinfo=ZoneInfo("America/Detroit"))
   stop_reason=(draining_reason if draining_reason is not None
      else "OPERATOR_STOP" if STOP or (run/"STOP_A12").exists()
      else "DEADLINE" if datetime.datetime.now(ZoneInfo("America/Detroit"))>=deadline
      else "STAGE_CEILING" if current_spend()>=cap-30
      else "CAMPAIGN_CEILING" if (campaign0*3600+total_spend())>=CAMPAIGN_CAP*3600-30
      else "PROJECT_CEILING" if (project0*3600+total_spend())>=PROJECT_CAP*3600-30
      else None)
   if stop_reason:
    draining_reason=stop_reason
    die_own()
    if not active:
     save("PARTIAL_"+stage+"_"+stop_reason)
     return False
    time.sleep(2)
    continue
   occupied={v["gpu"] for _,v in active.values()}
   for idx,job in remaining:
    if len(active)>=4:break
    # Reserve enough headroom for each active model load and one run.
    if (current_spend()+600*(len(active)+1)>cap or
        campaign0*3600+total_spend()+600*(len(active)+1)>CAMPAIGN_CAP*3600 or
        project0*3600+total_spend()+600*(len(active)+1)>PROJECT_CAP*3600):
     break
    candidates=[c for c in cards if c.index not in occupied
                and c.visibility=="known"]
    gpu,_=choose_gpu(candidates,RESERVE_MIB[stage],
        safety_margin_mib=2048,fraction=.80,clean_only=False)
    if gpu is None:break
    cmd,dest,digest,exe=make_command(a,run,job,gpu,roots,P,0)
    gate(not dest.exists(),"NEW_TASK_OUTPUT_ALREADY_EXISTS")
    key=f"{stage}_{job['kind']}_{idx}"
    log=run/"logs"/(key+".log")
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(Path(__file__).resolve().parents[2]),
      PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
      PYTHONPYCACHEPREFIX=str(run/"pycache"/key),
      TMPDIR=str(ROOT/"tmp"),CUBLAS_WORKSPACE_CONFIG=":4096:8",
      OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
      OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",
      B20_LF_ENABLE="0",B21_CONT_ENABLE="0")
    handle=log.open("x")
    proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[2]),env=env,start_new_session=True)
    active[idx]=(job,dict(proc=proc,gpu=gpu,start=time.monotonic(),
        dest=dest,root_sha=digest,log=log,handle=handle,attempt=0))
    occupied.add(gpu)
    emit(run,"TASK_LAUNCHED",stage=stage,job=job,gpu=gpu,pid=proc.pid)
    save("RUN_"+stage)
   time.sleep(2)
  save(stage+"_COMPLETE")
  return True
 try:return loop()
 except BaseException:
  die_own()
  for idx,(job,v) in list(active.items()):
   v["proc"].wait()
   v["handle"].close()
   elapsed=time.monotonic()-v["start"]
   state["stage_seconds"][stage]+=elapsed
   name=f"{stage}_{job['kind']}_{idx}_A_EXCEPTION"
   write(run/"cost"/(name+".json"),dict(status="A12_FAILED_ATTEMPT_CHARGED",
      job=job,seconds=elapsed,charged_gpu_h=elapsed/3600,gpu=v["gpu"],
      pid=v["proc"].pid,returncode=v["proc"].returncode))
   del active[idx]
  save("STOP_"+stage+"_EXCEPTION")
  raise
def main(a):
 wt,roots,P,ledger,g1a,project0,campaign0=prepar(a)
 jobs=task_set(roots,P)
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_campaign_prod_")
      and not run.exists(),"NEW_CAMPAIGN_RUN")
 with (BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for p in ("logs","cost","G2","G4","G5","pycache"):
   (run/p).mkdir()
  for sig in ("0.02","0.10"):(run/"G2"/("sigma_"+sig)).mkdir()
  for stage in ("G4","G5"):
   for phase in ("REPLAY","PROD"):(run/stage/phase/"gpu").mkdir(parents=True)
  state=dict(status="ADMITTED",head=a.expected_head,
    project_initial_gpu_h=project0,campaign_initial_gpu_h=campaign0,
    parent_ledger_sha256=sha(a.ledger),G1a_final_sha256=sha(G1A),
    stage_seconds={x:0. for x in CAP},done={},active=[],
    G3_status="NOT_ASSESSED",G1b_status="NOT_AUTHORIZED_TO_AUTO_LAUNCH",
    G6_status="CONDITIONAL_NO_WORK_UNTIL_CLEAN_AUDIT",
    G7_status="NOT_YET_ADMITTED",
    GT_not_read=True,D4_outcomes_not_opened=True)
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
  for stage in ("G2","G4","G5"):
   phase_lists=(jobs["G2_SIGMA_0.02"],jobs["G2_SIGMA_0.10"]) if stage=="G2" else (jobs[stage],)
   success=True
   for smoke,production in phase_lists:
    for kind,subset in (("SMOKE_OR_REPLAY",smoke),("PRODUCTION",production)):
     state["done"]={}
     if not run_queue(a,run,stage,subset,roots,P,state,project0,campaign0):
      success=False
      break
     if kind=="SMOKE_OR_REPLAY":
      # Prospective projection using native sampling times, not model
      # import wall; if even the best-case projected stage exceeds cap,
      # refuse production rather than modifying prereg runtime limits.
      done=state["done"]
      estimate=smoke_projection(run,stage,smoke,P,roots,
             state["stage_seconds"][stage])
      emit(run,"STAGE_SMOKE_OR_REPLAY_PASS",stage=stage,
           completed=len(done),projection=estimate)
    if not success:break
   if not success:break
   emit(run,"STAGE_GENERATION_COMPLETE",stage=stage,
        n_roots=N[stage],gpu_h=state["stage_seconds"][stage]/3600)
   write(run/f"XP_A12_{stage}_FINAL_STATUS.json",dict(
     status=f"XP_A12_{stage}_GENERATION_COMPLETE",
     n_roots=N[stage],failed_attempts=0,
     charged_gpu_h=state["stage_seconds"][stage]/3600,
     stage_cap_gpu_h=CAP[stage],head=a.expected_head,
     outcomes_GT_not_read=True))
  state["status"]="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
  state["project_charged_gpu_h"]=project0+sum(state["stage_seconds"].values())/3600
  state["A12_charged_gpu_h"]=campaign0+sum(state["stage_seconds"].values())/3600
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
  print("A12_CAMPAIGN_AUTOPILOT_STATUS",json.dumps(dict(
    project_gpu_h=state["project_charged_gpu_h"],
    A12_gpu_h=state["A12_charged_gpu_h"],
    G2_G4_G5_stage_seconds=state["stage_seconds"],
    G3_status=state["G3_status"],G1b_status=state["G1b_status"],
    root=str(run)),sort_keys=True),flush=True)
def entry():
 p=argparse.ArgumentParser()
 for x in ("expected-head","ledger","g2-prep","g45-prep","run"):
  p.add_argument("--"+x,required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_CAMPAIGN_AUTOPILOT",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
