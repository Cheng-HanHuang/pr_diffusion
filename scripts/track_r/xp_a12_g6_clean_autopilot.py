#!/usr/bin/env python3
"""A12 G6 optional clean timing controller: 9 FAST-IN, 9 DAPS-4K FFHQ.

Only uses truly CLEAN GPUs; no foreign PID is terminated or manipulated.
If previous clean receipts already cover both required batches, the
preparation stage writes no run, so this controller cannot launch.
Stage cap 1 GPU-hour, charged process wall time for every attempt.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g45_prepare_cpu import sha
from scripts.track_r.xp_a12_g6_prepare_cpu import timing_check
DAPS=ROOT/"conda-envs/daps/bin/python"
FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
def write(p,x):
 with Path(p).open("x") as f:json.dump(x,f,indent=2,sort_keys=True);f.write("\n")
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_CONTROLLER")
 gate(not all(timing_check()[x] for x in ("fast_clean","daps_clean")),
      "CLEAN_TIMINGS_ALREADY_EXIST")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *xs:subprocess.check_output(["git","-C",str(wt),*xs],
                  text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"WORKTREE")
 prep=Path(a.prep);run=Path(a.run)
 gate(prep.parent==BASE and prep.name.startswith("xp_a12_g6_prepare_")
      and run.parent==BASE and run.name.startswith("xp_a12_g6_prod_")
      and not run.exists(),"OUTPUT")
 rec=json.loads((prep/"XP_A12_G6_PREP_RECEIPT.json").read_text())
 gate(rec["status"]=="XP_A12_G6_CONDITIONAL_CLEAN_TIMING_PREP_PASS"
      and rec["head"]==a.expected_head,"PREP")
 f=prep/"XP_A12_G6_FAST_IN_TASKS.json"
 d=prep/"XP_A12_G6_DAPS_FFHQ_TASKS.json"
 gate(sha(f)==rec["fast_manifest_sha256"] and
      sha(d)==rec["daps_manifest_sha256"],"SHA")
 with (BASE/"XP_A12_G6_CONTROLLER.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for sub in ("gpu","fast","logs","pycache"):(run/sub).mkdir()
  tasks=[("FAST",0)]+[("DAPS",i) for i in range(9)]
  total_s=0.;cost=[];dirty=False
  for stage,index in tasks:
   gate(total_s+300<3600,"STAGE_1GPUH_CAP")
   # Wait for a verified clean card, not merely low utilization.
   started=time.monotonic();chosen=None
   while time.monotonic()-started<300:
    cards=collect()
    gpu,_=choose_gpu([x for x in cards if x.visibility=="known"
       and not x.foreign_pids and x.used_mib<=2048],
       12288,safety_margin_mib=2048,fraction=.80,clean_only=True)
    if gpu is not None:chosen=gpu;break
    time.sleep(5)
   gate(chosen is not None,"NO_CLEAN_GPU_AVAILABLE")
   if stage=="FAST":
    dest=run/"fast"/"chunk_00000_009"
    cmd=[str(FAST),str(wt/"scripts/track_r/xp_a12_campaign_fast_gpu.py"),
      "--expected-head",a.expected_head,"--roots",str(f),
      "--roots-sha",rec["fast_manifest_sha256"],
      "--roots-source-head",rec["head"],"--out",str(dest),
      "--start","0","--count","9","--physical-gpu",str(chosen),
      "--dataset","imagenet","--oversample","2.0"]
   else:
    cmd=[str(DAPS),str(wt/"scripts/track_r/xp_a12_g6_daps_gpu.py"),
      "--run",str(run),"--prep",str(prep),
      "--expected-head",a.expected_head,
      "--roots-source-head",rec["head"],
      "--index",str(index),"--gpu",str(chosen)]
   env=dict(os.environ)
   env.update(CUDA_VISIBLE_DEVICES=str(chosen),PYTHONPATH=str(wt),
     PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
     PYTHONPYCACHEPREFIX=str(run/"pycache"/f"{stage}_{index}"),
     CUBLAS_WORKSPACE_CONFIG=":4096:8",TMPDIR=str(ROOT/"tmp"),
     OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
   p=run/"logs"/f"{stage}_{index:03d}.log"
   t0=time.monotonic();foreign=set()
   with p.open("x") as handle:
    proc=subprocess.Popen(cmd,cwd=str(wt),env=env,
      stdout=handle,stderr=subprocess.STDOUT,
      start_new_session=True)
    while proc.poll() is None:
     cards=collect(own_roots=[proc.pid])
     card=next((x for x in cards if x.index==chosen),None)
     if card is None or card.visibility!="known":
      dirty=True
     else:
      foreign.update(card.foreign_pids)
      if card.foreign_pids:dirty=True
     time.sleep(3)
    rc=proc.wait()
   elapsed=time.monotonic()-t0;total_s+=elapsed
   row=dict(stage=stage,index=index,gpu=chosen,
     charged_gpu_seconds=elapsed,rc=rc,
     contaminated_or_visibility_unknown=dirty,
     foreign_pids_seen=sorted(foreign),
     source_head=a.expected_head,log=str(p))
   cost.append(row)
   write(run/"gpu"/f"G6_COST_{stage}_{index:03d}.json",row)
   gate(rc==0,"NATIVE_EXIT_"+stage+"_"+str(index))
   if stage=="FAST":
    x=json.loads((dest/"R0_5_CHUNK_COMPLETE.json").read_text())
    gate(x["status"]=="R0_5_CHUNK_PASS" and x["new"]==9,"FAST_NINE")
   else:
    mf=json.loads(d.read_text());r=mf["rows"][index]
    token=hashlib.sha256(r["uid"].encode()).hexdigest()[:24]
    p=run/"gpu"/("root_"+token)/"XP_T_DAPS4K_ROOT_COMPLETE.json"
    x=json.loads(p.read_text())
    gate(x["status"]=="XP_T_DAPS4K_ROOT_PASS"
         and x["manifest_sha256"]==rec["daps_manifest_sha256"]
         and x["uid"]==r["uid"],"DAPS_ONE")
   gate(total_s<3600,"G6_CAP_OVERRUN")
   print("A12_G6_NATIVE_ROOTS_PASS",stage,index,
         "charged_s",round(total_s,2),flush=True)
   if dirty:
    print("A12_G6_DIRTY_TIMING_STOP_NO_MORE_JOBS",flush=True)
    break
  result=dict(status=("XP_A12_G6_CLEAN_TIMING_COMPLETE"
       if not dirty and len(cost)==10 else "XP_A12_G6_PARTIAL_OR_DIRTY"),
       n_processes=len(cost),native_fast_roots=9 if cost else 0,
       native_daps_roots=sum(x["stage"]=="DAPS" for x in cost),
       wall_gpu_h=total_s/3600,cap_gpu_h=1.,
       clean_timing_valid=not dirty and len(cost)==10,
       source_head=a.expected_head,costs=cost,
       no_ground_truth_access=True)
  write(run/"XP_A12_G6_FINAL_STATUS.json",result)
  print("A12_G6_TERMINAL",json.dumps(result,sort_keys=True),flush=True)
def entry():
 p=argparse.ArgumentParser()
 for x in ("expected-head","prep","run"):p.add_argument("--"+x,required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G6_CLEAN",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
