#!/usr/bin/env python3
"""A22 G7 C2 budgeted multi-GPU controller; conservative no-signal scheduler.

Runs existing native XP-C2 diagnostic worker. Exact 280-image frozen GT.
Up to 4 processes/GPU after pilot. Never contacts another user's process.
"""
import argparse,fcntl,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import collect
from scripts.track_r.xp_a29_a21_controller import total_union
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
def gate(ok,msg):
 if not ok:raise RuntimeError("STOP_A22_G7_"+msg)
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 tmp=Path(str(p)+".tmp")
 tmp.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
 tmp.replace(p)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_SUPERVISOR")
 head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
 gate(head==a.expected_head,"PIN")
 gate(not subprocess.check_output(["git","-C",str(WT),"status","--porcelain"]), "DIRTY")
 r=Path(a.run);f=Path(a.freeze)
 gate(r.parent==BASE and r.name.startswith("xp_a12_g7_prod_a22_")
      and not r.exists() and f.parent==BASE and f.name.startswith("xp_c_c2_a22_"),"OUTPUT")
 proof=rd(f/"XP_C_C2_PREP.json")
 gate(proof["status"]=="XP_C_C2_PREP_PASS" and proof["n_images"]==280
      and sha(f/"XP_C_GT_IMAGES.json")==proof["C_manifest_sha256"]
      and sha(f/"XP_C2_EQUIVARIANCE_TASKS.json")==proof["C2_manifest_sha256"],"GT_FREEZE")
 a21=rd(BASE/"xp_a25_xpc3_prod_a21_20261103T182034Z/XP_C3_A21_FINAL_STATUS.json")
 baseline=float(a21["project_gpu_h"])
 gate(200<baseline<396 and a21["active"]==[],"PROJECT_HEADROOM")
 if a.preflight_only:
  print("A22_G7_C2_PREFLIGHT_PASS",proof["C2_manifest_sha256"],flush=True)
  return
 with (BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  r.mkdir()
  for d in ("results","logs","pycache"):(r/d).mkdir()
  active={};intervals=[];finished={};events=[];peak=0;workers=1
  def accounting():
   hrs=total_union(intervals,active)
   return sum(hrs.values()),hrs
  def write(status):
   used,per=accounting()
   state=dict(status=status,completed=finished,worker_events=events,
     active=[dict(pid=k,index=v["index"],gpu=v["gpu"]) for k,v in active.items()],
     gpu_occupancy_h=used,physical_gpu_h=per,worker_limit_per_gpu=workers,
     project_gpu_h=baseline+used,project_baseline_h=baseline,
     stage_cap_h=4,model_peak_estimate_mib=peak,source_hash=proof["C2_manifest_sha256"],
     head=a.expected_head,gt_diagnostic_only=True,no_A21_10_replication=True)
   save(r/"XP_A22_G7_C2_STATE.json",state)
   return state
  def collect_done():
   for pid,v in list(active.items()):
    code=v["proc"].poll()
    if code is None:continue
    end=time.monotonic()
    del active[pid]
    intervals.append((v["gpu"],v["start"],end))
    path=r/"results"/("%03d_equivariance.json"%v["index"])
    valid=False
    if path.is_file():
     z=rd(path)
     valid=(z["status"]=="XP_A12_G7_C2_IMAGE_EQUIVARIANCE_PASS"
       and z["n"]==36 and len(z["records"])==36
       and z["source_C2_manifest_sha256"]==proof["C2_manifest_sha256"])
    events.append(dict(index=v["index"],gpu=v["gpu"],seconds=end-v["start"],
        rc=code,valid=valid))
    if valid:finished[str(v["index"])]=sha(path)
    else:raise RuntimeError("NATIVE_WORKER_INCOMPLETE_"+str(v["index"]))
  def next_index():
   busy={v["index"] for v in active.values()}
   return next((i for i in range(280) if str(i) not in finished and i not in busy),None)
  # Pilot one image on each GPU; ramp after successful completion.
  status="RUNNING";started=time.monotonic();deadline=time.time()+3.5*3600
  try:
   write(status)
   # Exercise the exact frozen native worker import and GT manifest first,
   # before allocating GPU memory for the C2 production sample.
   env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",
     PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
     PYTHONPYCACHEPREFIX=str(r/"pycache"/"C2_PREFLIGHT"))
   p=subprocess.run([str(ROOT/"conda-envs/daps/bin/python"),
     str(WT/"scripts/track_r/xp_a12_g7_c2_gpu.py"),
     "--expected-head",a.expected_head,"--freeze",str(f),
     "--run",str(r),"--index","0","--gpu","0","--preflight-only"],
     env=env,cwd=str(WT),capture_output=True,text=True,timeout=120)
   gate(p.returncode==0,"NATIVE_IMPORT_AND_SOURCE_PREFLIGHT_"+p.stderr[-220:])
   print("A22_G7_NATIVE_IMPORT_PREFLIGHT_PASS",flush=True)
   while len(finished)<280:
    collect_done()
    hrs,_=accounting()
    if hrs>=3.5 or baseline+hrs>=399.5 or time.time()>deadline:
     status="PARTIAL_SAFE_BUDGET_STOP";break
    if len(finished)>=4:workers=4
    cards={c.index:c for c in collect(own_roots=list(active))}
    for gpu in range(4):
     c=cards.get(gpu)
     if not c or c.visibility!="known":continue
     on=[v for v in active.values() if v["gpu"]==gpu]
     if len(on)>=workers:continue
     # Existing A12 C2 12,288-MiB request plus 8-GiB extra.
     need=24576
     reserved=sum(need for v in on if v["proc"].pid not in c.process_pids)
     if c.free_mib-reserved<need or c.used_mib+reserved+need>int(.8*c.total_mib):continue
     ix=next_index()
     if ix is None:continue
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
       PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
       PYTHONPYCACHEPREFIX=str(r/"pycache"/("C2_%03d"%ix)))
     cmd=[str(ROOT/"conda-envs/daps/bin/python"),
       str(WT/"scripts/track_r/xp_a12_g7_c2_gpu.py"),
       "--expected-head",a.expected_head,"--freeze",str(f),
       "--run",str(r),"--index",str(ix),"--gpu",str(gpu)]
     log=(r/"logs"/("C2_%03d.log"%ix)).open("x")
     proc=subprocess.Popen(cmd,env=env,cwd=str(WT),
       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     log.close()
     active[proc.pid]=dict(proc=proc,index=ix,gpu=gpu,start=time.monotonic())
    if int(time.monotonic()-started)%60<4:
     print("A22_G7_PROGRESS",len(finished),"/280","GPUh",round(hrs,3),flush=True)
     write(status)
    time.sleep(4)
   else:status="A22_G7_C2_280_COMPLETE"
  except BaseException:
   status="STOP_A22_G7_RECONCILE"
   raise
  finally:
   # No signals: drain only our existing chunks; never touch foreign jobs.
   while active:
    collect_done();time.sleep(2)
   state=write(status)
   save(r/"XP_A22_G7_C2_FINAL_STATUS.json",state)
   print("A22_G7_C2_TERMINAL",status,"images",len(finished),
       "GPU_OCCUPANCY_H",round(accounting()[0],3),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 for k in ("expected-head","freeze","run"):p.add_argument("--"+k,required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A22_G7",e,flush=True);sys.exit(8)
