#!/usr/bin/env python3
"""A12 G7 XP-C2 supplementary 280-image GPU controller.

Must be launched separately with the frozen XP-C/C2 diagnostic-only
GT manifest. Charges every occupied worker, no foreign GPU kills,
4h stage budget and A12/project shared cap checked against a
verified earlier campaign state. C2 only: XP-C orientation priors and
UnivFD still require their own approved diagnostic adapter.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
STOP=False
def stopping(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stopping)
signal.signal(signal.SIGINT,stopping)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for t in iter(lambda:f.read(1<<20),b""):h.update(t)
 return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_name("."+p.name+".tmp"+str(os.getpid()))
 with t.open("x") as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
 os.replace(t,p)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                  text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"WORKTREE")
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_g7_prod_")
      and not run.exists(),"OUTPUT")
 prior=Path(a.prior_campaign)
 gate(prior.name=="XP_A12_CAMPAIGN_STATE.json" and
      prior.parent.parent==BASE and
      prior.parent.name.startswith("xp_a12_campaign_prod_") and
      prior.is_file() and not prior.is_symlink(),
      "PRIOR_A12_CANONICAL_CONTROLLER_PATH")
 state=jread(prior)
 gate(state["status"]=="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
      and state.get("active")==[] and
      state["head"] and
      0<state["A12_charged_gpu_h"]<150
      and 0<state["project_charged_gpu_h"]<400
      and sum(state["stage_seconds"].values())>=0,
      "A12_PRIOR_ACCOUNTING")
 cap=4*3600
 project0=float(state["project_charged_gpu_h"])
 campaign0=float(state["A12_charged_gpu_h"])
 if a.g6_final:
  prior6=Path(a.g6_final)
  gate(prior6.name=="XP_A13_G6_FINAL_STATUS.json" and
       prior6.parent.parent==BASE and
       prior6.parent.name.startswith("xp_a12_g6_prod_") and
       prior6.is_file() and not prior6.is_symlink(),"G6_PRIOR_PATH")
  g6=jread(prior6)
  gate(g6["prior_campaign_sha256"]==sha(prior)
       and g6["cap_gpu_h"]==1.
       and 0<=g6["wall_gpu_h"]<=1.
       and abs(g6["A12_total_charged_gpu_h"]-
             (campaign0+g6["wall_gpu_h"]))<1e-6
       and abs(g6["project_total_charged_gpu_h"]-
             (project0+g6["wall_gpu_h"]))<1e-6,
       "G6_PRIOR_CONSISTENCY")
  project0=float(g6["project_total_charged_gpu_h"])
  campaign0=float(g6["A12_total_charged_gpu_h"])
 gate(project0+4<=400 and campaign0+4<=150,"SHARED_HEADROOM")
 freeze=Path(a.freeze)
 gate(freeze.parent==BASE and freeze.name.startswith("xp_c_c2_"),"FREEZE")
 receipt=jread(freeze/"XP_C_C2_PREP.json")
 gate(receipt["status"]=="XP_C_C2_PREP_PASS" and
      receipt["n_images"]==280 and receipt["n_C2_evaluations"]==10080,
      "FROZEN280")
 from scripts.track_r.xp_a12_g367_readiness_cpu import sha as canonical
 gate(canonical(freeze/"XP_C_GT_IMAGES.json")==receipt["C_manifest_sha256"]
      and canonical(freeze/"XP_C2_EQUIVARIANCE_TASKS.json")==
      receipt["C2_manifest_sha256"],"FROZEN_SHA")
 with (BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for x in ("results","logs","cost","pycache"):(run/x).mkdir()
  completed={};active={};failed=[];used_s=0.
  def report(status):
   data=dict(status=status,expected_head=a.expected_head,
     completed_images=len(completed),
     completed_C2_records=len(completed)*36,
     stage_charged_gpu_h=used_s/3600,
     stage_ceiling_gpu_h=4.,
     A12_charged_gpu_h=campaign0+used_s/3600,
     project_charged_gpu_h=project0+used_s/3600,
     completed=completed,active=[dict(index=i,gpu=v["gpu"],pid=v["proc"].pid)
        for i,v in active.items()],failed=failed,
     prior_campaign_sha256=sha(prior),frozen_C2_sha256=receipt["C2_manifest_sha256"],
     no_inverse_solver_GT_access=True)
   write(run/"XP_A12_G7_C2_CONTROLLER_STATE.json",data)
   return data
  status="RUNNING"
  try:
   report(status)
   while len(completed)<280:
    for i,v in list(active.items()):
     rc=v["proc"].poll()
     if rc is None:continue
     v["handle"].close()
     elapsed=time.monotonic()-v["start"];used_s+=elapsed
     receiptpath=run/"cost"/f"C2_{i:03d}.json"
     write(receiptpath,dict(stage="G7_C2",index=i,
       gpu=v["gpu"],pid=v["proc"].pid,rc=rc,
       charged_gpu_seconds=elapsed,
       log=str(v["log"])))
     del active[i]
     if rc!=0:
      failed.append(dict(index=i,rc=rc,log=str(v["log"])))
      raise RuntimeError("STOP_A12_G7_NATIVE_RC_"+str(i))
     record=run/"results"/f"{i:03d}_equivariance.json"
     d=jread(record)
     gate(d["status"]=="XP_A12_G7_C2_IMAGE_EQUIVARIANCE_PASS"
          and d["n"]==36
          and d["source_C2_manifest_sha256"]==receipt["C2_manifest_sha256"],
          "G7_OUTPUT_"+str(i))
     completed[str(i)]=sha(record)
     if len(completed)%20==0:
      print("A12_G7_C2_VERIFIED",len(completed),"/280",flush=True)
     report("RUNNING")
    progress=used_s+sum(time.monotonic()-v["start"] for v in active.values())
    deadline=datetime.datetime(2026,11,15,23,59,
         tzinfo=ZoneInfo("America/Detroit"))
    stopreason=("OPERATOR" if STOP or (run/"STOP_G7").exists() else
     "STAGE_CEILING" if progress>=cap-60 else
     "DEADLINE" if datetime.datetime.now(ZoneInfo("America/Detroit"))>=deadline else
     "GLOBAL_CAP" if campaign0*3600+progress>=150*3600-60 or
                     project0*3600+progress>=400*3600-60 else None)
    if stopreason:
     for v in active.values():
      if v["proc"].poll() is None:
       try:os.killpg(v["proc"].pid,signal.SIGTERM)
       except ProcessLookupError:pass
     if not active:
      status="XP_A12_G7_C2_PARTIAL_"+stopreason;break
     time.sleep(2);continue
    cards=collect(own_roots=[v["proc"].pid for v in active.values()])
    occupied={v["gpu"] for v in active.values()}
    for i in range(280):
     if len(active)>=4:break
     if str(i) in completed or i in active:continue
     if progress+900*(len(active)+1)>cap:break
     cards_free=[c for c in cards if c.index not in occupied
                  and c.visibility=="known"]
     gpu,_=choose_gpu(cards_free,12288,safety_margin_mib=2048,
                      fraction=.80,clean_only=False)
     if gpu is None:break
     env=os.environ.copy()
     env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(wt),
      PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
      PYTHONPYCACHEPREFIX=str(run/"pycache"/f"C2_{i:03d}"),
      CUBLAS_WORKSPACE_CONFIG=":4096:8",TMPDIR=str(ROOT/"tmp"),
      OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
      OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
     cmd=[str(ROOT/"conda-envs/daps/bin/python"),
       str(wt/"scripts/track_r/xp_a12_g7_c2_gpu.py"),
       "--expected-head",a.expected_head,
       "--freeze",str(freeze),"--run",str(run),
       "--index",str(i),"--gpu",str(gpu)]
     log=run/"logs"/f"C2_{i:03d}.log"
     handle=log.open("x")
     proc=subprocess.Popen(cmd,cwd=str(wt),env=env,
       stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
     active[i]=dict(proc=proc,gpu=gpu,start=time.monotonic(),
                    handle=handle,log=log)
     occupied.add(gpu)
     report("RUNNING")
    time.sleep(3)
   else:status="XP_A12_G7_C2_280_IMAGES_COMPLETE"
  except BaseException:
   status="STOP_A12_G7_CONTROLLER_EXCEPTION"
   raise
  finally:
   for i,v in list(active.items()):
    if v["proc"].poll() is None:
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
    v["proc"].wait();v["handle"].close()
    elapsed=time.monotonic()-v["start"];used_s+=elapsed
    write(run/"cost"/f"C2_{i:03d}_INTERRUPTED.json",
      dict(index=i,charged_gpu_seconds=elapsed,
           charged_gpu_h=elapsed/3600,exitcode=v["proc"].returncode))
    del active[i]
   final=report(status)
   write(run/"XP_A12_G7_C2_FINAL_STATUS.json",final)
   print("A12_G7_C2_STATUS",status,"completed",len(completed),
         "charged_gpu_h",used_s/3600,flush=True)
def entry():
 ap=argparse.ArgumentParser()
 for x in ("expected-head","prior-campaign","freeze","run"):
  ap.add_argument("--"+x,required=True)
 ap.add_argument("--g6-final",default=None)
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G7_C2",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
