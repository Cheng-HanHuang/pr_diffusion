#!/usr/bin/env python3
"""A15 CP3 CLEAN GPU timing extension: three same-y FFHQ IDs,
one PC-DAPS native 4-particle bag and one DAPS-4K 400x10 root per ID.

G6's prior 400x5 F8 timing is explicitly NOT the DAPS-4K comparator.
All source manifests are reused without edits; no GT used by solvers.
No foreign GPU process is terminated. Excludes model load from reported
sampling wall; process wall charged separately for the project budget.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_a15_cp3_score_cpu import G3_PREP,G3_MANIFEST_SHA,A14_HEAD
from scripts.track_r.xp_a12_g45_prepare_cpu import DAPS4K,DAPS_HEAD
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

PY_PC=ROOT/"conda-envs/prdiff_track_r/bin/python"
PY_DAPS=ROOT/"conda-envs/daps/bin/python"
G6_F8_NOTE="G6_prior_DAPS400x5_does_NOT_measure_DAPS4K_400x10"
STAGE_CAP_GPU_H=3.
PROJECT_CAP_GPU_H=400.
def prepare():
 pcrec=jread(G3_PREP/"XP_A12_G3_PREP_RECEIPT.json")
 pcmanifest=G3_PREP/"XP_A12_G3_180_BAGS.json"
 pc=jread(pcmanifest)
 gate(sha(pcmanifest)==G3_MANIFEST_SHA==pcrec["manifest_sha256"]
      and pc["status"]=="XP_A12_G3_180_NATIVE_BAGS_FROZEN"
      and pc["n_bags"]==180 and pc["bag_size"]==4,
      "A17_TIMING_PC_SOURCE")
 matches=[]
 for p in BASE.glob("xp_a12_g45_prepare_*/XP_A12_G45_PREP_RECEIPT.json"):
  r=jread(p)
  if (r.get("head")==A14_HEAD and
      r.get("status")=="XP_A12_G45_INPUTS_FROZEN_CPU_PASS"):
   matches.append((p.parent,r))
 gate(len(matches)==1,"A17_TIMING_G4_PREP_UNIQUE")
 dp,dr=matches[0]
 manifest=dp/"XP_A12_G4_PROD_TASKS.json"
 daps=jread(manifest)
 gate(sha(manifest)==dr["stages"]["G4_PROD"]["sha256"]
      and daps["settings"]==DAPS4K
      and daps["source_head"]==DAPS_HEAD
      and daps["n_roots"]==720,
      "A17_TIMING_DAPS4K_SOURCE")
 pcfirst=sorted(((i,r) for i,r in enumerate(pc["rows"])
                 if r["cohort"]=="S1_b22"),
                key=lambda z:z[1]["image_id"])[:3]
 gate(len(pcfirst)==3,"A17_TIMING_FIRST3_B22")
 by_d={}
 for i,r in enumerate(daps["rows"]):
  if r["rep"]==4 and r["image_id"] not in by_d:
   by_d[r["image_id"]]=(i,r)
 jobs=[]
 for i,pr in pcfirst:
  did=pr["image_id"]
  gate(did in by_d,"A17_TIMING_PAIR_ID_"+did)
  j,drw=by_d[did]
  gate(pr["measurement_sha256"]==drw["measurement_sha256"]
       and sha(pr["measurement_path"])==sha(drw["measurement_path"])
       ==pr["measurement_sha256"],"A17_TIMING_SAME_Y_"+did)
  jobs.extend([dict(stage="PC",index=i,image_id=did),
               dict(stage="DAPS4K",index=j,image_id=did)])
 gate(len(jobs)==6,"A17_TIMING_3PAIRS")
 return dp,pcrec,dr,jobs

def budget():
 origin=BASE/"xp_a12_campaign_prod_20261030T022725Z"/"XP_A12_CAMPAIGN_STATE.json"
 st=jread(origin)
 gate(st["status"]=="A12_CAMPAIGN_SEQUENCED_STAGE_STOP_OR_COMPLETE"
      and st.get("active")==[],"A17_TIMING_SOURCE_A12_TERMINAL")
 project=float(st["project_charged_gpu_h"])
 gate(project+STAGE_CAP_GPU_H<=PROJECT_CAP_GPU_H,
      "A17_TIMING_PROJECT_BUDGET")
 return origin,project

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A17_TIMING_CONTROLLER_CPU")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *xs:subprocess.check_output(["git","-C",str(wt),*xs],
             text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"A17_TIMING_WT")
 dp,pc_source,dr,jobs=prepare()
 origin,project=budget()
 gate(PY_PC.is_file() and PY_DAPS.is_file(),"A17_TIMING_NATIVE_ENVS")
 if a.preflight_only:
  cards=collect()
  good=[x for x in cards if x.visibility=="known" and not x.foreign_pids and x.used_mib<=2048]
  print("A17_CP3_CLEAN_TIMING_PREFLIGHT_PASS",len(jobs),
        "GPU_CANDIDATES",len(good),
        "FROZEN_Y_HASH_PAIR_PASS",flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a17_cp3_timing_")
      and not out.exists() and not out.is_symlink(),
      "A17_TIMING_OUTPUT")
 stamp=out.name.split("xp_a17_cp3_timing_",1)[1]
 pc_run=BASE/("xp_a12_g3_prod_a17timing_"+stamp)
 daps_run=BASE/("xp_a12_g45_prod_a17timing_"+stamp)
 gate(not pc_run.exists() and not daps_run.exists(),
      "A17_TIMING_FRESH_NATIVE_OUTPUTS")
 out.mkdir(mode=0o700)
 (out/"logs").mkdir()
 pc_run.mkdir(mode=0o700)
 (pc_run/"roots").mkdir()
 (pc_run/"pycache").mkdir()
 daps_run.mkdir(mode=0o700)
 (daps_run/"G4"/"PROD"/"gpu").mkdir(parents=True)
 (daps_run/"pycache").mkdir()
 # This extension deliberately uses one truly clean GPU, serially.
 chosen=None;start=time.monotonic()
 while time.monotonic()-start<300:
  cards=collect()
  gpu,_=choose_gpu([x for x in cards if x.visibility=="known" and
                    not x.foreign_pids and x.used_mib<=2048],
         12288,safety_margin_mib=2048,fraction=.80,clean_only=True)
  if gpu is not None:
   chosen=gpu;break
  time.sleep(5)
 gate(chosen is not None,"A17_TIMING_NO_CLEAN_GPU")
 cost=0.;measurements=[];dirty=False
 for jid,job in enumerate(jobs):
  gate(cost<STAGE_CAP_GPU_H*3600,"A17_TIMING_CAP")
  stage=job["stage"];idx=job["index"]
  env=dict(os.environ)
  env.update(CUDA_VISIBLE_DEVICES=str(chosen),PYTHONPATH=str(wt),
      PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
      PYTHONPYCACHEPREFIX=str(
       (pc_run if stage=="PC" else daps_run)/"pycache"/f"{stage}_{idx}"),
      CUBLAS_WORKSPACE_CONFIG=":4096:8",
      TMPDIR=str(ROOT/"tmp"),OMP_NUM_THREADS="2",
      MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
  if stage=="PC":
   cmd=[str(PY_PC),str(wt/"scripts/track_r/xp_a12_g3_pcdaps_gpu.py"),
        "--run",str(pc_run),"--prep",str(G3_PREP),
        "--expected-head",a.expected_head,
        "--roots-source-head",pc_source["head"],
        "--index",str(idx),"--gpu",str(chosen)]
  else:
   cmd=[str(PY_DAPS),str(wt/"scripts/track_r/xp_a12_g45_daps_gpu.py"),
        "--run",str(daps_run),"--prep",str(dp),
        "--expected-head",a.expected_head,
        "--roots-source-head",dr["head"],
        "--stage","G4","--phase","PROD",
        "--index",str(idx),"--gpu",str(chosen)]
  log=out/"logs"/f"{jid:02d}_{stage}_{idx}.log"
  print("A17_TIMING_START",stage,job["image_id"],"GPU",chosen,flush=True)
  t=time.monotonic();seen=set();unknown=False
  with log.open("x") as handle:
   proc=subprocess.Popen(cmd,cwd=str(wt),env=env,stdout=handle,
                         stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    cards=collect(own_roots=[proc.pid])
    card=next((x for x in cards if x.index==chosen),None)
    if card is None or card.visibility!="known":
     unknown=True
    else:
     seen.update(card.foreign_pids)
    time.sleep(3)
   rc=proc.wait()
  elapsed=time.monotonic()-t;cost+=elapsed
  entry=dict(stage=stage,index=idx,image_id=job["image_id"],
             process_wall_seconds=elapsed,rc=rc,
             gpu_physical=chosen,clean=(not unknown and not seen),
             visibility_unknown=unknown,foreign_pids_seen=sorted(seen),
             native_log_sha256=sha(log))
  if rc==0:
   if stage=="PC":
    r=jread(pc_run/"roots"/("bag_"+hashlib.sha256(
         (jread(G3_PREP/"XP_A12_G3_180_BAGS.json")["rows"][idx]["uid"]).encode()
                         ).hexdigest()[:24])/"XP_A12_G3_BAG_COMPLETE.json")
    gate(r["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS"
         and r["image_id"]==job["image_id"]
         and r["runner_head"]==a.expected_head,
         "A17_TIMING_PC_RECEIPT")
    entry["native_sampling_wall_seconds"]=r["native_sampling_wall_seconds"]
    entry["native_prune_events"]=r["native_prune_events"]
   else:
    d=jread(dp/"XP_A12_G4_PROD_TASKS.json")["rows"][idx]
    rp=daps_run/"G4"/"PROD"/"gpu"/("root_"+hashlib.sha256(
       d["uid"].encode()).hexdigest()[:24])/"XP_T_DAPS4K_ROOT_COMPLETE.json"
    r=jread(rp)
    gate(r["status"]=="XP_T_DAPS4K_ROOT_PASS"
         and r["method"]=="DAPS-4K"
         and r["solver_config"]==DAPS4K
         and r["image_id"]==job["image_id"]
         and r["runner_head"]==a.expected_head,
         "A17_TIMING_DAPS_RECEIPT")
    entry["native_sampling_wall_seconds"]=r["native_sampling_wall_s"]
    entry["native_sampling_cuda_event_seconds"]=r["native_sampling_cuda_event_s"]
  measurements.append(entry)
  print("A17_TIMING_RESULT",stage,rc,round(elapsed,3),
        "CLEAN",entry["clean"],flush=True)
  if rc!=0 or not entry["clean"] or cost>=STAGE_CAP_GPU_H*3600:
   dirty=True;print("A17_TIMING_STOP_PARTIAL_OR_DIRTY",flush=True);break
  # No reusing a GPU newly occupied by a labmate.
  card=next((x for x in collect() if x.index==chosen),None)
  if card is None or card.foreign_pids or card.visibility!="known":
   dirty=True;print("A17_TIMING_STOP_POST_GPU_NOT_CLEAN",flush=True);break
 valid=(not dirty and len(measurements)==6 and
        all(e["rc"]==0 and e["clean"] for e in measurements))
 summary=dict(status=("A17_CP3_CLEAN_PAIRED_TIMING_PASS" if valid
                      else "A17_CP3_TIMING_PARTIAL_OR_DIRTY"),
    expected_head=a.expected_head,source_A12_state_sha256=sha(origin),
    A12_project_gpu_hours_before=project,
    additional_charged_gpu_hours=cost/3600,
    gpu_physical=chosen,clean_timing_valid=valid,
    three_same_signed_y_ids=True,
    timing_scope="native_sampling_only_excludes_model_load",
    prior_G6_comparison_invalid_for_DAPS4K=G6_F8_NOTE,
    DAPS4K_anneal_ODE="400x10",
    PC_DAPS_four_particle_bag=True,
    same_gpu_serial=True,no_GT_read=True,
    measurements=measurements,
    A15_XPI_matched_NFE_rule="WITHDRAWN_UNRESOLVED",
    no_matched_compute_claim=True)
 if valid:
  from statistics import mean
  summary["mean_native_sampling_seconds"]={
     method:mean(z["native_sampling_wall_seconds"] for z in measurements
                 if z["stage"]==method) for method in ("PC","DAPS4K")}
 write(out/"A17_CP3_CLEAN_TIMING_RECEIPT.json",summary)
 print("A17_TIMING_TERMINAL",json.dumps(
   dict(status=summary["status"],valid=valid,
        charged_h=cost/3600,mean_seconds=summary.get("mean_native_sampling_seconds")),
   sort_keys=True),flush=True)
 print("A17_TIMING_RECEIPT",out/"A17_CP3_CLEAN_TIMING_RECEIPT.json",flush=True)
 gate(valid,"A17_TIMING_COMPLETE_CLEAN_GATE")
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out")
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A17_CP3_CLEAN_TIMING",e,flush=True)
  sys.exit(8)
