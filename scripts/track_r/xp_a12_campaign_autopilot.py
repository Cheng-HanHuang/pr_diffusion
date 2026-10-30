#!/usr/bin/env python3
"""A13 automatic stage queue after completed/partial G1b: G2, G3, G4 and G5.

Single-controller shared 400 project / 150 A12 campaign GPU budget.
New generation has no GT or scoring. Never reads D4 run >=16 outcomes.
G3 requires a separately certified native PC-DAPS signed-y adapter;
when absent/skipped it publishes a blocked receipt. G1b is independently admitted first by xp_a13_g1b_autopilot.py
and its terminal receipt is mandatory here.
G6/G7 use separate stage-specific readiness gates and controllers.
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
G1B_EXPECTED_STATUS="XP_A13_G1B_3200_GENERATION_COMPLETE"
DEADLINE={"G2":(2026,11,3,12),"G3":(2026,11,3,12),"G4":(2026,11,4,12),"G5":(2026,11,4,12),"G1b":(2026,11,4,12)}
CAP={"G2":25.,"G3":45.,"G4":30.,"G5":40.,"G1b":40.}
N={"G2":2880,"G3":180,"G4":720,"G5":400,"G1b":3200}
RESERVE_MIB={"G2":8192,"G3":24576,"G4":12288,"G5":12288,"G1b":8192}
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
# A14 import only the immutable GPU-hour costs of the terminated G2 smoke.
# The old native results stay in their original provenance namespace.
OLD_G2_SMOKE=BASE/"xp_a12_campaign_prod_20261030T002042Z"
OLD_G2_HEAD="613034e10552d1288179aa89bca6e8d51c337639"
def imported_g2_smoke():
 old=OLD_G2_SMOKE
 gate(old.is_dir() and not old.is_symlink(),"A14_OLD_SMOKE_PATH")
 state=jread(old/"XP_A12_CAMPAIGN_STATE.json")
 gate(state["head"]==OLD_G2_HEAD and state["status"]=="G2_COMPLETE"
      and not state["active"] and len(state["done"])==2,
      "A14_OLD_SMOKE_TERMINAL")
 paths=sorted((old/"cost").glob("*.json"))
 gate(len(paths)==2,"A14_OLD_SMOKE_TWO_COSTS")
 costs=[jread(p) for p in paths]
 gate(all(v["stage"]=="G2" and v["job"]["sig"]=="0.02"
       and v["job"]["kind"]=="SMOKE" and v["rc"]==0
       and v["charged_gpu_h"]>0 for v in costs),
      "A14_OLD_SMOKE_VALID_COSTS")
 gate(sorted(v["job"]["start"] for v in costs)==[0,8],
      "A14_OLD_SMOKE_TWO_IMAGES")
 seconds=sum(float(v["elapsed_seconds"]) for v in costs)
 gate(0<seconds<1800 and
      abs(seconds-state["stage_seconds"]["G2"])<0.1,
      "A14_OLD_SMOKE_COST_RECONCILED")
 paths2=sorted((old/"G2"/"sigma_0.02").glob(
     "chunk_*/root_*/R0_FAST_ROOT_COMPLETE.json"))
 gate(len(paths2)==2 and all(
      jread(p)["status"]=="R0_FAST_ROOT_PASS" for p in paths2),
      "A14_OLD_SMOKE_TWO_NATIVE_RECEIPTS")
 return dict(status="XP_A14_PRIOR_G2_SMOKE_VERIFIED_CHARGED_ONCE",
   old_run=str(old),old_head=OLD_G2_HEAD,seconds=seconds,
   gpu_h=seconds/3600,receipts={str(p):sha(p) for p in paths+paths2},
   original_state_sha256=sha(old/"XP_A12_CAMPAIGN_STATE.json"))

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
 g1b_file=Path(a.g1b_final)
 gate(g1b_file.name=="XP_A13_G1B_FINAL_STATUS.json" and
      g1b_file.parent.parent==BASE and
      g1b_file.parent.name.startswith("xp_a13_g1b_prod_") and
      g1b_file.is_file() and not g1b_file.is_symlink(),
      "A13_G1B_RECEIPT_PATH")
 b=jread(g1b_file)
 gate(b["status"]==G1B_EXPECTED_STATUS or
      str(b["status"]).startswith("PARTIAL_"),"A13_G1B_TERMINAL_STATUS")
 gate(b["g1a_final_sha256"]==sha(G1A)
      if "g1a_final_sha256" in b else
      b["G1a_pixel_exact_replay"] is True,
      "A13_G1B_LINEAGE")
 gate(b["completed"]%32==0 and
      0<b["completed"]<=3200
      and 0<=b["stage_gpu_h"]<=40.05
      and b["source_head"] and
      b["stage_cap_gpu_h"]==40.
      and abs(b["project_total_gpu_h"]-
       (L["project_total_before_A12_gpu_h"]+g["stage_gpu_h"]+
         b["stage_gpu_h"]))<1e-4
      and abs(b["A12_total_gpu_h"]-
         (g["stage_gpu_h"]+b["stage_gpu_h"]))<1e-4,
      "A13_G1B_ACTUAL_PROJECT_CAMPAIGN_CHARGE")
 start=b["project_total_gpu_h"]
 campaign=b["A12_total_gpu_h"]
 gate(start<=PROJECT_CAP and campaign<=CAMPAIGN_CAP,
      "A13_SHARED_HEADROOM")
 P={}; roots={}
 for stage,p in (("G2",a.g2_prep),("G45",a.g45_prep))+(
      (("G3",a.g3_prep),) if a.g3_prep else ())+(
      (("G1b",a.g1b_prep),) if a.g1b_prep else ()):

  q=Path(p)
  gate(q.parent==BASE and q.is_dir() and not q.is_symlink()
       and q.name.startswith("xp_a12_"+stage.lower()+"_prepare_"),
       "PREP_PATH_"+stage)
  fname={"G2":"XP_A12_G2_PREP_RECEIPT.json",
         "G3":"XP_A12_G3_PREP_RECEIPT.json",
         "G45":"XP_A12_G45_PREP_RECEIPT.json",
         "G1b":"XP_A12_G1B_PREP_RECEIPT.json"}[stage]
  x=jread(q/fname)
  gate(x["head"]==a.expected_head and x["no_GPU"] is True
       and x["status"]=={
         "G2":"XP_A12_G2_INPUTS_FROZEN_CPU_PASS",
         "G3":"XP_A12_G3_CPU_PREP_PASS",
         "G45":"XP_A12_G45_INPUTS_FROZEN_CPU_PASS",
         "G1b":"XP_A12_G1B_CONDITIONAL_PREPARED_CPU_PASS"}[stage],
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
 if "G3" in P:
  p=roots["G3"]/"XP_A12_G3_180_BAGS.json"
  gate(sha(p)==P["G3"]["manifest_sha256"]
       and len(jread(p)["rows"])==180,"G3_MANIFEST")
  jobs["G3"]=([dict(stage="G3",kind="SMOKE",index=i)
               for i in range(2)],
              [dict(stage="G3",kind="PROD",index=i)
               for i in range(2,180)])
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

def a14_g2_jobs(P):
 """A14 preserves paired sigma panels and complete per-image run blocks."""
 image_panels={}
 for sig in ("0.02","0.10"):
  p=P["G2"]["per_sigma"][sig]["path"]
  rows=jread(p)["rows"]
  gate(len(rows)==1440,"A14_G2_ROWS_"+sig)
  for i in range(180):
   chunk=rows[8*i:8*i+8]
   gate([x["rep"] for x in chunk]==list(range(8))
        and len({x["image_id"] for x in chunk})==1,
        "A14_G2_ORDER_"+sig)
  image_panels[sig]=[rows[i*8]["image_id"] for i in range(180)]
 gate(image_panels["0.02"]==image_panels["0.10"]
      and len(set(image_panels["0.02"]))==180,"A14_PAIRED_IMAGES")
 names=image_panels["0.02"]
 gate(sum(x.startswith("S1_b22_") for x in names)==100
      and sum(x.startswith("S2_b24_dev80_") for x in names)==80,
      "A14_EXPOSED_COHORT_COUNTS")
 # Run the exact production worker on one whole image per sigma.
 pilot=[dict(stage="G2",kind="PILOT",sig=sig,start=0,count=8)
        for sig in ("0.02","0.10")]
 # Pilot completes all runs for the first image, so skip it thereafter.
 full=[dict(stage="G2",kind="PROD",sig=sig,start=8*i,count=8)
       for i in range(1,180) for sig in ("0.02","0.10")]
 # All images, 0..3 first; then 4..7 in cohort A and B.
 # Group by complete image, and interleave the two sigma settings.
 base=[dict(stage="G2",kind="BASE",sig=sig,start=8*i,count=4)
       for i in range(1,180) for sig in ("0.02","0.10")]
 tail=[dict(stage="G2",kind="TAIL",sig=sig,start=8*i+4,count=4)
       for i in range(1,180) for sig in ("0.02","0.10")]
 gate(len(pilot)==2 and len(full)==358 and len(base)==358
      and len(tail)==358,"A14_G2_JOB_GRID")
 return pilot,full,base,tail

def a14_g2_pilot_report(run,P,state,prior):
 samples={}
 for sig in ("0.02","0.10"):
  rows=jread(P["G2"]["per_sigma"][sig]["path"])["rows"]
  dest=run/"G2"/("sigma_"+sig)/"chunk_00000_008"
  times=[]
  for row in rows[:8]:
   rc=jread(root_folder(dest,row)/"R0_FAST_ROOT_COMPLETE.json")
   gate(rc["status"]=="R0_FAST_ROOT_PASS","A14_PILOT_NATIVE")
   seconds=float(rc["native_sample_wall_seconds"])
   gate(math.isfinite(seconds) and seconds>0,"A14_PILOT_POSITIVE_SAMPLE")
   times.append(seconds)
  samples[sig]=times
 launches=[]
 with (run/"XP_A12_CAMPAIGN_EVENTS.jsonl").open() as f:
  for line in f:
   d=json.loads(line)
   if d["event"]=="TASK_LAUNCHED" and d["job"]["stage"]=="G2":
    launches.append(d)
 gate(len(launches)==2,"A14_PILOT_TWO_LAUNCHES")
 remaining_per_sigma=1440-8
 projected_s=float(state["stage_seconds"]["G2"])+sum(
   1.25*max(samples[sig])*remaining_per_sigma
   for sig in ("0.02","0.10"))
 result=dict(status="XP_A14_G2_PRODUCTION_MODE_PILOT_PASS",
   n_images_per_sigma=1,n_runs_per_image=8,
   source_head=state["head"],native_sample_wall_seconds=samples,
   gpu_launch_telemetry=launches,
   previous_smoke_cost_gpu_h=prior["gpu_h"],
   billed_before_production_gpu_h=state["stage_seconds"]["G2"]/3600,
   full_projection_gpu_h=projected_s/3600,
   stage_cap_gpu_h=CAP["G2"],
   full_panel_fits=projected_s/3600<=CAP["G2"],
   no_GT_scoring=True,co_tenant_fallback_permitted=True)
 write(run/"XP_A14_G2_PILOT_REPORT.json",result)
 return result

def a14_g2_coverage(state):
 seen={sig:{} for sig in ("0.02","0.10")}
 for job in state.get("verified_jobs",{}).get("G2",[]):
  r=seen[job["sig"]]
  for index in range(job["start"],job["start"]+job["count"]):
   r.setdefault(index//8,set()).add(index%8)
 counts={}
 for sig,images in seen.items():
  counts[sig]=dict(base4=sum(all(k in v for k in range(4))
                              for v in images.values()),
    full8=sum(len(v)==8 for v in images.values()))
 paired=sum(len(seen["0.02"].get(i,set()))==8
            and len(seen["0.10"].get(i,set()))==8
            for i in range(180))
 return dict(per_sigma=counts,paired_full8_images=paired,
    H_G2_evaluable=all(v["full8"]>=60 for v in counts.values()))

def make_command(a,run,task,gpu,roots,P,attempt):
 stage=task["stage"];kind=task["kind"]
 if stage=="G1b":
  rec=P["G1b"]
  manifest=roots["G1b"]/"XP_A12_G1B_3200_TASKS.json"
  out=run/"G1b"/f"chunk_{task['start']:05d}_{task['count']:03d}"
  cmd=[str(ROOT_FAST_PY),str(Path(__file__).with_name("xp_a12_campaign_fast_gpu.py")),
    "--expected-head",a.expected_head,"--roots",str(manifest),
    "--roots-sha",rec["manifest_sha256"],
    "--roots-source-head",rec["head"],"--out",str(out),
    "--start",str(task["start"]),"--count",str(task["count"]),
    "--physical-gpu",str(gpu),"--dataset","imagenet",
    "--oversample","2.0"]
  return cmd,out,rec["manifest_sha256"],str(ROOT_FAST_PY)
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
 if stage=="G3":
  mf=jread(roots["G3"]/"XP_A12_G3_180_BAGS.json")
  row=mf["rows"][task["index"]]
  dest=run/"roots"/("bag_"+hashlib.sha256(
      row["uid"].encode()).hexdigest()[:24])
  env=ROOT/"conda-envs/daps/bin/python"
  cmd=[str(env),str(Path(__file__).with_name(
      "xp_a12_g3_pcdaps_gpu.py")),
       "--expected-head",a.expected_head,
       "--roots-source-head",P["G3"]["head"],
       "--run",str(run),"--prep",str(roots["G3"]),
       "--index",str(task["index"]),"--gpu",str(gpu)]
  return cmd,dest,P["G3"]["manifest_sha256"],str(env)
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
 if task["stage"] in ("G2","G1b"):
  rec=P["G2"]["per_sigma"][task["sig"]] if task["stage"]=="G2" else P["G1b"]
  mf=jread(rec["path"]) if task["stage"]=="G2" else jread(
       roots["G1b"]/"XP_A12_G1B_3200_TASKS.json")
  chunk=jread(dest/"R0_5_CHUNK_COMPLETE.json")
  gate(chunk["status"]=="R0_5_CHUNK_PASS"
       and chunk["head"]==a.expected_head
       and chunk["new"]==task["count"]
       and chunk["start"]==task["start"]
       and chunk["count"]==task["count"]
       and chunk["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
       "FAST_NATIVE_CHUNK")
  for row in mf["rows"][task["start"]:task["start"]+task["count"]]:
   r=validate_complete(root_folder(dest,row),row,
         a.expected_head,source_sha)
   gate(r["gt_access"] is False,"A12_FAST_NO_GT")
  return sha(dest/"R0_5_CHUNK_COMPLETE.json")
 stage=task["stage"];phase=task["kind"]
 if stage=="G3":
  mf=jread(roots["G3"]/"XP_A12_G3_180_BAGS.json")
  row=mf["rows"][task["index"]]
  receipt=dest/"XP_A12_G3_BAG_COMPLETE.json"
  d=jread(receipt)
  gate(d["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS"
       and d["uid"]==row["uid"]
       and d["runner_head"]==a.expected_head
       and d["manifest_sha256"]==source_sha
       and d["bag_size"]==4 and len(d["particle_pngs"])==4
       and d["no_ground_truth_access"] is True
       and sha(dest/"particles.pt")==d["particles_pt_sha256"]
       and all(sha(x["path"])==x["sha256"]
               for x in d["particle_pngs"]),
       "G3_PC_NATIVE_PARTICLE_BAG")
  return sha(receipt)
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
  elif stage=="G3":
   mf=jread(roots["G3"]/"XP_A12_G3_180_BAGS.json")
   row=mf["rows"][job["index"]]
   dest=run/"roots"/("bag_"+hashlib.sha256(
       row["uid"].encode()).hexdigest()[:24])
   rc=jread(dest/"XP_A12_G3_BAG_COMPLETE.json")
   gate(rc["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS",
        "SMOKE_PC_DAPS_PASS")
   t=float(rc["native_sampling_wall_seconds"])
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
 reduced=False
 if stage=="G3" and projected/3600>CAP[stage]:
  projected=stage_seconds+1.25*max(times)*(100-2)
  reduced=True
 gate(projected/3600<=CAP[stage],
      "STAGE_SMOKE_PROJECTS_OVER_CEILING_"+stage)
 return dict(smoke_native_seconds=times,projected_gpu_h=projected/3600,
    comparison_ceiling_gpu_h=CAP[stage],cohort_A_only=reduced,
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
   name=f"{stage}_{job.get('sig','')}_{job['kind']}_{idx}_A{v['attempt']}"
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
   state.setdefault("verified_jobs",{}).setdefault(stage,[]).append(job)
   emit(run,"ROOT_VERIFIED",job=job,digest=digest)
   save("RUN_"+stage)
 def loop():
  nonlocal draining_reason
  save("RUN_"+stage)
  while len(state["done"])<len(tasks):
   cards=collect(own_roots=[v["proc"].pid for j,v in active.values()])
   # A10.1 contract: stop only our own worker when observed total
   # device VRAM crosses 80%; do not touch foreign co-tenants.
   for _,v in active.values():
    card=next((c for c in cards if c.index==v["gpu"]),None)
    if (card is not None and v["proc"].poll() is None and
        card.used_mib>0.80*card.total_mib):
     emit(run,"VRAM80_OWN_WORKER_STOP",gpu=v["gpu"],
          pid=v["proc"].pid,used_mib=card.used_mib)
     try:os.killpg(v["proc"].pid,signal.SIGTERM)
     except ProcessLookupError:pass
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
    preferred=tuple(c.index for c in candidates if not c.foreign_pids and c.used_mib<=2048) if stage=="G2" else ()
    gpu,_=choose_gpu(candidates,RESERVE_MIB[stage],
        safety_margin_mib=2048,fraction=.80,clean_only=False,preferred=preferred)
    if gpu is None:break
    cmd,dest,digest,exe=make_command(a,run,job,gpu,roots,P,0)
    gate(not dest.exists(),"NEW_TASK_OUTPUT_ALREADY_EXISTS")
    key=f"{stage}_{job.get('sig','')}_{job['kind']}_{idx}"
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
    card=next((c for c in cards if c.index==gpu),None)
    emit(run,"TASK_LAUNCHED",stage=stage,job=job,gpu=gpu,pid=proc.pid,
      foreign_pids=list(card.foreign_pids) if card else [],
      gpu_memory_used_mib=card.used_mib if card else None,
      gpu_utilization_pct=card.utilization_pct if card else None)
    save("RUN_"+stage)
   if not active and remaining and (current_spend()+600>cap or
      campaign0*3600+total_spend()+600>CAMPAIGN_CAP*3600 or
      project0*3600+total_spend()+600>PROJECT_CAP*3600):
    save("PARTIAL_"+stage+"_RESERVATION_CEILING");return False
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
   name=f"{stage}_{job.get('sig','')}_{job['kind']}_{idx}_A_EXCEPTION"
   write(run/"cost"/(name+".json"),dict(status="A12_FAILED_ATTEMPT_CHARGED",
      job=job,seconds=elapsed,charged_gpu_h=elapsed/3600,gpu=v["gpu"],
      pid=v["proc"].pid,returncode=v["proc"].returncode))
   del active[idx]
  save("STOP_"+stage+"_EXCEPTION")
  raise
def main(a):
 wt,roots,P,ledger,g1a,project0,campaign0=prepar(a)
 prior=imported_g2_smoke() if a.a14 else None
 jobs=task_set(roots,P)
 if a.preflight_only:
  print("XP_A13_G2G5_PREPARE_AND_DUAL_LEDGER_PREFLIGHT_PASS",flush=True)
  print("A13_PRIOR_PROJECT_GPU_H",project0,flush=True)
  print("A13_PRIOR_A12_GPU_H",campaign0,flush=True)
  print("A13_READY_STAGES",sorted(jobs),flush=True)
  print("G3_MISSING_PRODUCER_SKIP",bool("G3" not in jobs),flush=True)
  if prior:print("A14_PRIOR_G2_SMOKE_CHARGED_GPU_H",prior["gpu_h"],flush=True)
  return
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_campaign_prod_")
      and not run.exists(),"NEW_CAMPAIGN_RUN")
 with (BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run.mkdir(mode=0o700)
  for p in ("logs","cost","G2","G4","G5","G1b","roots","pycache"):
   (run/p).mkdir()
  for sig in ("0.02","0.10"):(run/"G2"/("sigma_"+sig)).mkdir()
  for stage in ("G4","G5"):
   for phase in ("REPLAY","PROD"):(run/stage/phase/"gpu").mkdir(parents=True)
  state=dict(status="ADMITTED",head=a.expected_head,
    project_initial_gpu_h=project0,campaign_initial_gpu_h=campaign0,
    A14_head="f31901c808954f2bee724ab8a3834e2b65ef25c9",
    G1b_final_receipt_sha256=sha(a.g1b_final),
    parent_ledger_sha256=sha(a.ledger),G1a_final_sha256=sha(G1A),
    stage_seconds={x:(prior["seconds"] if prior and x=="G2" else 0.) for x in CAP},
    done={},active=[],verified_jobs={},
    G3_status="NOT_ASSESSED",G1b_status="NOT_AUTHORIZED_TO_AUTO_LAUNCH",
    G6_status="CONDITIONAL_NO_WORK_UNTIL_CLEAN_AUDIT",
    G7_status="NOT_YET_ADMITTED",
    GT_not_read=True,D4_outcomes_not_opened=True)
  if prior:
   write(run/"XP_A14_IMPORTED_G2_SMOKE_COST.json",prior)
   state["A14_previous_smoke_receipt_sha256"]=sha(run/"XP_A14_IMPORTED_G2_SMOKE_COST.json")
  if a.a14:
   state["G3_status"]=("A14_CONDITIONAL_SOURCE_READY_SEPARATE_CERTIFICATION"
      if "G3" in jobs else
      "A14_SOURCE_ABSENT_PENDING_OCT31_TIER_C_GATE")
   write(run/"XP_A14_G3_CONDITIONAL_STATUS.json",dict(
      status=state["G3_status"],deadline_ET="2026-11-04T12:00:00-05:00",
      Tier_C_if_absent_ET="2026-10-31T12:00:00-04:00",
      source_head=a.expected_head,not_launched_by_this_A14_controller=True))
  write(run/"XP_A12_CAMPAIGN_STATE.json",state)
  for stage in (("G4","G5","G2") if a.a14 else ("G2","G3","G4","G5")):
   if stage=="G3" and "G3" not in jobs:
    state["G3_status"]="CONDITIONAL_SOURCE_ABSENT_NOT_LAUNCHED"
    emit(run,"G3_CODE_ABSENT_SKIP_PER_A12")
    write(run/"XP_A12_G3_CODE_ABSENT_SKIP.json",dict(
      status="XP_A12_G3_CONDITIONAL_CODE_ABSENT_NO_GPU",
      reason="PCDAPS_NATIVE_SOURCE_NOT_CERTIFIED_OR_NOT_PREPARED",
      do_not_claim_180_bags=True,
      source="https://github.com/voilalab/PC-DAPS",
      pinned_head="59b2f046"))
    continue
   if stage=="G2" and a.a14:
    pilot,full,base,tail=a14_g2_jobs(P)
    state["done"]={}
    gate(run_queue(a,run,"G2",pilot,roots,P,state,project0,campaign0),
         "A14_TWO_PRODUCTION_PILOTS_MUST_PASS")
    report=a14_g2_pilot_report(run,P,state,prior)
    emit(run,"A14_G2_PILOT_COMPLETE",projection_gpu_h=report["full_projection_gpu_h"],
         full_fit=report["full_panel_fits"])
    segments=(("FULL",full),) if report["full_panel_fits"] else (
      ("BASE_BOTH_SIGMAS",base),("TAIL_COHORT_A_THEN_B",tail))
    complete=True
    for tag,subset in segments:
     state["done"]={}
     if not run_queue(a,run,"G2",subset,roots,P,state,project0,campaign0):
      emit(run,"A14_G2_PARTIAL_BUDGET",segment=tag)
      complete=False
      break
    coverage=a14_g2_coverage(state)
    result=dict(status=("XP_A14_G2_ALL_2880_COMPLETE" if complete else
                        "XP_A14_G2_IMAGE_COMPLETE_PARTIAL"),
       source_head=a.expected_head,completed_chunks=len(state["verified_jobs"]["G2"]),
       old_smoke_billed_once=True,charged_gpu_h=state["stage_seconds"]["G2"]/3600,
       stage_cap_gpu_h=CAP["G2"],coverage=coverage,
       H_G2_evaluable=coverage["H_G2_evaluable"],no_GT_scoring=True)
    write(run/"XP_A14_G2_PROGRESS.json",result)
    emit(run,"A14_G2_STAGE_TERMINAL",coverage=coverage,complete=complete)
    if not complete:
     write(run/"XP_A13_G2_PARTIAL_STATUS.json",dict(
        status="XP_A13_STAGE_PARTIAL_CONTINUE",stage="G2",
        charged_gpu_h=state["stage_seconds"]["G2"]/3600,
        stage_ceiling_gpu_h=CAP["G2"],coverage=coverage))
    continue
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
      try:
       estimate=smoke_projection(run,stage,smoke,P,roots,
              state["stage_seconds"][stage])
      except RuntimeError as exc:
       if "STAGE_SMOKE_PROJECTS_OVER_CEILING_" not in str(exc):
        raise
       emit(run,"STAGE_PROJECTED_OVER_CAP_PARTIAL_PER_A13",
            stage=stage,reason=str(exc))
       estimate=dict(projection_pass=False,
          comparison_ceiling_gpu_h=CAP[stage],
          reason=str(exc),partial_permitted_by_A13=True)
      if stage=="G3" and estimate.get("cohort_A_only"):
       production=[x for x in production if x["index"]<100]
       state["G3_status"]="COHORT_A_REDUCTION_AUTHORIZED_BY_A12"
      emit(run,"STAGE_SMOKE_OR_REPLAY_PASS",stage=stage,
           completed=len(done),projection=estimate)
    if not success:break
   if not success:
    emit(run,"STAGE_PARTIAL_AT_CEILING_CONTINUE_PRIORITY_QUEUE",stage=stage)
    write(run/f"XP_A13_{stage}_PARTIAL_STATUS.json",dict(
       status="XP_A13_STAGE_PARTIAL_CONTINUE",
       stage=stage,charged_gpu_h=state["stage_seconds"][stage]/3600,
       stage_ceiling_gpu_h=CAP[stage],
       source_head=a.expected_head,failed_attempts_charged=True,
       no_ground_truth_loaded=True))
    continue
   completed_roots=(100 if stage=="G3" and
                  state["G3_status"]=="COHORT_A_REDUCTION_AUTHORIZED_BY_A12"
                  else N[stage])
   emit(run,"STAGE_GENERATION_COMPLETE",stage=stage,
        n_roots=completed_roots,gpu_h=state["stage_seconds"][stage]/3600)
   write(run/f"XP_A12_{stage}_FINAL_STATUS.json",dict(
     status=(f"XP_A12_{stage}_COHORT_A_REDUCTION_COMPLETE"
             if completed_roots!=N[stage] else
             f"XP_A12_{stage}_GENERATION_COMPLETE"),
     n_roots=completed_roots,requested_roots=N[stage],
     reduction_authorized_by_A12=(completed_roots!=N[stage]),
     failed_attempts=0,
     charged_gpu_h=state["stage_seconds"][stage]/3600,
     stage_cap_gpu_h=CAP[stage],head=a.expected_head,
     outcomes_GT_not_read=True))
  state["G1b_status"]="BILLED_BY_PRIOR_A13_G1B_RECEIPT"
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
 for x in ("expected-head","ledger","g1b-final","g2-prep","g45-prep","run"):
  p.add_argument("--"+x,required=True)
 p.add_argument("--g3-prep",default=None)
 p.add_argument("--a14",action="store_true")
 # Optional legacy G1b preparation is not passed for A13: G1b has
 # already completed and is authenticated by --g1b-final.
 p.add_argument("--g1b-prep",default=None)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_CAMPAIGN_AUTOPILOT",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
