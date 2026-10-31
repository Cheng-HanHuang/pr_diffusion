#!/usr/bin/env python3
"""A15 CP3 amendment: corrected native DAPS-4K receipt audit, paired CI and
fixed-K E1 comparisons. No GPU, no new image scoring, never repurposes M=2.

Read-only to original score and GT artifacts; writes a new fail-closed
evidence directory rather than silently editing A16's archived results.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_a12_g45_prepare_cpu import prep,DAPS4K,DAPS_HEAD
from scripts.track_r.xp_a16_cp3_joint_readout_cpu import (
 read_stage,record,aggregate,PIQ_STAMP,PROTOCOL_STAMP,
 NFE_STAMP,EXPECTED_PIQ_HEAD)
from scripts.track_r.xp_a15_cp3_protocol_cpu import verify

BOOT_SEED=20261031
BOOT_REPS=20000
METRICS=("PSNR_db","SSIM","LPIPS_VGG","GOOD")
G4_RUN_PER_NFE=4000
ORIGINAL_A16_JOINT=BASE/"xp_a16_cp3_joint_20261031T034143Z"/"XP_A16_CP3_JOINT_READOUT.json"

def receipt_audit(stage,data):
 # DAPS native immutable recorded solver settings, including precise
 # annealing/ODE/Langevin counts. A6 historical and A14 extension
 # methods both require identical published DAPS-4K configuration.
 gate(stage in ("G4","G5"),"A17_RECEIPT_STAGE")
 rec,S,C,R,strings,clips,runs=verify(
    BASE/("xp_a15_cp3_"+stage+"_"+PROTOCOL_STAMP),stage)
 rows=data["candidates"]
 entries=[]
 for (iid,rep),row in sorted(rows.items()):
  png=Path(row["reconstruction_path"])
  p=png.parent/"XP_T_DAPS4K_ROOT_COMPLETE.json"
  r=jread(p)
  cfg=r["solver_config"]
  gate(r["status"]=="XP_T_DAPS4K_ROOT_PASS"
       and r["method"]=="DAPS-4K" and r["native_head"]==DAPS_HEAD
       and r["uid"]==row["uid"]
       and r["measurement_sha256"]==row["measurement_sha256"]
       and sha(png)==r["reconstruction_png_sha256"]
       and cfg==DAPS4K
       and cfg["annealing_scheduler_config"]["num_steps"]==400
       and cfg["diffusion_scheduler_config"]["num_steps"]==10
       and cfg["mcmc_sampler_config"]["num_steps"]==100
       and r["native_sampling_wall_s"]>0
       and r["native_sampling_cuda_event_s"]>0,
       "A17_DAPS4K_RECEIPT_CONFIG_"+stage)
  entries.append(dict(image_id=iid,rep=rep,
      native_sampling_wall_s=float(r["native_sampling_wall_s"]),
      native_sampling_cuda_event_s=float(r["native_sampling_cuda_event_s"]),
      physical_gpu=r.get("gpu_physical"),
      root_receipt_sha256=sha(p)))
 gate(len(entries)==(1440 if stage=="G4" else 800),
      "A17_DAPS4K_FULL_NATIVE_RECEIPTS")
 return dict(n_roots=len(entries),annealing_transitions=400,
      diffusion_ODE_steps=10,langevin_steps_per_anneal=100,
      nominal_denoiser_forwards_per_run=G4_RUN_PER_NFE,
      note="Native config checked from each individual DAPS-4K root receipt",
      historical_wall_seconds_descriptive_only=True,
      timing_not_clean_GPU_certified=True)

def compute_same_particles(pc):
 by=pc["image_groups"];sc=pc["candidates"];cc=pc["clips"]
 table={}
 for (cohort,iid),rr in sorted(by.items()):
  native=[record(pc,(iid,k)) for k in range(4)]
  sel=min(range(4),key=lambda k:(float(sc[(iid,k)]["rho_signed"]),k))
  chosen=cc[(iid,sel)]["face_choice"]=="111"
  values={
    "PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN":{
      k:sum(t[k] for t in native)/4 for k in METRICS},
    "PC_E0":record(pc,(iid,sel)),
    "PC_E1_CLIP":record(pc,(iid,sel),flip=chosen)
  }
  table[iid]=dict(cohort=cohort,metrics=values)
 gate(len(table)==180,"A17_PC_180_SAME_PARTICLES")
 return table

def paired_bootstrap(table):
 ids=sorted(table)
 paired=np.array([[table[i]["metrics"]["PC_E1_CLIP"][m]
   -table[i]["metrics"]["PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN"][m]
   for m in METRICS] for i in ids],dtype=np.float64)
 gate(paired.shape==(180,len(METRICS)) and
      np.isfinite(paired).all(),"A17_BOOT_PAIRED_FINITE")
 rng=np.random.default_rng(BOOT_SEED)
 # Resample complete *image pairs*, never independent particles.
 sample=np.empty((BOOT_REPS,len(METRICS)),dtype=np.float64)
 for k in range(BOOT_REPS):
  take=rng.integers(0,180,size=180)
  sample[k,:]=paired[take,:].mean(axis=0)
 ci=np.quantile(sample,[.025,.975],axis=0,method="linear")
 return {
   "status":"A15_PC_E1_MINUS_NATIVE_PAIRED_BOOTSTRAP_95_PERCENTILE",
   "unit":"per_image_minus_mean_of_four_particle_metrics",
   "seed":BOOT_SEED,"bootstrap_replicates":BOOT_REPS,
   "resampling_unit":"complete_image",
   "n_image_pairs":180,
   "GT_used_only_for_evaluation":True,
   "metrics":{
      m:dict(delta_mean=float(paired[:,j].mean()),
         CI95_lower=float(ci[0,j]),CI95_upper=float(ci[1,j]),
         direction_higher_is_better=m!="LPIPS_VGG")
      for j,m in enumerate(METRICS)}
 }

def daps_fixed_e1(daps):
 # E0 minimum signed-rho within first K runs, then the frozen CLIP
 # decision for that same root. K=1,2, NOT the frozen M=2 stopping policy.
 grouped=defaultdict(lambda:defaultdict(list))
 for (cohort,iid),rr in sorted(daps["image_groups"].items()):
  for k in (1,2):
   chosen=min(range(k),key=lambda i:
    (float(daps["candidates"][(iid,i)]["rho_signed"]),i))
   clip=daps["clips"][(iid,chosen)]
   gate(clip["face_choice"] in ("000","111"),"A17_DAPS_FROZEN_CLIP")
   flip=clip["face_choice"]=="111"
   v=record(daps,(iid,chosen),flip=flip)
   gate(int(clip["face_chosen_good"])==int(v["GOOD"]),
        "A17_DAPS_CLIP_LABEL")
   for co in (cohort,"ALL"):
    grouped[co][k].append(v)
 return [dict(cohort=co,policy="DAPS_4K_FIXED_E1_K_"+str(k),
             mean_runs=k,nominal_NFE_per_image=k*G4_RUN_PER_NFE,
             matched_compute=False,**aggregate(g))
    for co,v in sorted(grouped.items()) for k,g in sorted(v.items())]

def pc_data_summary(table,bootstrap):
 rows=[]
 for cohort in ("B22","DEV80","ALL"):
  v=[s for s in table.values()
     if cohort=="ALL" or s["cohort"]==cohort]
  gate(len(v)==(100 if cohort=="B22" else 80 if cohort=="DEV80" else 180),
       "A17_COHORT_SIZE")
  for policy in ("PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN","PC_E0","PC_E1_CLIP"):
   vals=[z["metrics"][policy] for z in v]
   rows.append(dict(cohort=cohort,policy=policy,**aggregate(vals)))
 return rows

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A17_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(["git","-C",str(wt),*z],
                text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"A17_EXECUTOR_HEAD")
 prior=jread(ORIGINAL_A16_JOINT)
 gate(prior["status"]==
      "XP_A16_CP3_INDEPENDENT_JOINT_READOUT_PASS_UNMATCHED_NFE"
      and prior["CP3_RETURN_AUTHORIZED"] is False
      and len(prior["native_float_piq_spot_checks"])==3,
      "A17_PRIOR_AUDIT")
 daps=read_stage("G4");pc=read_stage("G3");dapsin=read_stage("G5")
 native_ff=receipt_audit("G4",daps)
 native_in=receipt_audit("G5",dapsin)
 bag_data=compute_same_particles(pc)
 boot=paired_bootstrap(bag_data)
 native=pc_data_summary(bag_data,boot)
 fixed=daps_fixed_e1(daps)
 point=jread(BASE/("xp_a15_cp3_protocol_"+PROTOCOL_STAMP)/
              "CP3_G4_PROTOCOL.json")
 gate(point["H1"] is True and
      point["weighted"]["runs_mean"]>=2 and
      point["weighted"]["runs_mean"]<=8,
      "A17_G4_FROZEN_POINT")
 p=point["weighted"]
 protocol_point=dict(status="FROZEN_M2_EPRIME8_POOLED",
    success=float(p["success_all"]),
    mean_runs=float(p["runs_mean"]),
    nominal_mean_NFE=float(p["runs_mean"])*G4_RUN_PER_NFE,
    selector="frozen_E1_self_mad3_M2_NMAX8",
    GT_selected=False,
    not_comparable_to_single_bag_at_equal_NFE=True,
    G4_protocol_json_sha256=sha(BASE/("xp_a15_cp3_protocol_"+PROTOCOL_STAMP)/
                       "CP3_G4_PROTOCOL.json"))
 ledger=jread(BASE/("xp_a16_cp3_nfe_"+NFE_STAMP)/
               "XP_A16_CP3_NFE_LEDGER.json")
 gate(ledger["observed_prune_events_sum"]==126 and
      ledger["config"]["nominal_per_bag_particle_equivalent_calls"]==1920,
      "A17_PC_NFE_BASE")
 pc_compute=dict(nominal_NFE_per_image=1920,
    logged_lower_bound_mean_NFE=float(ledger["observed_minimum_nfe_mean"]),
    number_of_logged_pruning_events=126,
    exact_NFE_per_bag_available=False,
    native_schedule="4 particles x (130 high-noise ODE 1 + 70 low-noise ODE 5)",
    unlogged_pruning_NFE_not_estimated=True)
 result=dict(
  status="A15_CP3_DESCRIPTIVE_EXTENSIONS_AUDITED_NOT_CHECKPOINT",
  expected_head=a.expected_head,
  planner_ref="817f78a88376cce3d67dacd6746b181490ab37ae",
  A15_XPI_matched_NFE_rule="WITHDRAWN_UNRESOLVED",
  no_M2_redefinition=True,no_matched_compute_claim=True,
  raw_A16_joint_sha256=sha(ORIGINAL_A16_JOINT),
  DAPS_4K_receipts=dict(G4=native_ff,G5=native_in),
  DAPS4K_nominal_NFE_per_run=4000,
  PC_DAPS_compute=pc_compute,
  same_particles= native,
  E1_minus_native_bootstrap=boot,
  DAPS_FFHQ_fixed_E1_K1_K2=fixed,
  frozen_Eprime8_protocol=protocol_point,
  clean_wall_clock_seconds="PENDING_G6_EXTENSION_AND_CLEAN_DAPS4K_TIMER",
  theory_inputs_release_required=True,
  no_GT_selection=True,no_GPU=True,no_CLIP_rerun=True,
  CP3_RETURN_AUTHORIZED_BY_PLANNER=True,
  CP3_PUBLICATION_READY=False,
  DAPS400x5_legacy_G6_not_substitutable=True)
 if a.preflight_only:
  print("A17_CP3_DESCRIPTIVE_PREFLIGHT_PASS",
        "DAPS_ROOTS",native_ff["n_roots"]+native_in["n_roots"],
        "PC_IMAGES",len(bag_data),flush=True)
  for m,z in boot["metrics"].items():
   print("A17_E1_MINUS_NATIVE",m,round(z["delta_mean"],6),
         round(z["CI95_lower"],6),round(z["CI95_upper"],6),flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a17_cp3_a15_")
      and not out.exists() and not out.is_symlink(),
      "A17_NEW_OUTPUT")
 out.mkdir(mode=0o700)
 write(out/"XP_A17_CP3_A15_DESCRIPTIVE.json",result)
 print("A17_CP3_A15_DESCRIPTIVE_PASS",out,flush=True)
 print("A17_DAPS4K_ANNEAL_400_ODE_10_LANGEVIN_100_NFE_4000",flush=True)
 for m,z in boot["metrics"].items():
  print("A17_E1_MINUS_NATIVE",m,round(z["delta_mean"],6),
        "[",round(z["CI95_lower"],6),round(z["CI95_upper"],6),"]",flush=True)
 for r in fixed:
  if r["cohort"]=="ALL":
   print("A17_FIXED_E1",r["policy"],"GOOD",round(r["GOOD"],6),
         "PSNR",round(r["PSNR_db"],6),
         "nominal_NFE",r["nominal_NFE_per_image"],flush=True)
 print("A17_FROZEN_EPRIME8",protocol_point["success"],
       protocol_point["mean_runs"],
       protocol_point["nominal_mean_NFE"],flush=True)
 print("A17_WALL_CLOCK_AND_RETURN_PUBLICATION_PENDING",flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out")
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A17_CP3_A15",e,flush=True)
  sys.exit(8)
