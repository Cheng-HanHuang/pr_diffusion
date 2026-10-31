#!/usr/bin/env python3
"""Independent A15 CP3 offline scientific consistency / protocol audit.

Read-only to scored stages. Rebuilds labels, tau, CLIP decisions and
all saved frozen-protocol experiments without GT or GPU access.
This is an analysis audit, NOT an authorization to publish CP3.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import (
    sha, safe, csvload, jread, write, PROTO)
from scripts.track_r.xp_a15_cp3_protocol_cpu import STAGES,COHORTS,verify
from scripts.track_r.xp_a15_cp3_score_cpu import G3_MANIFEST_SHA

STAGE_PATHS={
 "G2_0.02":"xp_a15_cp3_G2_0p02_",
 "G2_0.10":"xp_a15_cp3_G2_0p10_",
 "G4":"xp_a15_cp3_G4_",
 "G5":"xp_a15_cp3_G5_",
 "G3":"xp_a15_cp3_G3_",
}
def assert_close(a,b,name,tol=1e-10):
 gate(math.isfinite(float(a)) and math.isfinite(float(b))
      and abs(float(a)-float(b))<=tol,name)
def unique_stage_dir(stage,stamp):
 p=BASE/(STAGE_PATHS[stage]+stamp)
 gate(p.is_dir() and not p.is_symlink(),"CP3_AUDIT_STAGE_EXISTS_"+stage)
 return p
def independent_stage(stage,stage_dir):
 rec,S,C,R,strings,clips,runs=verify(stage_dir,stage)
 n=int(rec["n_runs_per_image"])
 by_s={r["image_id"]:r for r in strings}
 by_c={(r["image_id"],int(r["run"])):r for r in clips}
 by_r=defaultdict(dict)
 gate(len(by_s)==rec["n_images"] and
      len(by_c)==rec["n_candidates"],"CP3_AUDIT_KEYS_"+stage)
 counts=Counter()
 for row in runs:
  iid=row["image_id"]; i=int(row["run"])
  gate(iid in by_s and i not in by_r[iid] and 0<=i<n,
       "CP3_AUDIT_UNIQUE_RUN_"+stage)
  by_r[iid][i]=row
  s=by_s[iid]
  label=s["labels_by_run"].split(";")[i]
  rho=float(s["rho_signed_by_run"].split(";")[i])
  assert_close(rho,row["rho_signed"],"CP3_AUDIT_RHO_JOIN",1e-10)
  gate(label==row["label"]=="G" or label==row["label"],
       "CP3_AUDIT_LABEL_JOIN")
  raw=float(row["psnr_raw"]);best=float(row["psnr_orbit_best"])
  flip=float(row["psnr_flip"]); threshold=20 if stage=="G5" else 25
  tau=float(s["tau_signed_good_q95"])
  gate(all(math.isfinite(z) for z in (raw,best,flip,rho,tau))
       and rho>=0 and tau>0,"CP3_AUDIT_FINITE")
  if raw>=threshold: want="G"
  elif best>=threshold:
   # 'O_g' requires 111 to be an orbit-maximizer; these must be
   # consistent to the stored output precision, otherwise O_p.
   if label=="O_g":
    gate(flip>=threshold and abs(best-flip)<1e-9,
         "CP3_AUDIT_ORBIT_G")
    want="O_g"
   else:
    want="O_p"
  else:want="C" if rho<=tau else "I"
  gate(label==want,"CP3_AUDIT_LABEL_THRESHOLD_"+stage)
  gate(int(row["flip_good"])==int(flip>=threshold),
       "CP3_AUDIT_FLIP_GT")
  c=by_c[iid,i]
  mode=rec["clip_mode"]
  margin=float(c[mode+"_margin"])
  gate(math.isfinite(margin),"CP3_AUDIT_CLIP_MARGIN")
  choice="111" if margin>0 else "000"
  good=int(flip>=threshold) if choice=="111" else int(raw>=threshold)
  gate(c[mode+"_choice"]==choice and
       int(c[mode+"_chosen_good"])==good and c["label"]==label and
       c["cohort"]==row["cohort"],"CP3_AUDIT_CLIP_DECISION_"+stage)
  counts[label]+=1
 gate(set(by_s)==set(by_r) and all(set(x)==set(range(n)) for x in by_r.values()),
      "CP3_AUDIT_COMPLETE_NRUN_"+stage)
 # Independently recalculate the pre-registered family-level tau from
 # first 4 native runs per image (whether n=4 or n=8).
 good_rho=[
  float(row["rho_signed"]) for x in by_r.values() for i,row in x.items()
  if i<4 and float(row["psnr_raw"])>=(20 if stage=="G5" else 25)
 ]
 gate(bool(good_rho),"CP3_AUDIT_TAU_NO_GOOD")
 independently_calibrated_tau=float(np.quantile(good_rho,.95,method="linear"))
 for tau in rec["tau_by_cohort"].values():
  assert_close(tau,independently_calibrated_tau,"CP3_AUDIT_GT_TAU_"+stage)
 for s in strings:
  gate(set(s["labels_by_run"].split(";"))<=set(("G","O_g","O_p","C","I"))
       and sum(int(s["n_"+k]) for k in ("G","O_g","O_p","C","I"))==n,
       "CP3_AUDIT_LABEL_COUNTS_"+stage)
 gate(dict(counts)==rec["labels"],"CP3_AUDIT_GLOBAL_LABELS_"+stage)
 return dict(stage=stage,n_images=len(by_s),n_candidates=len(runs),
    n_per_image=n,labels=dict(counts),
    tau_gt_q95=independently_calibrated_tau,
    rho_join_pass=True,clip_decisions_pass=True,
    native_scoring_files_hash_verified=True)
def rerun_protocol(out,original_script,existing_file,stage,cohort,
                   stage_dir,mode):
 # Protocol version/seed/range/threshold are replayed from original
 # record. We require exact equality of all quantitative output fields,
 # not simply the high-level hypothesis booleans.
 prior=jread(existing_file)
 gate(prior["protocol_version"]=="v1" and
      prior["N_MAX"]==8 and prior["M"]==2 and
      prior["seed"]==20261029 and prior["R"]==4000,
      "CP3_AUDIT_FROZEN_PRIMARY_PARAMETERS")
 command=[sys.executable,"-I",str(original_script),
    "--strings",str(stage_dir/"CP3_A9_STRINGS.csv"),
    "--cohort",cohort,"--m","2","--nmax","8",
    "--R","4000","--seed","20261029"]
 if prior.get("tau_rule")=="self_mad3":
  command+=["--tau-rule","self_mad3"]
 else:
  gate(prior.get("tau_rule")=="frozen","CP3_AUDIT_TAU_RULE")
  command+=["--tau",repr(float(prior["tau"]))]
 if mode:
  p=out/(stage+"_"+cohort+"_CLIP.csv")
  # These files are already present in the provisional protocol output
  # and are hashed in its gate; they do not contain GT image content.
  gate(p.is_file(),"CP3_AUDIT_CLIP_SLICE")
  command+=["--clip",str(p),
    "--clip-good-col",mode+"_chosen_good",
    "--clip-margin-col",mode+"_margin"]
 artifact=out/("A15_CP3_AUDIT_REPLAY_"+stage+"_"+cohort+".json")
 gate(not artifact.exists(),"CP3_AUDIT_REPLAY_ALREADY_EXISTS")
 x=subprocess.run(command+["--out",str(artifact)],
                  capture_output=True,text=True)
 gate(x.returncode==0,"CP3_AUDIT_PROTOCOL_CHILD_"+stage+"_"+cohort+
      x.stderr[-400:])
 replay=jread(artifact)
 numeric=0
 for key,value in prior.items():
  if isinstance(value,(int,float)) and not isinstance(value,bool):
   assert_close(value,replay[key],
       "CP3_AUDIT_PROTOCOL_NUM_"+stage+"_"+cohort+"_"+key,1e-10)
   numeric+=1
  elif key in ("fixed_curves","by_cohort"):
   # Exact Python-native JSON structural equality is strict enough
   # for scalar arrays and their nested dictionaries.
   gate(replay[key]==value,
        "CP3_AUDIT_PROTOCOL_CURVES_"+stage+"_"+cohort)
  elif key not in ("inputs",):
   gate(value==replay.get(key),
        "CP3_AUDIT_PROTOCOL_META_"+stage+"_"+cohort+"_"+key)
 artifact.unlink()
 return dict(stage=stage,cohort=cohort,n_numeric_verified=numeric,
     exact_protocol_reexecution=True)
def check_pc_descriptive(out,stage):
 p=out/"CP3_G3_PC_DAPS_DESCRIPTIVE.json"
 d=jread(p)
 R=csvload(stage/"CP3_LOCAL_CANDIDATES.csv")
 C=csvload(stage/"CP3_CLIP_DECISIONS.csv")
 by=defaultdict(dict)
 cc={(r["image_id"],int(r["run"])):r for r in C}
 for r in R:by[r["image_id"]][int(r["run"])]=r
 gate(len(by)==180 and len(cc)==720 and
      all(set(x)==set(range(4)) for x in by.values()),
      "CP3_PC_PREAMBLE")
 agg=defaultdict(list)
 for iid,arr in by.items():
  ordered=[arr[i] for i in range(4)]
  sel=min(ordered,key=lambda r:(float(r["rho_signed"]),int(r["run"])))
  co=cc[iid,int(sel["run"])]
  agg["first_psnr_db"].append(float(ordered[0]["psnr_raw"]))
  agg["PC_DAPS_own_metric_mean_psnr_db"].append(
      sum(float(z["psnr_raw"]) for z in ordered)/4)
  agg["E0_PSNR"].append(float(sel["psnr_raw"]))
  agg["E1_PSNR"].append(float(sel["psnr_flip"] if co["face_choice"]=="111"
                          else sel["psnr_raw"]))
  agg["first_GOOD25"].append(int(ordered[0]["label"]=="G"))
  agg["mean_GOOD25_per_particle"].append(
      sum(z["label"]=="G" for z in ordered)/4)
  agg["E0_GOOD25"].append(int(sel["label"]=="G"))
  agg["E1_GOOD25"].append(int(co["face_chosen_good"]))
  agg["orbit_error_per_particle"].append(
      sum(z["label"] in ("O_g","O_p") for z in ordered)/4)
 for key,vals in agg.items():
  assert_close(sum(vals)/len(vals),d["mean_over_images"][key],
               "CP3_AUDIT_PC_DESCRIPTIVE_"+key)
 return dict(recomputed_metrics=list(agg),n_bags=180,
             native_metric_mean_vs_pixel_mean_separated=True)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_AUDIT_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(["git","-C",str(wt),*args],
                                         timeout=60)
 gate(git("rev-parse","HEAD").decode().strip()==a.expected_head
      and not git("status","--porcelain"),"CP3_AUDIT_HEAD")
 stamp=a.stamp
 gate(len(stamp)==16 and stamp.endswith("Z") and stamp[:8].isdigit()
      and stamp[8]=="T" and stamp[9:15].isdigit(),"CP3_AUDIT_STAMP")
 dest=BASE/("xp_a15_cp3_protocol_"+stamp)
 gate(dest.is_dir() and not dest.is_symlink(),"CP3_AUDIT_OUTPUT")
 g=jread(dest/"XP_A15_CP3_PROTOCOL_GATE.json")
 gate(g["status"]=="XP_A15_CP3_PROVISIONAL_PROTOCOL_AUDIT_NOT_FINAL"
      and g["CP3_RETURN_AUTHORIZED"] is False
      and g["independent_protocol_reproduction_complete"] is False,
      "CP3_AUDIT_SOURCE_NOT_FINAL")
 for filename,digest in g["SHA256SUMS"].items():
  gate(sha(dest/filename)==digest,"CP3_AUDIT_ORIGINAL_SHA_"+filename)
 src=git("show",PROTO[0]+":"+PROTO[1])
 gate(hashlib.sha256(src).hexdigest()==PROTO[2]
      and sha(dest/"gtfree_protocol_v1_original.py")==PROTO[2],
      "CP3_AUDIT_PROTO_SHA")
 scores={}
 for stage in STAGES:
  scores[stage]=independent_stage(stage,unique_stage_dir(stage,stamp))
 print("CP3_A15_ALL_SCORES_LABELS_CLIP_TAU_REAUDITED",flush=True)
 results=[]
 for stage in STAGES:
  if stage=="G3":continue
  stage_dir=unique_stage_dir(stage,stamp)
  for cohort in (("IN",) if stage=="G5" else ("B22","DEV80")):
   original=dest/("PROTOCOL_"+stage_dir.name+"_"+cohort+".json")
   gate(original.is_file(),"CP3_AUDIT_ORIGINAL_PROTOCOL_"+stage+"_"+cohort)
   mode=("g1" if stage=="G5" else "face") if stage in ("G4","G5") else None
   results.append(rerun_protocol(dest,
    dest/"gtfree_protocol_v1_original.py",original,stage,cohort,
    stage_dir,mode))
   print("CP3_A15_FROZEN_PROTOCOL_REPLAY_PASS",stage,cohort,flush=True)
 pc=check_pc_descriptive(dest,unique_stage_dir("G3",stamp))
 h1={}
 for stage in ("G4","G5"):
  doc=jread(dest/("CP3_"+stage+"_PROTOCOL.json"))
  gate(doc["H1"]==bool(doc["weighted"]["gain_vs_fixed_equal_runs"]>=-.01),
       "CP3_AUDIT_H1_"+stage)
  h1[stage]=doc["H1"]
 h_g={}
 for stage in ("G2_0.02","G2_0.10"):
  doc=jread(dest/("CP3_"+stage.replace(".","p")+"_PROTOCOL.json"))
  expected=.0500829*float(stage.split("_")[1])/.05
  assert_close(doc["H_G1_tau_scaled"],expected,"CP3_AUDIT_HG1_SCALE")
  assert_close(doc["H_G1_tau_gt"],scores[stage]["tau_gt_q95"],
      "CP3_AUDIT_HG1_GT")
  h_g[stage]={
   "HG1":doc["H_G1_pass"],
   "HG2":doc["H_G2_pass"]
  }
 record=dict(
  status="XP_A15_CP3_INDEPENDENT_SCORES_PROTOCOL_AUDIT_PASS_NOT_FINAL",
  expected_head=a.expected_head,
  stages=scores,
  frozen_protocol_exact_reproductions=results,
  pc_daps_descriptive_independently_recomputed=pc,
  H1_daps=h1,H_G=h_g,GT_not_used_for_selection=True,
  original_provisional_gate_sha256=sha(dest/"XP_A15_CP3_PROTOCOL_GATE.json"),
  native_no_regeneration=True,no_CLIP_rerun=True,no_GPU=True,
  unresolved=[
    "PER_BAG_PC_DAPS_DENOSIER_FORWARD_COUNT_AND_MATCHED_DAPS_NFE",
    "SSIM_AND_PIQ_VGG_LPIPS_OFFICIAL_PUBLISHED_PROTOCOL_METRICS",
    "INDEPENDENT_Y_GT_IDENTITY_SECONDARY_AUDIT_IF_NOT_PREVIOUSLY_CERTIFIED",
    "SANITIZED_CP3_RETURN_AND_PLANNER_REPRODUCTION"
  ],
  CP3_RETURN_AUTHORIZED=False,
 )
 output=dest/"XP_A15_CP3_INDEPENDENT_AUDIT.json"
 write(output,record)
 print("CP3_INDEPENDENT_AUDIT_PASS__STILL_NFE_AND_METRICS_PENDING",flush=True)
 print("CP3_INDEPENDENT_AUDIT",output,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--stamp",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A15_CP3_INDEPENDENT_AUDIT",e,flush=True)
  sys.exit(8)
