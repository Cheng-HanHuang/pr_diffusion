#!/usr/bin/env python3
"""A15 CP3: frozen protocol on five A14 scored panels and G3 descriptive.

Provisional only until independent audit, exact NFE accounting and
per-particle SSIM/LPIPS pass. This script NEVER declares CP3 returned.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,safe,csvload,csvwrite,write,PROTO

STAGES=("G2_0.02","G2_0.10","G4","G5","G3")
COHORTS={"B22":100,"DEV80":80}
def verify(directory,stage):
 p=Path(directory)
 gate(p.parent==BASE and p.name.startswith("xp_a15_cp3_"+stage.replace(".","p")+"_")
      and not p.is_symlink(),"CP3_STAGE_PATH")
 rec=jread(p/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json")
 gate(rec["status"]=="XP_A15_CP3_STAGE_CPU_SCORED_NOT_CHECKPOINT"
      and rec["stage"]==stage and rec["no_GPU"] is True
      and rec["n_images"]==(100 if stage=="G5" else 180)
      and rec["n_candidates"]==(800 if stage=="G5" else
       720 if stage=="G3" else 1440),
      "CP3_STAGE_RECEIPT")
 for file,digest in rec["SHA256SUMS"].items():
  gate(sha(p/file)==digest,"CP3_STAGE_SHA_"+file)
 S=p/"CP3_A9_STRINGS.csv";C=p/"CP3_CLIP_DECISIONS.csv"
 R=p/"CP3_LOCAL_CANDIDATES.csv"
 strings=csvload(S);clips=csvload(C);runs=csvload(R)
 n=int(rec["n_runs_per_image"])
 gate(len(strings)==rec["n_images"] and
      len(clips)==len(runs)==rec["n_candidates"] and
      all(len(r["labels_by_run"].split(";"))==n for r in strings)
      and all(sum(int(r["n_"+c]) for c in ("G","O_g","O_p","C","I"))==n
              for r in strings),"CP3_STAGE_N_RUNS")
 return rec,S,C,R,strings,clips,runs
def gt_free_protocol(out,script,strings,clip,cohort,variant,tau):
 args=[sys.executable,"-I",str(script),"--strings",str(strings),
       "--m","2","--nmax","8","--R","4000","--seed","20261029"]
 if cohort:args+=["--cohort",cohort]
 if clip:
  args+=["--clip",str(clip),
         "--clip-good-col",("g1_chosen_good" if variant=="G1" else "face_chosen_good"),
         "--clip-margin-col",("g1_margin" if variant=="G1" else "face_margin")]
 if tau=="self_mad3":args+=["--tau-rule","self_mad3"]
 else:args+=["--tau",str(tau)]
 dest=out/("PROTOCOL_"+str(strings.parent.name)+"_"+(cohort or "ALL")+".json")
 args+=["--out",str(dest)]
 cp=subprocess.run(args,capture_output=True,text=True)
 gate(cp.returncode==0,"CP3_FROZEN_PROTOCOL_"+cp.stdout[-300:]+cp.stderr[-300:])
 doc=jread(dest)
 gate(doc["protocol_version"]=="v1" and doc["N_MAX"]==8
      and doc["n_images"]==(COHORTS[cohort] if cohort in COHORTS else
         100 if cohort=="IN" else 180)
      and doc["M"]==2,"CP3_PROTOCOL_SCHEMA")
 return doc,sha(dest)
def pooled(cohorts):
 counts={k:COHORTS[k] for k in cohorts}
 n=sum(counts.values())
 weighted={}
 for key in set.intersection(*(set(cohorts[c]) for c in cohorts)):
  vals=[cohorts[c][key] for c in cohorts]
  if all(isinstance(v,(int,float)) and not isinstance(v,bool)
         and math.isfinite(v) for v in vals):
   weighted[key]=sum(cohorts[c][key]*counts[c]/n for c in cohorts)
 return weighted
def pc_daps_descriptive(runs,clips):
 by=defaultdict(dict)
 selection={(r["image_id"],int(r["run"])):r for r in clips}
 gate(len(selection)==720,"CP3_PC_CLIP_720")
 for r in runs:
  k=r["image_id"];i=int(r["run"])
  gate(i not in by[k],"CP3_PC_UNIQUE")
  by[k][i]=r
 gate(len(by)==180 and all(set(x)==set(range(4)) for x in by.values()),
      "CP3_PC_FOUR")
 vals=defaultdict(list)
 for iid,rows in sorted(by.items()):
  rr=[rows[j] for j in range(4)]
  sel=min(rr,key=lambda x:(float(x["rho_signed"]),int(x["run"])))
  c=selection[iid,int(sel["run"])]
  flipped=c["face_choice"]=="111"
  E1=float(sel["psnr_flip"] if flipped else sel["psnr_raw"])
  scores=[float(x["psnr_raw"]) for x in rr]
  vals["first_psnr_db"].append(scores[0])
  vals["PC_DAPS_own_metric_mean_psnr_db"].append(sum(scores)/4)
  vals["E0_PSNR"].append(float(sel["psnr_raw"]))
  vals["E1_PSNR"].append(E1)
  vals["first_GOOD25"].append(int(rr[0]["label"]=="G"))
  vals["mean_GOOD25_per_particle"].append(sum(x["label"]=="G" for x in rr)/4)
  vals["E0_GOOD25"].append(int(sel["label"]=="G"))
  vals["E1_GOOD25"].append(int(c["face_chosen_good"]))
  vals["orbit_error_per_particle"].append(sum(x["label"] in ("O_g","O_p") for x in rr)/4)
 return dict(status="PC_DAPS_NATIVE_FOUR_PARTICLE_METRIC_MEAN_DESCRIPTIVE",
   n_bags=180,n_particles=720,
   mean_over_images={name:sum(v)/len(v) for name,v in vals.items()},
   own_estimate_is_mean_of_four_particle_METRICS_not_pixel_mean=True,
   DAPS_matched_NFE_comparison_status="PENDING_AUTHENTICATED_PER_BAG_NFE",
   not_sufficient_for_CP3_release=True)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],timeout=60)
 gate(git("rev-parse","HEAD").decode().strip()==a.expected_head
      and not git("status","--porcelain"),"CP3_GIT")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a15_cp3_protocol_")
      and not out.exists(),"CP3_NEW_OUTPUT")
 paths={"G2_0.02":a.g2_02,"G2_0.10":a.g2_10,
        "G4":a.g4,"G5":a.g5,"G3":a.g3}
 panels={k:verify(paths[k],k) for k in STAGES}
 frozen=git("show",PROTO[0]+":"+PROTO[1])
 gate(hashlib.sha256(frozen).hexdigest()==PROTO[2],"CP3_PROTOCOL_SOURCE_SHA")
 if a.preflight_only:
  print("CP3_ALL_FIVE_PANELS_VERIFIED_NOT_YET_CHECKPOINT",flush=True)
  return
 out.mkdir(mode=0o700)
 script=out/"gtfree_protocol_v1_original.py"
 with script.open("xb") as f:f.write(frozen)
 docs={}
 for stage in STAGES:
  receipt,S,C,R,strings,clips,runs=panels[stage]
  if stage=="G3":
   docs["G3"]=pc_daps_descriptive(runs,clips)
   write(out/"CP3_G3_PC_DAPS_DESCRIPTIVE.json",docs["G3"])
   continue
  useclip=stage in ("G4","G5")
  tau=("self_mad3" if useclip else .0500829*float(stage.split("_")[1])/.05)
  per={}
  cohort_list=("IN",) if stage=="G5" else ("B22","DEV80")
  # The frozen protocol indexes CLIP by (image_id, run) only. Slice
  # cohorts first to prevent any B22/DEV80 ID collision.
  clip_by_cohort={}
  if useclip:
   for cohort in cohort_list:
    fieldnames=list(clips[0])
    rows=[r for r in clips if r["cohort"]==cohort]
    gate(len(rows)==(100 if cohort=="IN" else COHORTS[cohort])*8,
         "CP3_COHORT_CLIP")
    cut=out/(stage+"_CLIP_"+cohort+".csv")
    csvwrite(cut,rows,fieldnames)
    clip_by_cohort[cohort]=cut
  for cohort in cohort_list:
   r,digest=gt_free_protocol(out,script,S,
        clip_by_cohort.get(cohort),
        cohort,("G1" if stage=="G5" else "face"),tau)
   per[cohort]=r
  weighted=(per["IN"] if stage=="G5" else pooled(per))
  h1=weighted["gain_vs_fixed_equal_runs"]>=-.01
  doc=dict(status="XP_A15_CP3_PROTOCOL_STAGE_PASS",
      stage=stage,threshold_rule=tau,cohorts={
        k:dict(n=100 if k=="IN" else COHORTS[k],
               success_all=v["success_all"],runs_mean=v["runs_mean"],
               gain_vs_equal_compute=v["gain_vs_fixed_equal_runs"])
        for k,v in per.items()},
      weighted=weighted,H1=bool(h1),protocol_sha256=PROTO[2],
      GT_for_evaluation_only=True,no_training_or_retuning=True)
  if stage.startswith("G2_"):
   observed_tau=list({float(s["tau_signed_good_q95"]) for s in strings})
   gate(len(observed_tau)==1,"CP3_SIGMA_UNIQUE_GT_TAU")
   doc["H_G1_tau_gt"]=observed_tau[0]
   doc["H_G1_tau_scaled"]=tau
   doc["H_G1_pass"]=abs(observed_tau[0]-tau)/observed_tau[0]<=.01
   doc["H_G2_pass"]=bool(h1)
   doc["E0_only_primary_no_CLIP"]=True
  docs[stage]=doc
  write(out/("CP3_"+stage.replace(".","p")+"_PROTOCOL.json"),doc)
 g3=docs["G3"]
 complete=all(s in docs for s in STAGES)
 gate(complete,"CP3_FIVE_PANELS")
 release=dict(status="XP_A15_CP3_PROVISIONAL_PROTOCOL_AUDIT_NOT_FINAL",
   expected_head=a.expected_head,
   panels={k:dict(status=v["status"]) for k,v in docs.items()},
   H_G1={k:docs[k]["H_G1_pass"] for k in ("G2_0.02","G2_0.10")},
   H_G2={k:docs[k]["H_G2_pass"] for k in ("G2_0.02","G2_0.10")},
   H1_G4=docs["G4"]["H1"],H1_G5=docs["G5"]["H1"],
   PC_DAPS_native_particle_mean=g3["mean_over_images"],
   NFE_matched_comparison_complete=False,
   independent_protocol_reproduction_complete=False,
   LPIPS_SSIM_complete=False,CP3_RETURN_AUTHORIZED=False,
   SHA256SUMS={p.name:sha(p) for p in out.iterdir() if p.is_file()
       and p.name!="XP_A15_CP3_PROTOCOL_GATE.json"},
   no_GPU=True)
 write(out/"XP_A15_CP3_PROTOCOL_GATE.json",release)
 print("CP3_SCORED_PROTOCOL_NOT_RETURNABLE_NFE_AND_AUDIT_PENDING",
       out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 for x in ("g2-02","g2-10","g3","g4","g5","out"):
  p.add_argument("--"+x,required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A15_CP3_PROTOCOL",e,flush=True);sys.exit(8)
