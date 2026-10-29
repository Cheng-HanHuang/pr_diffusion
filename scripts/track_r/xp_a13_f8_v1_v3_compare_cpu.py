#!/usr/bin/env python3
"""A13 CP1 F8 primary v1/secondary v3 DAPS prospective comparison.

Frozen 112 DAPS forecast rows each. Use exact 8-run E0/E1/oracle
U-statistics and registered before-outcome v1/v3 commits. Report all
variants and both cohorts, with no model re-estimation. Add A9.2
prefix-vs-extension composition and failure categories.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import Counter,defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a7_postfreeze_kernel_cpu import exact_panel,rank_prob,compare_interval
from scripts.track_r.xp_b16_v1_v3_admission_cpu import FREEZES
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open("x") as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def score(a):
 sc=Path(a.score_dir);cl=Path(a.clip_dir)
 gate(sc.parent==BASE and sc.name.startswith("xp_a13_f8_score_") and
      cl.parent==BASE and cl.name.startswith("xp_a13_f8_clip_"),
      "SOURCE_INPUT_PATHS")
 srec=load(sc/"XP_A13_F8_SCORING_RECEIPT.json")
 crec=load(cl/"XP_A13_F8_FACECLIP_RECEIPT.json")
 scored=sc/"XP_A13_F8_1440_SCORED_ROOTS.csv"
 clip=cl/"XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv"
 gate(srec["status"]=="XP_A13_F8_1440_CPU_SCORED_PASS"
      and crec["status"]=="XP_A13_F8_1440_N2_FACE_CLIP_PASS"
      and sha(scored)==srec["scoring_csv_sha256"]
      and sha(clip)==crec["clip_csv_sha256"]
      and crec["score_sha256"]==srec["scoring_csv_sha256"],
      "SOURCE_SHA")
 with scored.open(newline="") as f:rr=list(csv.DictReader(f))
 with clip.open(newline="") as f:cc=list(csv.DictReader(f))
 gate(len(rr)==len(cc)==1440,"1440_CLIP_SCORE_GRID")
 cm={}
 for c in cc:
  k=(c["cohort"],c["image_id"],int(c["run"]))
  gate(k not in cm,"CLIP_DUP");cm[k]=c
 panels=defaultdict(dict)
 for row in rr:
  k=(row["cohort"],row["image_id"])
  r=int(row["run"])
  gate(r not in panels[k] and (k[0] in ("B22","DEV80"))
       and (k[0],k[1],r) in cm,"COHORT_RUN_ID")
  c=cm[k[0],k[1],r]
  gate(c["label"]==row["label"] and
       int(c["f_chosen_good"]) in (0,1),"CLIP_LABEL")
  panels[k][r]=dict(run=r,rho=float(row["rho_signed"]),
   good_raw=row["label"]=="G",
   good_orbit=row["label"] in ("G","O_g","O_p"),
   chosen_good=int(c["f_chosen_good"]),
   label=row["label"],psnr_raw=float(row["psnr_raw"]),
   psnr_orbit_best=float(row["psnr_orbit_best"]),
   clip_margin=float(c["f_margin"]))
 gate(len(panels)==180 and
      all(set(v)==set(range(8)) for v in panels.values()),"FULL_PANEL")
 groups=defaultdict(list)
 for (cohort,iid),runs in sorted(panels.items()):
  r=[runs[x] for x in range(8)]
  groups[cohort].append((iid,r))
  groups["POOLED"].append((cohort+":"+iid,r))
 gate(len(groups["B22"])==100 and
      len(groups["DEV80"])==80 and
      len(groups["POOLED"])==180,"COHORT_COUNTS")
 obs={};per_image=[]
 for coh in ("B22","DEV80","POOLED"):
  ks=(4,6,8) if coh=="B22" else (6,8)
  for k in ks:
   vals=defaultdict(list)
   for image,r in groups[coh]:
    exact=exact_panel(r,k)
    order=sorted(r,key=lambda x:(x["rho"],x["run"]))
    exact["E1_CLIP"]=sum(float(q["chosen_good"])*
          rank_prob(8,k,idx+1) for idx,q in enumerate(order))
    for pol,val in exact.items():
     vals[pol].append(float(val))
     per_image.append(dict(cohort=coh,image_id=image,
          K=k,policy=pol,exact_success=val))
   for pol,ss in vals.items():
    obs[coh,k,pol]=sum(ss)/len(ss)
   gate(obs[coh,k,"E0"]<=obs[coh,k,"O_raw"]+1e-10 and
        obs[coh,k,"O_raw"]<=obs[coh,k,"O_orbit"]+1e-10,
        "F8_ORACLE_MONOTONE")
 comparisons=[];cover={}
 for ver,(commit,path,digest,n) in FREEZES.items():
  if ver not in ("v1","v3"):continue
  blob=subprocess.check_output(["git","show",commit+":"+path])
  gate(hashlib.sha256(blob).hexdigest()==digest,"FROZEN_FORECAST_SHA_"+ver)
  forecast=json.loads(blob)
  prediction=[r for r in forecast["predictions"]
             if r["family"]=="DAPS" and r["panel"]=="primary"
                and r["panel_n_max"]==8]
  gate(len(prediction)==112,"DAPS_FORECAST_ROW_COUNT_"+ver)
  seen=set()
  for row in prediction:
   key=(row["cohort"],int(row["K"]),row["policy"])
   gate(key in obs,"FROZEN_FORECAST_K_GRID_"+ver)
   index=(row["cohort"],row["K"],row["policy"],row["model_variant"])
   gate(index not in seen,"FORECAST_DUP");seen.add(index)
   result=compare_interval(obs[key],dict(row,
                    interval_type=forecast["interval_type"]))
   comparisons.append(dict(version=ver,cohort=row["cohort"],
      K=row["K"],policy=row["policy"],
      model_variant=row["model_variant"],**result))
  gate(len(seen)==112,"FROZEN_DAPS_COVERAGE_"+ver)
  cover[ver]={}
  for variant in ("race","S2","tail","race_tau1"):
   x=[r for r in comparisons if r["version"]==ver and
      r["model_variant"]==variant]
   gate(len(x)==28,"VARIANT_ROWS_"+variant)
   cover[ver][variant]=dict(n=28,inside=sum(r["inside"] for r in x),
      worst_abs_standardized_residual=max(
       abs(r["standardized_residual"]) for r in x))
 exchange=[];never=[];selected_losses=[]
 for cohort,rr in groups.items():
  oldn=2 if cohort=="B22" else 4 if cohort=="DEV80" else None
  if oldn:
   for lab in ("G","O_g","O_p","C","I"):
    early=sum(r["label"]==lab for _,runs in rr
              for r in runs[:oldn])
    late=sum(r["label"]==lab for _,runs in rr
              for r in runs[oldn:])
    exchange.append(dict(cohort=cohort,label=lab,
      historical_rate=early/(oldn*len(rr)),
      later_rate=late/((8-oldn)*len(rr)),
      difference=late/((8-oldn)*len(rr))-early/(oldn*len(rr))))
  if cohort=="POOLED":continue
  for iid,runs in rr:
   raw=sum(r["good_raw"] for r in runs)
   orbit=sum(r["good_orbit"] for r in runs)
   if raw==0:never.append(dict(cohort=cohort,image_id=iid,
                   n_G=0,n_orbit_good=orbit))
   pick=min(runs,key=lambda r:(r["rho"],r["run"]))
   if raw>0 and not pick["good_raw"]:
    selected_losses.append(dict(cohort=cohort,image_id=iid,
      chosen_run=pick["run"],chosen_label=pick["label"],
      n_G=raw,chosen_good_after_face=pick["chosen_good"]))
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a13_f8_compare_")
      and not out.exists() and not out.is_symlink(),"NEW_OUTPUT")
 out.mkdir(mode=0o700)
 summary=dict(status="XP_A13_F8_V1_PRIMARY_V3_SECONDARY_COMPARISON_PASS",
   head=a.expected_head,n_images=180,n_runs=8,
   forecast_v1_sha256=FREEZES["v1"][2],
   forecast_v3_sha256=FREEZES["v3"][2],
   forecast_rows=len(comparisons),comparisons=comparisons,
   coverage=cover,
   observed=[dict(cohort=c,K=k,policy=p,success=v)
             for (c,k,p),v in sorted(obs.items())],
   A9_2_early_late_composition=exchange,
   never_GOOD=never,selection_losses=selected_losses,
   n_never_GOOD=len(never),n_selection_losses=len(selected_losses),
   score_sha256=sha(scored),clip_sha256=sha(clip),
   no_forecast_refit=True,no_GPU=True)
 target=out/"XP_A13_F8_PROSPECTIVE_V1_V3.json"
 write(target,summary)
 per=out/"XP_A13_F8_PER_IMAGE_EXACT.csv"
 with per.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(per_image[0]))
  w.writeheader();w.writerows(per_image)
 write(out/"XP_A13_F8_COMPARISON_RECEIPT.json",dict(
   status=summary["status"],comparison_sha256=sha(target),
   per_image_sha256=sha(per),n_forecast_rows=len(comparisons),
   frozen_v1_v3=True))
 print("XP_A13_F8_V1_V3_PROSPECTIVE_PASS",len(comparisons),flush=True)
 print("F8_V1_COVERAGE",cover["v1"],flush=True)
 print("F8_V3_COVERAGE",cover["v3"],flush=True)
 print("F8_COMP_OUT",out,flush=True)
def main():
 p=argparse.ArgumentParser()
 for k in ("expected-head","score-dir","clip-dir","out"):
  p.add_argument("--"+k,required=True)
 a=p.parse_args()
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                  text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"SOURCE")
 score(a)
if __name__=="__main__":
 try:main()
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_F8_V1_V3",e,flush=True)
  sys.exit(8)
