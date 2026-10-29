#!/usr/bin/env python3
"""A13 CP1 F8: 180x8 DAPS 400x5 Good25 frozen RGB8 CPU scoring.

Reuses exact R0.7 historical prefix B22 n2 and DEV80 n4.
Authenticates 920 generated native PNGs, source y and GT identity;
does not run a GPU model or change any DAPS numerical method.
New roots are scored after v1/v3 registered, before protocol evaluation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,traceback
from pathlib import Path
from collections import Counter,defaultdict
import torch
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from prdiffusion.track_r.physics import (
 to_rgb8,from_rgb8,rho_signed,load_png_rgb8)
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from scripts.track_r.xp_b16_v1_v3_score_cpu import HEAD_ROWS,COLUMNS
R07=BASE/"r0_7_cpu_20261023T080434Z"
CAT=R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json"
SCORES=R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl"
R07_RECEIPT=R07/"analysis/R0_7_COMPLETE_RECEIPT.json"
F8=BASE/"xp_f8_prod_20261029T071352Z"
TAU={"S1_b22":0.0474649,"S2_b24_dev80":0.0475986}
NAMES={"S1_b22":"B22","S2_b24_dev80":"DEV80"}
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def safe(path):
 p=Path(path)
 gate(p.is_file() and not p.is_symlink() and
      ROOT.resolve() in p.resolve().parents and
      not any(k in str(p.resolve()).lower() for k in BAD),
      "SAFE_EXPOSED_SOURCE")
 return p
def load(p):return json.loads(safe(p).read_text())
def write(p,obj):
 with p.open("x") as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def label(raw,orbit,best,rho,tau):
 if raw>=25:return "G"
 if orbit>=25:return "O_g" if best=="111" else "O_p"
 return "C" if rho<=tau else "I"
def r07_source():
 rec=load(R07_RECEIPT)
 gate(rec["n_scored"]==3240 and sha(SCORES)==rec["evaluated_rows_sha256"],"R07_RECEIPT")
 catalog=load(CAT)["rows"]
 rows=defaultdict(dict);identity={}
 for x in catalog:
  if x["family"]!="DAPS" or x["source"] not in TAU:continue
  cohort=x["source"];iid=str(x["image_id"]).zfill(5)
  key=(cohort,iid);r=int(x["run_index"])
  gate(r not in rows[key],"HIST_DUP")
  rows[key][r]=x
  ids=(x["gt_path"],x["gt_format"],x["measurement_path"],
        x["measurement_sha256"])
  if key in identity:gate(identity[key]==ids,"HIST_GT_Y_IDENTITY")
  identity[key]=ids
 gate(len(rows)==180 and sum(x[0]=="S1_b22" for x in rows)==100,
      "HIST_180")
 for k,rr in rows.items():
  gate(set(rr)==set(range(2 if k[0]=="S1_b22" else 4)),
       "HIST_RUN_PREFIX")
 previous=defaultdict(dict)
 with safe(SCORES).open() as f:
  for line in f:
   x=json.loads(line)
   if x["family"]!="DAPS" or x["source"] not in TAU:continue
   key=(x["source"],str(x["image_id"]).zfill(5))
   rep=int(x["run_index"])
   gate(rep not in previous[key],"HIST_SCORE_DUP")
   previous[key][rep]=x
 gate(set(previous)==set(rows) and
      all(set(previous[k])==set(rows[k]) for k in rows),
      "HIST_SCORE_180_PREFIX")
 return rows,identity,previous
def load_production():
 manifest=F8/"XP_F8_PROD_TASKS.json"
 prep=load(F8/"XP_F8_PROD_PREP.json")
 state=load(F8/"XP_F8_CONTROLLER_STATE.json")
 done=load(F8/"XP_F8_PROD_LAST_STATUS.json")
 gate(prep["status"]=="XP_F8_920_GT_FREE_PRODUCTION_PREP_PASS"
      and sha(manifest)==prep["manifest_sha256"]
      and state["status"]=="XP_F8_920_GENERATION_COMPLETE_EMBARGOED"
      and done["status"]=="XP_F8_920_GENERATION_COMPLETE_EMBARGOED"
      and done["complete"]==920
      and len(state["done"])==920 and state["active"]==[]
      and len(state["failure_log"])==0,
      "F8_GENERATION_COMPLETE")
 doc=load(manifest);rr=doc["rows"]
 gate(len(rr)==920 and doc["n_total"]==920,"F8_920")
 grouped=defaultdict(dict)
 for idx,row in enumerate(rr):
  gate(row["index"]==idx and
       row["cohort"] in TAU and
       row["rep"] in (range(2,8) if row["cohort"]=="S1_b22" else range(4,8)),
       "NEW_ROOT_INDEX")
  cohort=row["cohort"];iid=str(row["image_id"]).zfill(5)
  key=(cohort,iid)
  gate(row["rep"] not in grouped[key],"F8_DUP")
  receipt_ref=state["done"][str(idx)]
  attempt=int(receipt_ref["attempt"])
  p=F8/"roots"/f"r{idx:04d}_a{attempt}"/"XP_F8_NEW_ROOT_COMPLETE.json"
  gate(sha(p)==receipt_ref["receipt_sha256"],"F8_STATE_RECEIPT")
  r=load(p)
  gate(r["status"]=="XP_F8_NEW_ROOT_GENERATION_PASS_EMBARGOED"
      and r["root_index"]==idx
      and r["uid"]==row["uid"]
      and r["rep"]==row["rep"]
      and r["measurement_sha256"]==row["measurement_sha256"]
      and r["native_seed"]==row["native_seed"]
      and r["no_GT_scoring"] is True
      and r["no_labels_calculated"] is True
      and r["manifest_sha256"]==prep["manifest_sha256"]
      and sha(safe(r["output_path"]))==r["output_sha256"],
      "F8_NATIVE_PNG_RECEIPT")
  grouped[key][row["rep"]]=(row,r)
 gate(len(grouped)==180 and
      all(set(x)==set(range(2,8) if k[0]=="S1_b22" else range(4,8))
          for k,x in grouped.items()),"920_FULL_EXTENSION")
 return grouped,prep
def score(a):
 historic,identity,prefix=r07_source()
 future,prep=load_production()
 gate(set(future)==set(identity),"F8_GT_COHORT_IDENTITY")
 torch.set_num_threads(4)
 panels={}; scored=[]
 for cohort,iid in sorted(identity):
  gtpath,gtfmt,ypath,yhash=identity[cohort,iid]
  gate(sha(safe(ypath))==yhash,"F8_Y_SHA")
  y=read_tensor(safe(ypath),("measurement","y","observation"),(1,3,384,384))
  gt=to_rgb8(gt_source(dict(gt_path=gtpath,gt_format=gtfmt)).float())
  selected={}
  for rep,row in sorted(historic[cohort,iid].items()):
   h=prefix[cohort,iid][rep]
   gate(h["uid"]==row["uid"],"HIST_UID")
   raw=float(h["psnr_raw"]);orbit=float(h["psnr_orbit_best"])
   rho=float(h["rho_signed_raw"]); best=str(h["s_gt"])
   lbl=label(raw,orbit,best,rho,TAU[cohort])
   prior={"GOOD":"G","ORBIT_ERROR":"O_g" if best=="111" else "O_p",
     "NONORBIT_CONSISTENT":"C","NONORBIT_INCONSISTENT":"I"}.get(
      h["raw_label"])
   gate(lbl==prior,"R07_LABEL_TAU_IDENTITY")
   selected[rep]=dict(image_id=iid,cohort=NAMES[cohort],run=rep,
     label=lbl,rho_signed=rho,psnr_raw=raw,
     psnr_orbit_best=orbit,s_gt=best,
     flip_good=int(float(h["psnr_by_orbit"]["111"])>=25),
     reconstruction_path=row["candidate_path"],
     reconstruction_format=row["candidate_format"],
     reconstruction_sha256=row["candidate_sha256"],
     measurement_sha256=yhash,uid=row["uid"])
  for rep,(row,receipt) in sorted(future[cohort,iid].items()):
   gate(row["measurement_sha256"]==yhash,"F8_SHARED_Y_SHA")
   x8=load_png_rgb8(safe(receipt["output_path"]))
   orbit_map=orbit_psnr_channel_mse(x8,gt)
   raw=float(orbit_map["000"])
   best=max(sorted(orbit_map),key=lambda k:orbit_map[k])
   orbit=float(orbit_map[best])
   rho=float(rho_signed(from_rgb8(x8),y)[0])
   gate(all(math.isfinite(t) for t in (raw,orbit,rho))
        and rho>=0,"F8_NUMERIC")
   selected[rep]=dict(image_id=iid,cohort=NAMES[cohort],run=rep,
      label=label(raw,orbit,best,rho,TAU[cohort]),
      rho_signed=rho,psnr_raw=raw,psnr_orbit_best=orbit,s_gt=best,
      flip_good=int(orbit_map["111"]>=25),
      reconstruction_path=receipt["output_path"],
      reconstruction_format="png",
      reconstruction_sha256=receipt["output_sha256"],
      measurement_sha256=yhash,uid=row["uid"])
  gate(set(selected)==set(range(8)),"F8_8RUN_PANEL")
  panels[cohort,iid]=selected
  scored.extend(selected[r] for r in range(8))
  if len(panels)%10==0:
   print("A13_F8_IMAGE_SCORED",len(panels),"/180",flush=True)
 gate(len(panels)==180 and len(scored)==1440,"F8_1440")
 strings=[]
 for cohort,iid in sorted(panels):
  rr=[panels[cohort,iid][i] for i in range(8)]
  cc=Counter(x["label"] for x in rr)
  strings.append(dict(image_id=iid,cohort=NAMES[cohort],
    labels_by_run=";".join(x["label"] for x in rr),
    rho_signed_by_run=";".join(format(x["rho_signed"],".10g") for x in rr),
    tau_signed_good_q95=TAU[cohort],
    n_G=cc["G"],n_O_g=cc["O_g"],n_O_p=cc["O_p"],
    n_C=cc["C"],n_I=cc["I"]))
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a13_f8_score_")
      and not out.exists() and not out.is_symlink(),"OUTPUT")
 out.mkdir(mode=0o700)
 s=out/"XP_A13_F8_8RUN_STRINGS.csv"
 with s.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=HEAD_ROWS);w.writeheader();w.writerows(strings)
 c=out/"XP_A13_F8_1440_SCORED_ROOTS.csv"
 with c.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader()
  w.writerows(sorted(scored,key=lambda x:(x["cohort"],x["image_id"],x["run"])))
 payload=dict(status="XP_A13_F8_1440_CPU_SCORED_PASS",
    source_head=a.expected_head,n_images=180,n_runs=8,
    n_historical=520,n_new=920,n_candidates=1440,
    cohort_images={"B22":100,"DEV80":80},
    frozen_TAU={"B22":TAU["S1_b22"],
                "DEV80":TAU["S2_b24_dev80"]},
    scoring_csv_sha256=sha(c),strings_csv_sha256=sha(s),
    f8_manifest_sha256=prep["manifest_sha256"],
    historical_R07_scores_sha256=sha(SCORES),
    all_920_root_receipts_verified=True,
    scores_after_v1_v3_registry=True,no_GPU=True,
    class_counts=dict(Counter(r["label"] for r in scored)))
 write(out/"XP_A13_F8_SCORING_RECEIPT.json",payload)
 print("XP_A13_F8_1440_CPU_SCORING_PASS",flush=True)
 print("F8_STRINGS_SHA",sha(s),flush=True)
 print("F8_SCORED_CANDIDATES_SHA",sha(c),flush=True)
 print("F8_SCORE_OUT",out,flush=True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:__import__("subprocess").check_output(["git","-C",str(wt),*q],
                     text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"PINNED_WORKTREE")
 score(a)
if __name__=="__main__":
 try:main()
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_F8_SCORING",e,flush=True)
  sys.exit(8)
