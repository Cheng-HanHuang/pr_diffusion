#!/usr/bin/env python3
"""A17.2 XP-N first four runs ONLY: CPU GT evaluation and frozen G1 CLIP.

Reads exact 400 pilot native roots and their previously frozen new-image
GT/y. Intentionally never reads other XP-N phases or their receipts,
states, candidate paths, outcomes, predictions, labels or PSNRs.
Output sanitized D1-style PREFIX4 and CLIP4 for prospective v5.
"""
from __future__ import annotations
import argparse,csv,hashlib,math,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write,clip_new
from scripts.track_r.xp_a18_xpn_contract import (
 Y_SHA,AUDIT_SHA,FROZEN_Y,PHASES,validate_xpn_source)
from prdiffusion.track_r.physics import to_rgb8,rho_signed
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse

PHASE="A18_XPN_PILOT_0_3"
MF="XP_N_PILOT_0_3_TASKS.json"
GT_FILE=FROZEN_Y.parent/"XP_N_GT_ONLY_CPU.json"
GT_SHA="e5f4f9d69c66e3dd8ec61b4c38184625259b1c7670f5635711d51762b2e73824"
COLP=("image_id","source","family","n_runs","n_GOOD",
      "n_ORBIT_global","n_ORBIT_partial","n_CONSISTENT",
      "n_INCONSISTENT","run_labels","e0_rank_labels",
      "e0_rank_order","rho_signed_by_run","tau_signed_good_q95")
COLC=("image_id","run","label","g1_choice","g1_margin","g1_chosen_good")
LOCAL_COLS=("image_id","run","label","rho_signed","psnr_raw",
            "psnr_orbit_best","psnr_flip","flip_good",
            "reconstruction_path","reconstruction_sha256")

def csv_x(p,rs,columns):
 with p.open("x",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
  w.writeheader();w.writerows(rs)

def inputs(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A19_XPN_P4_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *x:subprocess.check_output(["git","-C",str(wt),*x],
                                      text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"A19_XPN_CLEAN_HEAD")
 tasks=Path(a.tasks);run=Path(a.run)
 gate(tasks.parent==BASE and tasks.name.startswith("xp_a18_xpn_tasks_")
      and run.parent==BASE and run.name.startswith("xp_a18_xpn_prod_")
      and not tasks.is_symlink() and not run.is_symlink()
      and tasks.is_dir() and run.is_dir(),
      "A19_XPN_PINNED_SOURCE_DIRS")
 receipt=jread(tasks/"XP_N_TASKS_PREP_RECEIPT.json")
 mfpath=tasks/MF
 pilot=jread(mfpath);digest=sha(mfpath);rows=pilot["rows"]
 gate(receipt["status"]=="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
      and receipt["phase_files"][PHASE]["sha256"]==digest
      and len(rows)==400
      and receipt["source_gt_y_audit_sha256"]==AUDIT_SHA,
      "A19_XPN_P4_FROZEN_MANIFEST")
 producer=receipt["expected_head"]
 for st in range(0,400,20):
  validate_xpn_source(mfpath,digest,pilot,rows,producer,st,20)
  dest=run/"production"/PHASE/f"chunk_{st:05d}_020"
  gate(dest.is_dir() and not dest.is_symlink(),
       "A19_XPN_ONLY_PILOT_CHUNK_MISSING_"+str(st))
  meta=jread(dest/"R0_5_CHUNK_META.json")
  complete=jread(dest/"R0_5_CHUNK_COMPLETE.json")
  gate(meta["start"]==st and meta["count"]==20
       and meta["root_sha256"]==digest
       and complete["status"]=="R0_5_CHUNK_PASS"
       and complete["start"]==st and complete["count"]==20
       and complete["new"]==20 and complete["head"]==meta["head"]
       and complete["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json")
       and meta["uids"]==[r["uid"] for r in rows[st:st+20]],
       "A19_XPN_P4_NATIVE_COMPLETE_"+str(st))
 gate(sha(GT_FILE)==GT_SHA and sha(FROZEN_Y)==Y_SHA,
      "A19_XPN_P4_PINNED_GT_Y_MANIFEST_SHA")
 gt=jread(GT_FILE);ym=jread(FROZEN_Y)
 gate(gt["n_images"]==ym["n_images"]==100
      and len(gt["rows"])==len(ym["rows"])==100,
      "A19_XPN_P4_100_GT")
 return rows,digest,run,gt,ym

def score_prefix(rows,digest,run,gt,ym):
 import torch,numpy as np
 torch.set_num_threads(4)
 gby={r["basename"]:r for r in gt["rows"]}
 yby={r["basename"]:r for r in ym["rows"]}
 allgood=[];out=[]
 for image_index,iid in enumerate(sorted(gby)):
  r4=rows[image_index*4:image_index*4+4]
  gate([r["image_id"] for r in r4]==[iid]*4
       and [r["rep"] for r in r4]==list(range(4)),
       "A19_XPN_P4_IMAGE_ALIGNMENT")
  g=gby[iid];yr=yby[iid]
  gp=Path(g["gt_path"])
  yp=Path(yr["measurement_path"])
  gate(sha(gp)==g["gt_file_sha256"]
       and sha(yp)==yr["measurement_sha256"]
       and g["measurement_sha256"]==yr["measurement_sha256"],
       "A19_XPN_P4_GT_Y_SHA")
  with torch.inference_mode():
   xgt=torch.load(gp,map_location="cpu",weights_only=True)["gt_model_range"].float()
   y=torch.load(yp,map_location="cpu",weights_only=True)["measurement"].float()
   gate(tuple(xgt.shape)==(1,3,256,256) and
        tuple(y.shape)==(1,3,384,384),"A19_XPN_P4_GT_Y_SHAPE")
   rgb_gt=to_rgb8(xgt)
   for r in r4:
    rep=int(r["rep"])
    start=(image_index//5)*20
    dest=run/"production"/PHASE/f"chunk_{start:05d}_020"
    path=root_folder(dest,r)
    rc=validate_complete(path,r,jread(dest/"R0_5_CHUNK_META.json")["head"],digest)
    xx=torch.load(path/"reconstruction.pt",map_location="cpu",weights_only=True)["reconstruction"].float()
    gate(tuple(xx.shape)==(1,3,256,256)
         and bool(torch.isfinite(xx).all()),"A19_XPN_P4_NATIVE_FLOAT")
    img=to_rgb8(xx)
    from prdiffusion.track_r.physics import load_png_rgb8
    gate(torch.equal(img,load_png_rgb8(path/"reconstruction.png")),
         "A19_XPN_P4_FLOAT_PNG_IDENTITY")
    orbit=orbit_psnr_channel_mse(img,rgb_gt)
    best=max(sorted(orbit),key=lambda key:orbit[key])
    raw=float(orbit["000"]);orbest=float(orbit[best])
    rho=float(rho_signed(xx,y)[0])
    gate(all(math.isfinite(v) for v in (raw,orbest,rho))
         and rho>=0,"A19_XPN_P4_FINITE")
    if raw>=20:allgood.append(rho)
    out.append(dict(image_id=iid,run=rep,label=None,
      rho_signed=format(rho,".15g"),psnr_raw=format(raw,".12g"),
      psnr_orbit_best=format(orbest,".12g"),
      psnr_flip=format(float(orbit["111"]),".12g"),
      flip_good=int(float(orbit["111"])>=20),
      reconstruction_path=str(path/"reconstruction.png"),
      reconstruction_sha256=rc["reconstruction_png_sha256"],
      _best=best))
  if (image_index+1)%10==0:
   print("A19_XPN_P4_SCORED",image_index+1,"/100",flush=True)
 gate(len(out)==400 and len(allgood)>0,
      "A19_XPN_P4_GOOD_CALIBRATION_EXISTENCE")
 tau=float(np.quantile(allgood,.95,method="linear"))
 gate(math.isfinite(tau) and 0<tau<.2,"A19_XPN_P4_TAU")
 for r in out:
  raw=float(r["psnr_raw"]);orb=float(r["psnr_orbit_best"])
  r["label"]=("G" if raw>=20 else
    "O_g" if orb>=20 and r["_best"]=="111" else
    "O_p" if orb>=20 else
    "C" if float(r["rho_signed"])<=tau else "I")
  del r["_best"]
 return out,tau

def summaries(rows,tau):
 mapping={"G":"G","O_g":"O","O_p":"P","C":"C","I":"I"}
 grouped={}
 for r in rows:grouped.setdefault(r["image_id"],[]).append(r)
 gate(len(grouped)==100 and all(len(v)==4 for v in grouped.values()),
      "A19_XPN_P4_100X4")
 result=[]
 for iid,g in sorted(grouped.items()):
  q=sorted(g,key=lambda r:int(r["run"]))
  gate([int(r["run"]) for r in q]==list(range(4)),
       "A19_XPN_P4_RUN_ORDER")
  labs="".join(mapping[r["label"]] for r in q)
  order=sorted(range(4),key=lambda k:(float(q[k]["rho_signed"]),k))
  ranked="".join(labs[k] for k in order)
  c=Counter(labs)
  result.append(dict(image_id=iid,source="XP_N",family="FAST",
   n_runs=4,n_GOOD=c["G"],n_ORBIT_global=c["O"],
   n_ORBIT_partial=c["P"],n_CONSISTENT=c["C"],
   n_INCONSISTENT=c["I"],run_labels=labs,
   e0_rank_labels=ranked,e0_rank_order=",".join(map(str,order)),
   rho_signed_by_run=";".join(r["rho_signed"] for r in q),
   tau_signed_good_q95=format(tau,".17g")))
 return result

def main(a):
 rows,digest,run,gt,ym=inputs(a)
 if a.preflight_only:
  print("A19_XPN_PREFIX4_NATIVE_400_PREFLIGHT_PASS",flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a19_xpn_prefix4_")
      and not out.exists() and not out.is_symlink(),
      "A19_XPN_P4_OUTPUT_FRESH")
 sc,tau=score_prefix(rows,digest,run,gt,ym)
 clip,info=clip_new(sc,mode="g1",threshold=20)
 gate(len(clip)==400
      and len({(x["image_id"],int(x["run"])) for x in clip})==400
      and all(x["g1_choice"] in ("000","111")
              and math.isfinite(float(x["g1_margin"]))
              and x["g1_chosen_good"] in (0,1)
              and (x["label"] not in ("C","I") or
                  x["g1_chosen_good"]==0) for x in clip),
      "A19_XPN_P4_CLIP_400_GOOD")
 p4=summaries(sc,tau)
 maplabels={"G":"G","O_g":"O","O_p":"P","C":"C","I":"I"}
 cleaned=[{**r,"label":maplabels[r["label"]]} for r in clip]
 for r in cleaned:
  g=next(z for z in p4 if z["image_id"]==r["image_id"])
  gate(r["label"]==g["run_labels"][int(r["run"])],
       "A19_XPN_P4_G1_LABEL_AGREEMENT")
 out.mkdir(mode=0o700)
 p=out/"XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv"
 c=out/"XP_N_RUNS_0_3_G1_DECISIONS.csv"
 csv_x(p,p4,COLP)
 csv_x(c,cleaned,COLC)
 # No PSNR/native paths/GT in the public two CSVs or receipt.
 write(out/"XP_N_PREFIX4_RECEIPT.json",dict(
  status="A17_XPN_PREFIX4_CLIP4_CPU_SCORED_AND_SHA_FROZEN",
  expected_head=a.expected_head,n_images=100,n_pilot_roots=400,
  source_pilot_manifest_sha256=digest,
  source_y_manifest_sha256=Y_SHA,source_gt_y_audit_sha256=AUDIT_SHA,
  source_gt_manifest_sha256=GT_SHA,
  frozen_good_q95_tau=tau,GT_only_for_evaluation=True,
  G1_frozen_clip_source=info,
  no_runs_ge_4_read=True,no_GPU_solver_rerun=True,
  local_raw_scores_not_published=True,
  prefix4_sha256=sha(p),clip4_sha256=sha(c),
  pilot_generation_run=str(run)))
 print("A19_XPN_PREFIX4_CLIP4_PASS",
       "PREFIX_SHA",sha(p),"CLIP_SHA",sha(c),"TAU",tau,flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--tasks",required=True)
 ap.add_argument("--run",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A19_XPN_PREFIX4",str(e),flush=True)
  sys.exit(8)
