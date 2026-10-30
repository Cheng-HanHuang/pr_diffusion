#!/usr/bin/env python3
"""A15 CP3 GT-only scoring of frozen A14 G2/G4/G5 and native G3.

Each stage runs independently; no new inverse-solving GPU. Native
receipts and model/y hashes gate GT access. Scores PSNR raw/orbit and
rho_signed, derives per-family run0..3 GOOD rho q95, and applies frozen
face/G1 CLIP. Local candidate-level scores stay off GitHub. CP3 is
NOT published by this stage; separate protocol, NFE and forecast gates.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a12_g45_prepare_cpu import prep as source_g45,STAGES
from scripts.track_r.xp_a13_f8_score_cpu import r07_source
from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from prdiffusion.track_r.physics import to_rgb8,from_rgb8,rho_signed,load_png_rgb8
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from scripts.track_r.xp_a15_cp2_d4_cpu import (
 safe,sha,jread,csvwrite,write,clip_new)

A14_HEAD="0acb0081d992405f034f246318c3c5dbc6e3282f"
G3_HEAD="75722c92de4d0489dd202eeea860075f857bffb9"
A14_RUN=BASE/"xp_a12_campaign_prod_20261030T022725Z"
G3_RUN=BASE/"xp_a12_g3_prod_20261030T161425Z"
G3_PREP=BASE/"xp_a12_g3_prepare_20261030T153655Z"
G3_MANIFEST_SHA="3ad52d84216ccd57c14d8b59c3a643a9b89f59adfce908dac2f43d4d1639ff84"
STAGE_CHOICES=("G2_0.02","G2_0.10","G4","G5","G3")
BASE_COLS=("image_id","cohort","labels_by_run","rho_signed_by_run",
  "tau_signed_good_q95","n_G","n_O_g","n_O_p","n_C","n_I")
RUN_COLS=("image_id","cohort","run","label","rho_signed","psnr_raw",
          "psnr_orbit_best","flip_good","reconstruction_path",
          "reconstruction_sha256","uid","measurement_sha256")
SRC_COH={"S1_b22":"B22","S2_b24_dev80":"DEV80"}
def find_prep(prefix,receipt_key,stage,manifest_expected):
 found=[]
 for p in BASE.glob(prefix):
  if not p.is_dir() or p.is_symlink():continue
  if not (p/receipt_key).is_file():continue
  rec=jread(p/receipt_key)
  if stage=="G2":
   x=rec.get("per_sigma",{}).get(manifest_expected[0],{})
   if x.get("sha256")==manifest_expected[1] and rec.get("head")==A14_HEAD:
    found.append(p)
  else:
   x=rec.get("stages",{}).get(stage+"_PROD",{})
   if x.get("sha256")==manifest_expected and rec.get("head")==A14_HEAD:
    found.append(p)
 gate(len(found)==1,"CP3_UNIQUE_PREP_"+stage)
 return found[0]
def common_sources():
 _,ident,_=r07_source()
 gt_ff={}
 for (cohort,iid),(path,fmt,ypath,ysha) in ident.items():
  gt_ff[f"{cohort}_{iid}"]=dict(gt_path=path,gt_format=fmt,
      source=cohort,image_id=iid,measurement_path=ypath,measurement_sha256=ysha)
 gate(len(gt_ff)==180,"CP3_FFHQ_180_GT")
 i,j=(IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json",
      IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json")
 gate(sha(j)==COMMON_Y_SHA,"CP3_IMAGENET_Y_SHA")
 src=jread(i);y=jread(j)
 gt_in={r["basename"]:r for r in src["rows"]}
 yy={r["basename"]:r for r in y["rows"]}
 gate(len(gt_in)==len(yy)==100 and set(gt_in)==set(yy),
      "CP3_IMAGENET_100_GT")
 return gt_ff,gt_in,yy
def ff_key(cohort,iid):
 if cohort in SRC_COH:return cohort+"_"+str(iid).zfill(5)
 if str(iid).startswith("S1_b22_"):
  return str(iid)
 if str(iid).startswith("S2_b24_dev80_"):
  return str(iid)
 raise RuntimeError("STOP_CP3_UNRECOGNIZED_EXPOSED_FFHQ_COHORT")
def verify_native_row(kind,run,row,manifest_sha,head,phase=None):
 uid=row["uid"];token=hashlib.sha256(uid.encode()).hexdigest()[:24]
 if kind=="G3":
  dest=run/"roots"/("bag_"+token)
  receipt=jread(dest/"XP_A12_G3_BAG_COMPLETE.json")
  gate(receipt["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS"
       and receipt["runner_head"]==head
       and receipt["uid"]==uid and receipt["manifest_sha256"]==manifest_sha
       and receipt["measurement_sha256"]==row["measurement_sha256"]
       and receipt["bag_size"]==4 and len(receipt["particle_pngs"])==4
       and receipt["reported_metric_rule"]=="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN"
       and receipt["no_ground_truth_access"] is True
       and sha(dest/"particles.pt")==receipt["particles_pt_sha256"],
       "CP3_G3_BAG_RECEIPT")
  for i,p in enumerate(receipt["particle_pngs"]):
   path=dest/f"particle_{i}.png"
   gate(Path(p["path"])==path and sha(path)==p["sha256"],
        "CP3_G3_PARTICLE_SHA")
  return [dict(rep=i,uid=uid+"|particle"+str(i),
     image_id=row["image_id"],cohort=row["cohort"],
     measurement_path=row["measurement_path"],
     measurement_sha256=row["measurement_sha256"],
     png=dest/f"particle_{i}.png",hash=receipt["particle_pngs"][i]["sha256"])
     for i in range(4)]
 dest=(run/"gpu"/("root_"+token) if phase=="HIST" else
       run/kind/"PROD"/"gpu"/("root_"+token))
 proof=jread(dest/"XP_T_DAPS4K_ROOT_COMPLETE.json")
 gate(proof["status"]=="XP_T_DAPS4K_ROOT_PASS"
      and proof["uid"]==uid and proof["measurement_sha256"]==row["measurement_sha256"]
      and proof["manifest_sha256"]==manifest_sha
      and proof["gt_free_generation"] is True
      and proof["model_load_strict"] is True
      and sha(dest/"reconstruction.png")==proof["reconstruction_png_sha256"]
      and sha(dest/"reconstruction.pt")==proof["reconstruction_pt_sha256"],
      "CP3_DAPS_NATIVE_RECEIPT")
 if phase!="HIST":gate(proof["runner_head"]==head,"CP3_DAPS_NEW_HEAD")
 return [dict(rep=row["rep"],uid=uid,image_id=row["image_id"],
    measurement_path=row["measurement_path"],
    measurement_sha256=row["measurement_sha256"],
    png=dest/"reconstruction.png",hash=proof["reconstruction_png_sha256"])]
def stage_records(stage):
 if stage.startswith("G2_"):
  sig=stage.split("_",1)[1]
  gate(sig in ("0.02","0.10"),"CP3_SIGMA")
  root=A14_RUN/"G2"/("sigma_"+sig)
  # Identify only manifest frozen with the committed A14 runner.
  manifests=[]
  for p in BASE.glob("xp_a12_g2_prepare_*/XP_A12_G2_PREP_RECEIPT.json"):
   doc=jread(p)
   if doc.get("head")==A14_HEAD and doc.get("status")=="XP_A12_G2_INPUTS_FROZEN_CPU_PASS":
    manifests.append((p.parent,doc))
  gate(len(manifests)==1,"CP3_G2_FROZEN_PREP_UNIQUE")
  p,doc=manifests[0]
  mf=p/("XP_A12_G2_SIGMA_"+sig+"_TASKS.json")
  # Actual manifest path from frozen receipt, no reconstructed guess.
  mf=Path(doc["per_sigma"][sig]["path"])
  gate(sha(mf)==doc["per_sigma"][sig]["sha256"],"CP3_G2_MANIFEST_SHA")
  manifest=jread(mf);rows=manifest["rows"]
  gate(len(rows)==1440 and manifest["sigma"]==float(sig)
       and manifest["gt_paths_supplied"] is False,"CP3_G2_PANEL")
  seen=set();output=[]
  for i,row in enumerate(rows):
   chunk=root/f"chunk_{(i//8)*8:05d}_008"
   folder=root_folder(chunk,row)
   rec=validate_complete(folder,row,A14_HEAD,sha(mf))
   gate(rec["gt_access"] is False,"CP3_G2_GT_FREE_NATIVE")
   key=(row["image_id"],row["rep"])
   gate(key not in seen,"CP3_G2_DUP")
   seen.add(key)
   output.append(dict(rep=row["rep"],uid=row["uid"],image_id=row["image_id"],
       measurement_path=row["measurement_path"],
       measurement_sha256=row["measurement_sha256"],
       png=folder/"reconstruction.png",hash=rec["reconstruction_png_sha256"]))
  gate(len(output)==1440 and len(seen)==1440,"CP3_G2_1440")
  return output,dict(stage=stage,manifest_sha256=sha(mf),native_head=A14_HEAD,
          sigma=float(sig),expected_images=180,expected_runs=8)
 if stage in ("G4","G5"):
  original,oldpath,new,replay,cap=source_g45(stage)
  # The exact A14 CPU prep is discovered from its preserved receipt.
  found=[]
  for p in BASE.glob("xp_a12_g45_prepare_*/XP_A12_G45_PREP_RECEIPT.json"):
   doc=jread(p)
   if doc.get("head")==A14_HEAD and doc.get("status")=="XP_A12_G45_INPUTS_FROZEN_CPU_PASS":
    found.append((p.parent,doc))
  gate(len(found)==1,"CP3_G45_PREP_UNIQUE")
  pr,doc=found[0]
  mf=pr/f"XP_A12_{stage}_PROD_TASKS.json"
  sha_new=sha(mf)
  gate(jread(mf)["rows"]==new and
       doc["stages"][stage+"_PROD"]["sha256"]==sha_new,
       "CP3_G45_FROZEN_NEW")
  oldrun=STAGES[stage][1]
  rows=[]
  for r in original["rows"]:
   rows+=verify_native_row(stage,oldrun,r,sha(oldpath),original["head"],phase="HIST")
  for r in new:
   rows+=verify_native_row(stage,A14_RUN,r,sha_new,A14_HEAD,phase="PROD")
  n=180 if stage=="G4" else 100
  gate(len(rows)==n*8,"CP3_G45_FULL_EIGHT")
  return rows,dict(stage=stage,manifest_sha256=sha_new,
    historical_manifest_sha256=sha(oldpath),native_head=A14_HEAD,
    expected_images=n,expected_runs=8)
 if stage=="G3":
  mf=G3_PREP/"XP_A12_G3_180_BAGS.json"
  gate(sha(mf)==G3_MANIFEST_SHA,"CP3_G3_FROZEN_MANIFEST")
  m=jread(mf)
  done=jread(G3_RUN/"XP_A14_G3_FINAL_STATUS.json")
  gate(done["status"]=="XP_A14_G3_NATIVE_FULL180_GENERATION_COMPLETE"
       and done["verified_bags"]==180
       and done["source_head"]==G3_HEAD
       and done["prep_manifest_sha256"]==G3_MANIFEST_SHA,
       "CP3_G3_FINAL_STATUS")
  rows=[]
  for r in m["rows"]:
   rows+=verify_native_row("G3",G3_RUN,r,sha(mf),G3_HEAD)
  gate(len(rows)==720,"CP3_G3_720")
  return rows,dict(stage="G3",manifest_sha256=sha(mf),
    native_head=G3_HEAD,expected_images=180,expected_runs=4,
    native_method_reporting="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN",
    no_claim_of_pixel_mean_or_matched_NFE=True)
 raise RuntimeError("STOP_CP3_UNKNOWN_STAGE")
def score(stage,raw,meta,out):
 import torch
 from PIL import Image
 from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
 torch.set_num_threads(4)
 ff,gtimg,imgy=common_sources()
 tf=Compose([ToTensor(),Resize(256),CenterCrop(256)])
 groups=defaultdict(list)
 for row in raw:
  iid=row["image_id"]
  key=ff_key(row.get("cohort"),iid) if stage!="G5" else iid
  groups[key].append(row)
 n=meta["expected_runs"];expected=meta["expected_images"]
 gate(len(groups)==expected and
      all(sorted(r["rep"] for r in v)==list(range(n)) for v in groups.values()),
      "CP3_IMAGE_RUN_GRID")
 tmp=[]
 for index,(key,rr) in enumerate(sorted(groups.items())):
  rr.sort(key=lambda r:r["rep"])
  if stage=="G5":
   g=gtimg[key];yr=imgy[key]
   gate(g["source_sha256"]==yr["source_image_sha256"]
        and sha(g["source_path"])==g["source_sha256"],
        "CP3_IMAGENET_GT_SHA")
   with Image.open(safe(g["source_path"])) as im:
    gt=to_rgb8(tf(im.convert("RGB")).unsqueeze(0)*2-1)
   cohort="IN";goal=20.
  else:
   g=ff[key]
   gate(sha(g["gt_path"]) and
        g["image_id"]==str(g["image_id"]).zfill(5),"CP3_FFHQ_GT_SOURCE")
   gt=to_rgb8(gt_source(dict(gt_path=g["gt_path"],
              gt_format=g["gt_format"])).float())
   cohort=SRC_COH[g["source"]];goal=25.
  ypath=rr[0]["measurement_path"]
  ysha=rr[0]["measurement_sha256"]
  gate(all((r["measurement_path"],r["measurement_sha256"])==(ypath,ysha)
           for r in rr) and sha(ypath)==ysha,"CP3_Y_IDENTITY")
  y=read_tensor(safe(ypath),("measurement","y","observation"),(1,3,384,384))
  for r in rr:
   path=safe(r["png"])
   gate(sha(path)==r["hash"],"CP3_PNG_SHA")
   x8=load_png_rgb8(path)
   orb=orbit_psnr_channel_mse(x8,gt)
   best=max(sorted(orb),key=lambda s:orb[s])
   rawpsnr=float(orb["000"]);bestpsnr=float(orb[best])
   rho=float(rho_signed(from_rgb8(x8),y)[0])
   gate(all(math.isfinite(v) for v in (rawpsnr,bestpsnr,rho))
        and rho>=0,"CP3_SCORES_FINITE")
   tmp.append(dict(image_id=key,cohort=cohort,run=r["rep"],
      raw=rawpsnr,orbit=bestpsnr,best=best,rho=rho,
      flip_good=int(float(orb["111"])>=goal),
      reconstruction_path=str(path),reconstruction_sha256=r["hash"],
      uid=r["uid"],measurement_sha256=ysha))
  if (index+1)%10==0:
   print("CP3_"+stage+"_IMAGES_SCORED",index+1,"/",expected,flush=True)
 # Each family/settings/dataset computes tau on run 0..3 only.
 # FFHQ cohorts retain separate calibrations. G2 sigma is also separate.
 tau={}
 for cohort in sorted({r["cohort"] for r in tmp}):
  calibration=[r["rho"] for r in tmp
               if r["cohort"]==cohort and r["run"]<4
               and r["raw"]>=(20 if stage=="G5" else 25)]
  gate(bool(calibration),"CP3_NO_GOOD_RUNS_TO_CALIBRATE_TAU_"+cohort)
  tau[cohort]=float(np.quantile(calibration,.95,method="linear"))
  gate(tau[cohort]>0 and math.isfinite(tau[cohort]),"CP3_TAU_FINITE")
 import numpy as np
 final=[]
 for r in tmp:
  goal=20 if stage=="G5" else 25
  label=("G" if r["raw"]>=goal else
         "O_g" if r["orbit"]>=goal and r["best"]=="111" else
         "O_p" if r["orbit"]>=goal else
         "C" if r["rho"]<=tau[r["cohort"]] else "I")
  final.append(dict(image_id=r["image_id"],cohort=r["cohort"],
    run=r["run"],label=label,rho_signed=format(r["rho"],".15g"),
    psnr_raw=format(r["raw"],".12g"),
    psnr_orbit_best=format(r["orbit"],".12g"),
    flip_good=r["flip_good"],
    reconstruction_path=r["reconstruction_path"],
    reconstruction_sha256=r["reconstruction_sha256"],
    uid=r["uid"],measurement_sha256=r["measurement_sha256"]))
 output=[]
 for key,rr in sorted(groups.items()):
  group=sorted((r for r in final if r["image_id"]==key),key=lambda r:r["run"])
  gate([r["run"] for r in group]==list(range(n)),"CP3_LABEL_PANEL")
  cc=Counter(r["label"] for r in group)
  output.append(dict(image_id=key,cohort=group[0]["cohort"],
     labels_by_run=";".join(r["label"] for r in group),
     rho_signed_by_run=";".join(r["rho_signed"] for r in group),
     tau_signed_good_q95=format(tau[group[0]["cohort"]],".17g"),
     n_G=cc["G"],n_O_g=cc["O_g"],n_O_p=cc["O_p"],n_C=cc["C"],n_I=cc["I"]))
 csvwrite(out/"CP3_LOCAL_CANDIDATES.csv",final,RUN_COLS)
 csvwrite(out/"CP3_A9_STRINGS.csv",output,BASE_COLS)
 mode="g1" if stage=="G5" else "face"
 cliprows,clipmeta=clip_new(final,mode=mode,threshold=(20 if stage=="G5" else 25))
 for r in cliprows:
  r["cohort"]=next(x["cohort"] for x in final
                   if x["image_id"]==r["image_id"] and x["run"]==r["run"])
 clip_cols=("image_id","cohort","run","label",
            mode+"_choice",mode+"_margin",mode+"_chosen_good")
 csvwrite(out/"CP3_CLIP_DECISIONS.csv",cliprows,clip_cols)
 result=dict(status="XP_A15_CP3_STAGE_CPU_SCORED_NOT_CHECKPOINT",
   stage=stage,expected_head=a.expected_head,
   n_images=expected,n_candidates=len(final),
   n_runs_per_image=n,tau_by_cohort=tau,clip_mode=mode,
   clip_source=clipmeta,root_provenance=meta,
   labels=dict(Counter(r["label"] for r in final)),
   outcome_GT_for_evaluation_only=True,
   no_GPU=True,science_settings_unchanged=True,
   SHA256SUMS={n:sha(out/n) for n in (
      "CP3_LOCAL_CANDIDATES.csv","CP3_A9_STRINGS.csv",
      "CP3_CLIP_DECISIONS.csv")},
   CP3_released=False,protocol_analysis_pending=True,
   matching_NFE_analysis_pending=stage=="G3")
 write(out/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json",result)
 print("CP3_STAGE_NATIVE_SCORING_PASS",stage,len(final),flush=True)
 print("CP3_STAGE_DIR",out,flush=True)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(["git","-C",str(wt),*args],text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CP3_WORKTREE")
 rows,meta=stage_records(a.stage)
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a15_cp3_"+a.stage.replace(".","p")+"_")
      and not out.exists(),"CP3_OUTPUT_PATH")
 if a.preflight_only:
  print("CP3_FROZEN_NATIVE_RECEIPTS_PREFLIGHT_PASS",a.stage,
        len(rows),flush=True)
  return
 out.mkdir(mode=0o700)
 score(a.stage,rows,meta,out)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--stage",choices=STAGE_CHOICES,required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A15_CP3_STAGE",e,flush=True);sys.exit(8)
