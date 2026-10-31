#!/usr/bin/env python3
"""Independent A16 CP3 readout from immutable CPU artifacts; no new inference.

Audits official native-float piq results and joins stage panels using
actual image ids and frozen measurement hashes. Separates descriptive
near-nominal compute comparisons from the preregistered M=2 protocol.
Never rebrands PC-DAPS as exact NFE matched and NEVER authorizes CP3.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import jread,sha,csvload,write
from scripts.track_r.xp_a15_cp3_protocol_cpu import verify

STAGE_ORDER=("G4","G5","G3")
PROTOCOL_STAMP="20261030T210259Z"
PIQ_STAMP="20261031T030003Z"
NFE_STAMP="20261031T024828Z"
EXPECTED_PIQ_HEAD="a898024eae69158ab2d072710e54fe827688d31e"
EXPECTED_NFE_HEAD="b6ec9bc8a51f149aeef08e3a3e89be7b9f2b0414"
WEIGHT_SHA={
 "pretrained_lpips_weights_sha256":"389818252bd0a5d9e36789d4f3502d82a4d8363773cab976158c224f268eefc3",
 "pretrained_vgg16_weights_sha256":"397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"}
METRIC_KEYS=("SSIM_raw","SSIM_111","LPIPS_VGG_raw","LPIPS_VGG_111")

def near(x,y,key,tol=1e-7):
 gate(math.isfinite(float(x)) and math.isfinite(float(y)) and
      abs(float(x)-float(y)) <= tol,key)
def metric_root(stage):
 return BASE/("xp_a16_cp3_piq_"+stage+"_"+PIQ_STAMP)
def stage_root(stage):
 return BASE/("xp_a15_cp3_"+stage+"_"+PROTOCOL_STAMP)
def read_stage(stage):
 rec,sp,cp,rp,strings,clips,runs=verify(stage_root(stage),stage)
 out=metric_root(stage)
 receipt=jread(out/"XP_A16_CP3_NATIVE_PIQ_METRICS_RECEIPT.json")
 rows=csvload(out/"XP_A16_CP3_NATIVE_PIQ_METRICS.csv")
 gate(receipt["status"]=="XP_A16_CP3_PUBLISHED_PROTOCOL_PIQ_CPU_METRICS_COMPLETE_NOT_CP3"
      and receipt["expected_head"]==EXPECTED_PIQ_HEAD
      and receipt["stage"]==stage
      and receipt["images"]==rec["n_images"]
      and receipt["candidates"]==rec["n_candidates"]
      and receipt["exact_float_native_pt"] is True
      and receipt["GT_used_for_evaluation_only"] is True
      and receipt["CP3_RETURN_AUTHORIZED"] is False
      and receipt["pretrained"]["piq_version"]=="0.8.0"
      and all(receipt["pretrained"][k]==v for k,v in WEIGHT_SHA.items())
      and receipt["input_stage_scoring_receipt_sha256"]==
           sha(stage_root(stage)/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json")
      and receipt["metrics_sha256"]==sha(out/"XP_A16_CP3_NATIVE_PIQ_METRICS.csv"),
      "CP3_READOUT_PINNED_PIQ_RECEIPT_"+stage)
 sc={(r["image_id"],int(r["run"])):r for r in runs}
 cc={(r["image_id"],int(r["run"])):r for r in clips}
 mm={}
 gate(len(sc)==len(cc)==len(rows)==rec["n_candidates"],
      "CP3_READOUT_PANEL_SIZE_"+stage)
 for r in rows:
  key=(r["image_id"],int(r["run"]))
  gate(key not in mm and key in sc and key in cc
       and r["uid"]==sc[key]["uid"]
       and r["cohort"]==sc[key]["cohort"],
       "CP3_READOUT_RECORD_KEY_"+stage)
  for met in METRIC_KEYS:
   gate(math.isfinite(float(r[met])) and
        (-1.000001<=float(r[met])<=1.000001 if met.startswith("SSIM") else
         abs(float(r[met]))<=10.0),
        "CP3_READOUT_METRIC_FINITE_"+met)
  mm[key]=dict(r,run=int(r["run"]),**{
    met:float(r[met]) for met in METRIC_KEYS})
 by=defaultdict(list)
 for r in mm.values():by[(r["cohort"],r["image_id"])].append(r)
 gate(len(by)==rec["n_images"]
      and all(sorted(z["run"] for z in v)==list(range(rec["n_runs_per_image"]))
              for v in by.values()),"CP3_READOUT_IMAGE_COMPLETENESS_"+stage)
 # Independent CSV re-aggregation of frozen producer summaries.
 published={(r["cohort"],r["rule"],r["metric"]):r
            for r in receipt["mean_per_metric"]}
 ours=defaultdict(list)
 for (cohort,iid),group in by.items():
  ordered=sorted(group,key=lambda r:r["run"])
  if stage=="G3":
   for metric in ("SSIM_raw","LPIPS_VGG_raw"):
    ours[(cohort,"PC_NATIVE_METRIC_MEAN_4",metric)].append(
       mean(r[metric] for r in ordered))
  else:
   for k in (4,8):
    for metric,agg in (("SSIM_raw",max),("LPIPS_VGG_raw",min)):
     ours[(cohort,"DAPS_BEST_OF_"+str(k)+"_ORACLE_PER_METRIC",metric)].append(
        agg(r[metric] for r in ordered[:k]))
 gate(set(published)==set(ours),"CP3_READOUT_METRIC_SUMMARY_KEYS_"+stage)
 for key,values in ours.items():
  near(mean(values),published[key]["mean"],"CP3_READOUT_AGG_"+stage,1e-10)
  gate(len(values)==published[key]["n_images"],
       "CP3_READOUT_COHORT_N_"+stage)
 print("CP3_PIQ_INDEPENDENT_FULL_CSV_AGGREGATION_PASS",stage,len(mm),flush=True)
 return dict(stage=stage,receipt=receipt,candidates=sc,
             clips=cc,metrics=mm,image_groups=by,score_rec=rec)

def independently_spot_recompute(stage,data):
 # Explicit independent spot-check using saved float .pt and GT:
 # one fixed image and one native root per stage. This does not run
 # a solver, CLIP or GPU. Full candidate provenance is checked by
 # original stage and metric receipts; this spot-check validates score
 # units/channel/rotation.
 import torch,piq
 from scripts.track_r.xp_a16_cp3_piq_metrics_cpu import (
      original_tensor,true_image)
 from scripts.track_r.xp_a15_cp3_score_cpu import common_sources
 ff,gi,_=common_sources()
 iid=sorted(k[1] for k in data["image_groups"])[0]
 key=(iid,0)
 r=data["candidates"][key]
 raw=original_tensor(r,stage,{})
 gt=true_image(iid,stage,ff,gi)
 x=(raw.clamp(-1,1)+1)/2
 g=(gt.clamp(-1,1)+1)/2
 with torch.inference_mode():
  s0=float(piq.ssim(x,g,data_range=1,reduction="none")[0])
  s1=float(piq.ssim(torch.flip(x,(-2,-1)),g,
                     data_range=1,reduction="none")[0])
  # Load model once per stage, cached-only; neither weights nor source
  # code are changed by the check.
  import torch.hub
  old=torch.hub.load_state_dict_from_url
  def cached(url,*args,**kw):
   p=Path(torch.hub.get_dir())/"checkpoints"/(
       kw.get("file_name") or url.rsplit("/",1)[-1])
   gate(p.is_file(),"CP3_READOUT_NO_PRETRAINED_DOWNLOAD")
   return torch.load(p,map_location="cpu",weights_only=True)
  torch.hub.load_state_dict_from_url=cached
  try:
   lp=piq.LPIPS(replace_pooling=True,reduction="none").eval()
  finally:
   torch.hub.load_state_dict_from_url=old
  p0=float(lp(x,g)[0])
  p1=float(lp(torch.flip(x,(-2,-1)),g)[0])
 for n,v in (("SSIM_raw",s0),("SSIM_111",s1),
             ("LPIPS_VGG_raw",p0),("LPIPS_VGG_111",p1)):
  near(data["metrics"][key][n],v,"CP3_READOUT_SPOT_"+stage+"_"+n,2e-6)
 return dict(stage=stage,image_id=iid,run=0,
             independent_PiQ_SSIM_LPIPS_float_spot_pass=True)

def record(stage_data,idx,flip=False):
 sc=stage_data["candidates"][idx]
 mm=stage_data["metrics"][idx]
 good=20 if stage_data["stage"]=="G5" else 25
 psnr=float(sc["psnr_flip"] if flip else sc["psnr_raw"])
 return dict(
  PSNR_db=psnr,
  GOOD=float(psnr>=good),
  SSIM=float(mm["SSIM_111"] if flip else mm["SSIM_raw"]),
  LPIPS_VGG=float(mm["LPIPS_VGG_111"] if flip else mm["LPIPS_VGG_raw"]),
  orbit_error=float(sc["label"] in ("O_g","O_p")),
  measurement_sha=sc["measurement_sha256"])
def aggregate(series):
 gate(bool(series),"CP3_READOUT_EMPTY_AGG")
 names=("PSNR_db","GOOD","SSIM","LPIPS_VGG","orbit_error")
 return dict(n_images=len(series),**{k:mean(z[k] for z in series) for k in names})
def daps_native_per_metric_oracles(daps):
 # PSNR follows the frozen scored RGB8 metric, while official SSIM/LPIPS
 # come from native float tensors. All are separate per-metric GT oracles,
 # NEVER GT-free selectors.
 ag=defaultdict(list)
 for (cohort,iid),rs in daps["image_groups"].items():
  for k in (4,8):
   scored=[daps["candidates"][(iid,j)] for j in range(k)]
   mm=[daps["metrics"][(iid,j)] for j in range(k)]
   ag[(cohort,k,"PSNR_rgb8_raw_db")].append(
      max(float(row["psnr_raw"]) for row in scored))
   ag[(cohort,k,"SSIM_piq_float_raw")].append(
      max(float(row["SSIM_raw"]) for row in mm))
   ag[(cohort,k,"LPIPS_VGG_piq_float_raw")].append(
      min(float(row["LPIPS_VGG_raw"]) for row in mm))
 result=[]
 for (cohort,k,metric),vals in sorted(ag.items()):
  result.append(dict(cohort=cohort,method=daps["stage"],
      rule="GT_PER_METRIC_ORACLE_BEST_"+str(k),
      metric=metric,n_images=len(vals),mean=mean(vals)))
 return result

def paired_pc_vs_daps(pc,daps):
 pgroup=pc["image_groups"];dgroup=daps["image_groups"]
 gate(set(pgroup)==set(dgroup) and len(pgroup)==180,
      "CP3_READOUT_PC_DAPS_IDENTICAL_180_IDS")
 scores=defaultdict(list)
 overlap=[]
 for (cohort,iid) in sorted(pgroup):
  pidx=(iid,0); didx=(iid,0)
  p0=[record(pc,(iid,k)) for k in range(4)]
  d0=record(daps,didx)
  gate(len({x["measurement_sha"] for x in p0})==1
       and p0[0]["measurement_sha"]==d0["measurement_sha"],
       "CP3_READOUT_PC_DAPS_IDENTICAL_FROZEN_SIGNED_Y_"+iid)
  co=pc["clips"]
  best=min(range(4),key=lambda k:
       (float(pc["candidates"][(iid,k)]["rho_signed"]),k))
  flip=co[(iid,best)]["face_choice"]=="111"
  pc_e0=record(pc,(iid,best))
  pc_e1=record(pc,(iid,best),flip=flip)
  native=dict(PSNR_db=mean(z["PSNR_db"] for z in p0),
      GOOD=mean(z["GOOD"] for z in p0),
      SSIM=mean(z["SSIM"] for z in p0),
      LPIPS_VGG=mean(z["LPIPS_VGG"] for z in p0),
      orbit_error=mean(z["orbit_error"] for z in p0))
  scores[(cohort,"PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN")].append(native)
  scores[(cohort,"PC_E0_RESIDUAL_SELECTION")].append(pc_e0)
  scores[(cohort,"PC_E1_CLIP_ORIENTATION")].append(pc_e1)
  scores[(cohort,"DAPS_4K_ONE_RUN")].append(d0)
  overlap.append(iid)
 for policy in ("PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN",
                "PC_E0_RESIDUAL_SELECTION",
                "PC_E1_CLIP_ORIENTATION","DAPS_4K_ONE_RUN"):
  full=[v for (c,n),arr in scores.items() if n==policy for v in arr]
  gate(len(full)==180,"CP3_READOUT_FULL_180_"+policy)
  scores[("ALL",policy)]=full
 table=[dict(cohort=c,policy=p,**aggregate(v)) for (c,p),v in sorted(scores.items())]
 return dict(
  status="PC_DAPS_VS_DAPS_ONE_RUN_DESCRIPTIVE_NEAR_NOMINAL_COMPUTE_ONLY",
  same_images=True,same_signed_measurements_verified=True,
  n_images=len(overlap),n_unique_images=len(set(overlap)),
  accounting="PC-DAPS nominal bag 1920 plus incompletely logged pruning; DAPS-4K run 2000",
  frozen_M2_protocol_applied_to_one_run=False,
  not_exact_NFE_matched=True,
  table=table)

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_READOUT_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(["git","-C",str(wt),*args],
     text=True,timeout=60).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CP3_READOUT_HEAD")
 nfe_file=BASE/("xp_a16_cp3_nfe_"+NFE_STAMP)/"XP_A16_CP3_NFE_LEDGER.json"
 ident_file=BASE/("xp_a16_cp3_gt_y_"+NFE_STAMP)/"XP_A16_CP3_IMAGENET_GT_Y_IDENTITY.json"
 audit_file=BASE/("xp_a15_cp3_protocol_"+PROTOCOL_STAMP)/"XP_A15_CP3_INDEPENDENT_AUDIT.json"
 nfe=jread(nfe_file);ident=jread(ident_file);audit=jread(audit_file)
 gate(nfe["status"]=="XP_A16_CP3_PC_NFE_NOMINAL_AUDIT_ONLY_NOT_MATCHED_COMPUTE"
      and nfe["expected_head"]==EXPECTED_NFE_HEAD
      and nfe["exact_pc_nfe_available"] is False
      and nfe["matched_nfe_certified"] is False
      and nfe["bags"]==180
      and nfe["config"]["nominal_per_bag_particle_equivalent_calls"]==1920
      and nfe["comparison"]["DAPS_historical_400x5_nominal_nfe_per_run"]==2000
      and ident["status"]=="XP_A16_CP3_IMAGENET_GT_Y_EXACT_100_PASS"
      and ident["n_images"]==100
      and ident["expected_head"]==EXPECTED_NFE_HEAD
      and ident["worst_rms"]<=1e-4
      and audit["status"]=="XP_A15_CP3_INDEPENDENT_SCORES_PROTOCOL_AUDIT_PASS_NOT_FINAL"
      and len(audit["frozen_protocol_exact_reproductions"])==7,
      "CP3_READOUT_SCIENCE_RECEIPTS")
 data={k:read_stage(k) for k in STAGE_ORDER}
 if a.spot_checks:
  import torch
  torch.set_num_threads(4)
  spots=[independently_spot_recompute(k,data[k]) for k in STAGE_ORDER]
 else:spots=[]
 paired=paired_pc_vs_daps(data["G3"],data["G4"])
 docdir=BASE/("xp_a15_cp3_protocol_"+PROTOCOL_STAMP)
 pp={s:jread(docdir/("CP3_"+s.replace(".","p")+"_PROTOCOL.json"))
     for s in ("G2_0.02","G2_0.10","G4","G5")}
 hg={s:dict(HG1=bool(pp[s]["H_G1_pass"]),HG2=bool(pp[s]["H_G2_pass"]),
            tau_scaled=float(pp[s]["H_G1_tau_scaled"]),
            tau_gt=float(pp[s]["H_G1_tau_gt"]),
            rel_tau_error=abs(pp[s]["H_G1_tau_scaled"]-pp[s]["H_G1_tau_gt"])/
                          pp[s]["H_G1_tau_gt"],
            E0_success=float(pp[s]["weighted"]["success_all"]),
            mean_runs=float(pp[s]["weighted"]["runs_mean"]))
     for s in ("G2_0.02","G2_0.10")}
 h1={s:dict(pass_H1=pp[s]["H1"],success=pp[s]["weighted"]["success_all"],
             mean_runs=pp[s]["weighted"]["runs_mean"],
             gain_vs_equal_NFE_baseline=pp[s]["weighted"]["gain_vs_fixed_equal_runs"])
     for s in ("G4","G5")}
 gate(all(not x["HG1"] and x["HG2"] for x in hg.values())
      and all(x["pass_H1"] for x in h1.values()),
      "CP3_READOUT_PRE_REGISTERED_H_TEST_IDENTITY")
 pc_min=mean(z["minimum_nfe"] for z in nfe["per_bag"])
 gate(nfe["observed_prune_events_sum"]==126
      and abs(pc_min-(1920+126*12/180))<1e-10,
      "CP3_READOUT_NFE_LOWER_BOUND")
 audit_summary={
  "status":"XP_A16_CP3_INDEPENDENT_JOINT_READOUT_PASS_UNMATCHED_NFE",
  "expected_head":a.expected_head,
  "piq_stages":{s:dict(n_images=data[s]["score_rec"]["n_images"],
         n_candidates=data[s]["score_rec"]["n_candidates"],
         metric_csv_sha256=data[s]["receipt"]["metrics_sha256"],
         metric_receipt_sha256=sha(metric_root(s)/"XP_A16_CP3_NATIVE_PIQ_METRICS_RECEIPT.json"),
         pretrained_weights=data[s]["receipt"]["pretrained"])
         for s in STAGE_ORDER},
  "original_cp3_protocol_audit_sha256":sha(audit_file),
  "nfe_ledger_sha256":sha(nfe_file),
  "gt_y_identity_sha256":sha(ident_file),
  "native_float_piq_spot_checks":spots,
  "hypotheses":{"G2_noise":hg,"DAPS_4K":h1},
  "nfe":{
    "pc_nominal_per_bag":1920,
    "daps_one_run_nominal":2000,
    "observed_native_prune_events":126,
    "pc_mean_minimum_nfe_from_logged_events":pc_min,
    "per_bag_exact_nfe_unavailable":True,
    "frozen_M2_budget_at_least_two_DAPS_runs":True,
    "valid_near_nominal_compute_descriptive_only":True,
    "exact_matched_NFE":False
  },
  "pc_daps_vs_daps_same_180_frozen_y":paired,
  "daps_GT_per_metric_oracles_not_GT_free":{
    stage:daps_native_per_metric_oracles(data[stage])
    for stage in ("G4","G5")
  },
  "important_caveat":"DAPS one-run comparison is NOT the frozen M=2 stopping policy; PC-DAPS has unknown extra pruning forward calls; no exact compute matching",
  "GT_used_evaluation_only":True,"new_solver_calls":0,
  "new_CLIP_calls":0,"new_GPU_calls":0,"CP3_RETURN_AUTHORIZED":False,
  "sanitized_publication_pending":True
 }
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a16_cp3_joint_")
      and not out.exists() and not out.is_symlink(),
      "CP3_READOUT_NEW_OUT")
 out.mkdir(mode=0o700)
 result=out/"XP_A16_CP3_JOINT_READOUT.json"
 write(result,audit_summary)
 print("CP3_JOINT_READOUT_AUDIT_PASS",out,flush=True)
 for s,x in hg.items():
  print("CP3_HG",s,"H1",x["HG1"],"H2",x["HG2"],
        "tau_scaled",round(x["tau_scaled"],8),
        "tau_gt",round(x["tau_gt"],8),flush=True)
 for s,x in h1.items():
  print("CP3_H1_DAPS",s,x["pass_H1"],
        "success",round(x["success"],6),
        "gain",round(x["gain_vs_equal_NFE_baseline"],6),flush=True)
 for z in paired["table"]:
  if z["cohort"]=="ALL":
   print("CP3_PC_DAPS_NEAR_NOMINAL",z["policy"],
       "PSNR",round(z["PSNR_db"],4),
       "SSIM",round(z["SSIM"],4),
       "LPIPS_VGG",round(z["LPIPS_VGG"],4),
       "GOOD25",round(z["GOOD"],4),
       "orbit",round(z["orbit_error"],4),flush=True)
 for stage in ("G4","G5"):
  for z in audit_summary["daps_GT_per_metric_oracles_not_GT_free"][stage]:
   if z["rule"]=="GT_PER_METRIC_ORACLE_BEST_4":
    print("CP3_DAPS_ORACLE_BEST4",stage,z["cohort"],z["metric"],
          round(z["mean"],6),flush=True)
 print("CP3_JOINT_PC_NOT_EXACT_NFE_MATCHED__NO_PUBLICATION_GATE",flush=True)
 print("CP3_JOINT_READOUT_JSON",result,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--spot-checks",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A16_CP3_JOINT_READOUT",e,flush=True)
  sys.exit(8)
