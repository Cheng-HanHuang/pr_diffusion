#!/usr/bin/env python3
"""A16 CP3 PC-DAPS denoiser accounting; no solver run, no GT, no GPU.

The 4-particle nominal count is the number of per-particle calls implied
by frozen EDM schedules. Additional prior-consistency calls during
pruning are logged incompletely, hence exact matched-NFE is NOT certified.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_a15_cp3_score_cpu import (
 G3_RUN,G3_PREP,G3_MANIFEST_SHA,G3_HEAD)

PC=ROOT/"external/PC_DAPS_xp"
PC_HEAD="59b2f046"
STAGE="G3"
STAMP="20261030T210259Z"
def frozen_config():
 import yaml
 source=PC/"configs/sampler/edm_pcdaps.yaml"
 gate(source.is_file() and not source.is_symlink(),"NFE_PC_NATIVE_CONFIG")
 d=yaml.safe_load(source.read_text())
 a=d["annealing_scheduler_config"]
 b=d["diffusion_scheduler_config"]
 c=d["pcdaps_config"]
 gate(a["name"]==b["name"]=="edm"
      and a["num_steps"]==200 and b["num_steps"]==5
      and c["bag_size"]==4 and c["ode_steps_high_noise"]==1
      and c["ode_switch_ratio"]==.65
      and c["skip_final_langevin"] is True
      and c["prune"] is True and c["score_num_samples"]==1
      and [float(z) for z in c["score_sigmas"]]==[.5,1.,2.],
      "NFE_PINNED_CONFIG_CHANGED")
 s=(PC/"sampler.py").read_text()
 scheduler=(PC/"cores/scheduler.py").read_text()
 # Contracts certifying the nominal counting semantics, rather than
 # blindly multiplying YAML numbers.
 gate("self.num_steps = num_steps + 1" in scheduler
      and "for t, t_next in zip(time_steps[:-1], time_steps[1:])" in s
      and "denoised = model.tweedie(x / st, sigma_t)" in s
      and "self.get_ode_steps(step, num_steps)" in s
      and "return self.ode_steps_high_noise" in s,
      "NFE_SAMPLER_ACCOUNTING_SOURCE_DRIFT")
 nsteps=int(a["num_steps"])
 high=sum(j<c["ode_switch_ratio"]*nsteps for j in range(nsteps))
 low=nsteps-high
 ode_high=int(c["ode_steps_high_noise"])
 ode_low=int(b["num_steps"])
 per_particle=high*ode_high+low*ode_low
 nominal_per_bag=per_particle*int(c["bag_size"])
 gate((high,low,per_particle,nominal_per_bag)==
      (130,70,480,1920),"NFE_NOMINAL_CONTRACT")
 return dict(source_sha256=sha(source),sampler_sha256=sha(PC/"sampler.py"),
     scheduler_sha256=sha(PC/"cores/scheduler.py"),
     annealing_steps=nsteps,high_noise_steps=high,low_noise_steps=low,
     ode_steps_high=ode_high,ode_steps_low=ode_low,
     particles_per_bag=4,nominal_tweedie_calls_per_particle=per_particle,
     nominal_per_bag_particle_equivalent_calls=nominal_per_bag,
     extra_pruning_prior_consistency_calls_included=False)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_NFE_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda p,*q:subprocess.check_output(["git","-C",str(p),*q],
           text=True,timeout=60).strip()
 gate(git(wt,"rev-parse","HEAD")==a.expected_head
      and not git(wt,"status","--porcelain"),"CP3_NFE_EXECUTOR_HEAD")
 gate(PC.is_dir() and git(PC,"rev-parse","HEAD").startswith(PC_HEAD)
      and not git(PC,"status","--porcelain","--untracked-files=no"),
      "CP3_NFE_NATIVE_PIN")
 cfg=frozen_config()
 mf=G3_PREP/"XP_A12_G3_180_BAGS.json"
 gate(sha(mf)==G3_MANIFEST_SHA,"CP3_NFE_BAG_MANIFEST_SHA")
 source=jread(mf)
 gate(source["status"]=="XP_A12_G3_180_NATIVE_BAGS_FROZEN"
      and source["bag_size"]==4 and len(source["rows"])==180
      and source["gt_paths_supplied"] is False,
      "CP3_NFE_BAG_MANIFEST")
 done=jread(G3_RUN/"XP_A14_G3_FINAL_STATUS.json")
 gate(done["status"]=="XP_A14_G3_NATIVE_FULL180_GENERATION_COMPLETE"
      and done["verified_bags"]==180
      and done["source_head"]==G3_HEAD
      and done["prep_manifest_sha256"]==G3_MANIFEST_SHA,
      "CP3_NFE_FINAL_SOURCE")
 bags=[]
 for row in source["rows"]:
  uid=row["uid"]
  t=hashlib.sha256(uid.encode()).hexdigest()[:24]
  folder=G3_RUN/"roots"/("bag_"+t)
  receipt=jread(folder/"XP_A12_G3_BAG_COMPLETE.json")
  gate(receipt["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS"
       and receipt["uid"]==uid and receipt["runner_head"]==G3_HEAD
       and receipt["manifest_sha256"]==G3_MANIFEST_SHA
       and receipt["bag_size"]==4
       and receipt["no_ground_truth_access"] is True
       and receipt["reported_metric_rule"]=="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN"
       and sha(folder/"particles.pt")==receipt["particles_pt_sha256"],
       "CP3_NFE_BAG_RECEIPT_"+uid)
  n_events=receipt["native_prune_events"]
  gate(type(n_events) is int and n_events>=0,"CP3_NFE_PRUNE_COUNT")
  # Each logged event invokes a minimum of 3 scales × 4 particles.
  # An unlogged 'keep all and retry' also invokes the prior network.
  # Therefore the real NFE cannot be reconstructed from this receipt.
  min_extra=n_events*len([.5,1.,2.])*4
  bags.append(dict(uid=uid,image_id=row["image_id"],
     native_prune_events=n_events,
     nominal_nfe=cfg["nominal_per_bag_particle_equivalent_calls"],
     known_pruning_lower_bound=min_extra,
     minimum_nfe=cfg["nominal_per_bag_particle_equivalent_calls"]+min_extra,
     exact_total_nfe=None))
 gate(len(bags)==180 and len(set(r["uid"] for r in bags))==180,
      "CP3_NFE_FULL180")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a16_cp3_nfe_")
      and not out.exists() and not out.is_symlink(),
      "CP3_NFE_NEW_OUTPUT")
 if a.preflight_only:
  print("CP3_PC_NFE_READONLY_PREFLIGHT_PASS",
        len(bags),"nominal",cfg["nominal_per_bag_particle_equivalent_calls"],
        flush=True)
  return
 out.mkdir(mode=0o700)
 from statistics import mean
 report=dict(
  status="XP_A16_CP3_PC_NFE_NOMINAL_AUDIT_ONLY_NOT_MATCHED_COMPUTE",
  expected_head=a.expected_head,
  native_pc_daps_commit=git(PC,"rev-parse","HEAD"),
  native_runner_head=G3_HEAD,
  method="PC-DAPS",dataset="FFHQ",bags=180,
  config=cfg,
  comparison={
    "DAPS_historical_400x5_nominal_nfe_per_run":2000,
    "PC_DAPS_nominal_nfe_per_bag":cfg["nominal_per_bag_particle_equivalent_calls"],
    "nominal_PC_to_DAPS_ratio":cfg["nominal_per_bag_particle_equivalent_calls"]/2000,
    "DAPS_historical_vs_PC_same_180_ids":True,
    "comparison_claim":"CLOSE_NOMINAL_WORK_NOT_EXACT_MATCHED_NFE"
  },
  observed_prune_events_sum=sum(z["native_prune_events"] for z in bags),
  observed_minimum_nfe_mean=mean(z["minimum_nfe"] for z in bags),
  exact_pc_nfe_available=False,
  missing_trace="unlogged prior_consistency model calls when pruning stages retry; no native forward-call counter",
  per_bag=bags,
  no_gpu=True,no_gt=True,no_native_rerun=True,
  matched_nfe_certified=False,
  CP3_RETURN_AUTHORIZED=False
 )
 write(out/"XP_A16_CP3_NFE_LEDGER.json",report)
 print("CP3_NFE_LEDGER_NOMINAL_ONLY",
  "PC",cfg["nominal_per_bag_particle_equivalent_calls"],
  "DAPS",2000,
  "prune_events",report["observed_prune_events_sum"],flush=True)
 print("CP3_NFE_NOT_EXACT__DO_NOT_ASSERT_MATCHED_NFE",flush=True)
 print("CP3_NFE_LEDGER_DIR",out,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--expected-head",required=True)
 parser.add_argument("--out",required=True)
 parser.add_argument("--preflight-only",action="store_true")
 try:main(parser.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A16_CP3_NFE_LEDGER",e,flush=True)
  sys.exit(8)
