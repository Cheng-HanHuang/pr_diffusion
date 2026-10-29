#!/usr/bin/env python3
"""A12 G3 pinned PC-DAPS GPU one-bag native sampling, GT-free.

Source voilalab/PC-DAPS 59b2f046. Uses original PCDAPS.get_start()
and sampler.sample() with NO ground truth. Four returned particles
are kept separately. The upstream method's reporting quantity is
mean-of-particle metrics, NOT an additional generated fifth estimate:
we preserve the four native outputs and a clearly labeled auxiliary
pixel mean, never claim the latter is a native reconstruction.
Native model loads STRICTLY, following XP-T DAPS weight-trace gate.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from PIL import Image
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g3_prepare_cpu import PC,PC_HEAD,MODEL_SHA,sha,seed
from scripts.track_r.xp_t_daps4k_worker_gpu import (
    load_model_native,model_config)
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
    verify_clean_sparse as verify_native_daps)
def check(a):
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_g3_prod_")
      and run.is_dir() and not run.is_symlink(),"RUN_ROOT")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"WORKTREE")
 pcgit=lambda *q:subprocess.check_output(["git","-C",str(PC),*q],
               text=True,timeout=30).strip()
 gate(PC.is_dir() and pcgit("rev-parse","HEAD").startswith(PC_HEAD)
      and not pcgit("status","--porcelain","--untracked-files=no"),
      "PINNED_PC_NATIVE_SOURCE")
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
      and a.gpu in range(4)
      and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
      and os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
          str(run/"pycache")+"/"),"CUDA_AND_CACHE")
 prep=Path(a.prep)
 gate(prep.parent==BASE and prep.name.startswith("xp_a12_g3_prepare_")
      and prep.is_dir(),"INPUT_PREP")
 rec=json.loads((prep/"XP_A12_G3_PREP_RECEIPT.json").read_text())
 manifest=prep/"XP_A12_G3_180_BAGS.json"
 gate(rec["status"]=="XP_A12_G3_CPU_PREP_PASS" and
      rec["head"]==a.roots_source_head and
      sha(manifest)==rec["manifest_sha256"],"SOURCE_RECEIPT")
 mf=json.loads(manifest.read_text())
 gate(mf["status"]=="XP_A12_G3_180_NATIVE_BAGS_FROZEN"
      and mf["head"]==a.roots_source_head
      and mf["source_head"]==pcgit("rev-parse","HEAD")
      and mf["n_bags"]==180 and
      mf["bag_size"]==4 and
      mf["gt_paths_supplied"] is False
      and 0<=a.index<180,"NATIVE_BAG_MANIFEST")
 for p,expected in mf["native_source_files"].items():
  gate(sha(PC/p)==expected,"NATIVE_SOURCE_DRIFT_"+p)
 row=mf["rows"][a.index]
 gate(row["seed"]==seed(row["cohort"]+"_"+row["image_id"])
      and sha(row["measurement_path"])==row["measurement_sha256"],
      "BAG_Y_OR_SEED")
 gate(sha(ROOT/"models/ffhq_10m.pt")==MODEL_SHA,"MODEL_SHA")
 return run,mf,row,rec["manifest_sha256"]
def run_one(a,root,mf,row,mhash):
 import numpy as np
 import torch
 import yaml
 from scripts.b22.b22_smoke_common import (
    load_measurement,validate_reconstruction,save_model_range_png)
 torch.cuda.set_device(0)
 gate(torch.cuda.is_available() and torch.cuda.device_count()==1,"ONE_CUDA_DEVICE")
 device=torch.device("cuda:0")
 source_cwd=Path.cwd()
 sys.path.insert(0,str(PC))
 os.chdir(str(PC))
 try:
  np.random.seed(row["seed"]);torch.manual_seed(row["seed"])
  torch.cuda.manual_seed_all(row["seed"])
  torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.benchmark=False
  from sampler import get_sampler
  from forward_operator import get_operator
  sampler_cfg=yaml.safe_load((PC/"configs/sampler/edm_pcdaps.yaml").read_text())
  task_cfg=yaml.safe_load((PC/"configs/task/phase_retrieval.yaml").read_text())["pixel"]
  gate(sampler_cfg.pop("name")=="pcdaps"
       and sampler_cfg["annealing_scheduler_config"]["num_steps"]==200
       and sampler_cfg["diffusion_scheduler_config"]["num_steps"]==5
       and sampler_cfg["pcdaps_config"]["bag_size"]==4
       and task_cfg["operator"]==dict(name="phase_retrieval",
                                     oversample=2.0,sigma=0.05),
       "PC_PUBLISHED_PR_CONFIG")
  modelcfg=dict(checkpoint=str(ROOT/"models/ffhq_10m.pt"),
       num_channels=128,num_res_blocks=1,attention_resolutions=16)
  model,load_s,weight_trace=load_model_native(model_config(modelcfg))
  gate(weight_trace["unet_calls"]==1 and weight_trace["strict"] is True
       and weight_trace["missing_keys"]==[]
       and weight_trace["unexpected_keys"]==[],"STRICT_CHECKPOINT")
  model.eval()
  sampler=get_sampler(name="pcdaps",**sampler_cfg,
       mcmc_sampler_config=copy.deepcopy(task_cfg["mcmc_sampler_config"]))
  gate(sampler.bag_size==4 and sampler.prune is True
       and sampler.skip_final_langevin is True,
       "PC_NATIVE_BAG_AND_PRUNING")
  operator=get_operator(**copy.deepcopy(task_cfg["operator"]))
  y=load_measurement(row["measurement_path"],device=device)
  gate(tuple(y.shape)==(1,3,384,384)
       and bool(torch.isfinite(y).all()),"SIGNED_Y")
  xstart=sampler.get_start(1,model)
  gate(tuple(xstart.shape)==(1,4,3,256,256),"PC_PRIOR_SHAPE")
  torch.cuda.synchronize()
  t=time.monotonic()
  # No torch.no_grad(): PC-DAPS Langevin needs measurement gradients.
  result=sampler.sample(model,xstart,operator,y,
           evaluator=None,record=False,verbose=False)
  torch.cuda.synchronize()
  elapsed=time.monotonic()-t
  gate(tuple(result.shape)==(1,4,3,256,256)
       and bool(torch.isfinite(result).all())
       and 0<elapsed<7200,"PC_NATIVE_OUTPUT")
  # Only create output after all native computations succeed.
  dest=root/"roots"/("bag_"+hashlib.sha256(row["uid"].encode()).hexdigest()[:24])
  gate(not dest.exists(),"BAG_EXISTS")
  dest.mkdir(mode=0o700)
  bag=result.detach().cpu().float()
  torch.save({"particles":bag},dest/"particles.pt")
  files=[]
  for i in range(4):
   png=dest/f"particle_{i}.png"
   save_model_range_png(bag[:,i],png)
   files.append(dict(particle=i,path=str(png),sha256=sha(png)))
  # The native sampler's sole return is the four-particle bag.
  # The per-particle *metric mean* must be computed downstream under
  # the official PC-DAPS reporting protocol, not pixel averaged here.
  meta=dict(status="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS",
    uid=row["uid"],image_id=row["image_id"],cohort=row["cohort"],
    source_head=mf["source_head"],runner_head=a.expected_head,
    manifest_sha256=mhash,measurement_sha256=row["measurement_sha256"],
    checkpoint_sha256=MODEL_SHA,strict_checkpoint_trace=weight_trace,
    seed=row["seed"],method="PC-DAPS",bag_size=4,
    particles_pt_sha256=sha(dest/"particles.pt"),particle_pngs=files,
    native_sampling_wall_seconds=elapsed,
    native_model_loading_seconds=load_s,
    native_method_return="FOUR_PARTICLE_BAG",
    own_estimate_is_four_native_particles=True,
    reported_metric_rule="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN",
    native_prune_events=len(getattr(sampler,"prune_log",[])),
    no_ground_truth_access=True,no_GT_scoring=True)
  with (dest/"XP_A12_G3_BAG_COMPLETE.json").open("x") as f:
   json.dump(meta,f,sort_keys=True,indent=2);f.write("\n")
  print("XP_A12_G3_FOUR_PARTICLE_BAG_PASS",a.index,
        "wall_s",round(elapsed,2),flush=True)
 finally:
  os.chdir(source_cwd)
def main():
 parser=argparse.ArgumentParser()
 for x in ("run","prep","expected-head","roots-source-head"):
  parser.add_argument("--"+x,required=True)
 parser.add_argument("--index",required=True,type=int)
 parser.add_argument("--gpu",required=True,type=int)
 parser.add_argument("--preflight-only",action="store_true")
 args=parser.parse_args()
 root,mf,row,h=check(args)
 if args.preflight_only:
  print("XP_A12_G3_NATIVE_PREFLIGHT_PASS",args.index,flush=True)
  return 0
 return run_one(args,root,mf,row,h)
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G3_NATIVE",e,flush=True)
  sys.exit(8)
