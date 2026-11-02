#!/usr/bin/env python3
"""A19 XP-I2 ImageNet native PC-DAPS, one bag of four per D1 image.

Native PC-DAPS pinned 59b2f046, 200x5 with pruning, ImageNet DDPM.
GT-free: consumes frozen ORIGINAL-D1 signed y only, never GT or XP-N.
No new selector or methodological changes. Output 4 native particles.
Production launch requires prior 2-bag native repeatability replay,
manifest/code SHA freeze, A10.1 admission and shared budget ledger.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g367_readiness_cpu import PC,PC_HEAD
from scripts.track_r.xp_a24_xpi2_prepare_cpu import sha,seed,IMAGE_MODEL_SHA
from scripts.track_r.xp_a14_g3_mcmc_config import normalize_mcmc_config
from scripts.track_r.xp_t_daps4k_worker_gpu import load_model_native,model_config
from scripts.track_r.xp_t_daps4k_prepare_cpu import verify_clean_sparse

def check(a):
 wt=Path(__file__).resolve().parents[2]
 cmd=lambda root,*q:subprocess.check_output(["git","-C",str(root),*q],
                                text=True,timeout=45).strip()
 gate(cmd(wt,"rev-parse","HEAD")==a.expected_head
      and not cmd(wt,"status","--porcelain"),"A24_I2_PINNED_EXECUTOR")
 pc_head=cmd(PC,"rev-parse","HEAD")
 gate(pc_head.startswith(PC_HEAD)
      and not cmd(PC,"status","--porcelain","--untracked-files=no"),
      "A24_I2_PC_SOURCE")
 run=Path(a.run);prep=Path(a.prep)
 gate(run.parent==BASE and run.name.startswith("xp_a24_xpi2_prod_")
      and run.is_dir() and not run.is_symlink(),"A24_I2_RUN")
 gate(prep.parent==BASE and prep.name.startswith("xp_a24_xpi2_prepare_")
      and prep.is_dir() and not prep.is_symlink(),"A24_I2_PREP")
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
      and a.gpu in range(4)
      and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
      and os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
          str(run/"pycache")+"/"),"A24_I2_NATIVE_GPU_ENV")
 rec=json.loads((prep/"XP_A24_XPI2_PREP_RECEIPT.json").read_text())
 mfpath=prep/"XP_A24_XPI2_100_BAGS.json"
 gate(rec["status"]=="A19_XPI2_D1_100_BAGS_CPU_FROZEN_NOT_GENERATED"
      and rec["expected_head"]==a.roots_source_head
      and sha(mfpath)==rec["manifest_sha256"],
      "A24_I2_MANIFEST_RECEIPT")
 mf=json.loads(mfpath.read_text())
 gate(mf["status"]=="A19_XPI2_D1_100_NATIVE_FOUR_PARTICLE_BAGS_FROZEN"
      and mf["expected_head"]==a.roots_source_head
      and mf["source_head"]==pc_head
      and mf["dataset"]=="imagenet"
      and mf["n_bags"]==100 and mf["bag_size"]==4
      and mf["n_particles"]==400
      and mf["no_GT_paths"] is True
      and mf["generation_authorized"] is False
      and mf["two_bag_exact_replay_required"] is True
      and len(mf["rows"])==100 and 0<=a.index<100,
      "A24_I2_NATIVE_SCOPE")
 for name,digest in mf["native_source_files"].items():
  gate(sha(PC/name)==digest,"A24_I2_SOURCE_SHA_"+name)
 gate(sha(ROOT/"models/imagenet256.pt")==IMAGE_MODEL_SHA
      and mf["model_sha256"]==IMAGE_MODEL_SHA,"A24_I2_MODEL_SHA")
 row=mf["rows"][a.index]
 gate(row["dataset"]=="imagenet" and
      row["cohort"]=="XP_IMAGENET100"
      and row["seed"]==seed(row["image_id"])
      and row["bag_size"]==4
      and row["uid"]==f'XP|PCDAPS|imagenet|{row["image_id"]}|bag0'
      and sha(row["measurement_path"])==row["measurement_sha256"],
      "A24_I2_FROZEN_D1_Y_AND_SEED")
 return wt,run,mf,row,rec["manifest_sha256"]

def native(a,run,mf,row,digest):
 import numpy as np,torch,yaml
 from scripts.b22.b22_smoke_common import load_measurement,save_model_range_png
 torch.cuda.set_device(0)
 gate(torch.cuda.is_available() and torch.cuda.device_count()==1,
      "A24_I2_SINGLE_VISIBLE_DEVICE")
 device=torch.device("cuda:0")
 priorcwd=Path.cwd()
 sys.path.insert(0,str(PC))
 os.chdir(str(PC))
 try:
  np.random.seed(row["seed"])
  torch.manual_seed(row["seed"])
  torch.cuda.manual_seed_all(row["seed"])
  torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.benchmark=False
  from sampler import get_sampler
  from forward_operator import get_operator
  scfg=yaml.safe_load((PC/"configs/sampler/edm_pcdaps.yaml").read_text())
  task=yaml.safe_load((PC/"configs/task/phase_retrieval.yaml").read_text())["pixel"]
  gate(scfg.pop("name")=="pcdaps"
       and scfg["annealing_scheduler_config"]["num_steps"]==200
       and scfg["diffusion_scheduler_config"]["num_steps"]==5
       and scfg["pcdaps_config"]["bag_size"]==4
       and task["operator"]==dict(name="phase_retrieval",
                                  oversample=2.0,sigma=0.05),
       "A24_I2_FROZEN_NATIVE_CONFIG")
  modelcfg=dict(checkpoint=str(ROOT/"models/imagenet256.pt"),
     num_channels=256,num_res_blocks=2,attention_resolutions="32,16,8")
  model,load_s,trace=load_model_native(model_config(modelcfg))
  gate(trace["unet_calls"]==1 and trace["strict"] is True
       and trace["missing_keys"]==[] and trace["unexpected_keys"]==[],
       "A24_I2_STRICT_IMAGENET_WEIGHTS")
  model.eval()
  sampler=get_sampler(name="pcdaps",**scfg,
       mcmc_sampler_config=normalize_mcmc_config(
          task["mcmc_sampler_config"]))
  gate(sampler.bag_size==4 and sampler.prune is True
       and sampler.skip_final_langevin is True,
       "A24_I2_NATIVE_FOUR_PARTICLE_PRUNING")
  y=load_measurement(row["measurement_path"],device=device)
  gate(tuple(y.shape)==(1,3,384,384) and
       bool(torch.isfinite(y).all()),"A24_I2_SIGNED_Y")
  operator=get_operator(**copy.deepcopy(task["operator"]))
  x0=sampler.get_start(1,model)
  gate(tuple(x0.shape)==(1,4,3,256,256),"A24_I2_FOUR_STARTS")
  torch.cuda.synchronize()
  start=time.monotonic()
  # NO torch.no_grad(): PC-DAPS native measurement gradients needed.
  result=sampler.sample(model,x0,operator,y,
      evaluator=None,record=False,verbose=False)
  torch.cuda.synchronize()
  elapsed=time.monotonic()-start
  gate(tuple(result.shape)==(1,4,3,256,256)
       and bool(torch.isfinite(result).all())
       and 0<elapsed<7200,"A24_I2_NATIVE_OUTPUT")
  dest=run/"roots"/("bag_"+hashlib.sha256(
                  row["uid"].encode()).hexdigest()[:24])
  gate(not dest.exists(),"A24_I2_NEVER_OVERWRITE_BAG")
  dest.mkdir(mode=0o700)
  bag=result.detach().cpu().float()
  torch.save({"particles":bag},dest/"particles.pt")
  files=[]
  for j in range(4):
   png=dest/f"particle_{j}.png"
   save_model_range_png(bag[:,j],png)
   files.append(dict(particle=j,path=str(png),sha256=sha(png)))
  receipt=dict(status="A24_XPI2_NATIVE_FOUR_PARTICLE_BAG_PASS",
     uid=row["uid"],dataset="imagenet",image_id=row["image_id"],
     cohort=row["cohort"],seed=row["seed"],
     source_head=mf["source_head"],runner_head=a.expected_head,
     manifest_sha256=digest,measurement_sha256=row["measurement_sha256"],
     checkpoint_sha256=IMAGE_MODEL_SHA,strict_checkpoint_trace=trace,
     native_method="PC-DAPS",bag_size=4,
     native_return="FOUR_PARTICLE_BAG",
     reported_metric_rule="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN",
     particles_pt_sha256=sha(dest/"particles.pt"),
     particle_pngs=files,native_sampling_wall_seconds=elapsed,
     native_model_loading_seconds=load_s,
     native_prune_events=len(getattr(sampler,"prune_log",[])),
     GT_access=False,no_GT_scoring=True,
     no_data_dependent_model_or_seed_changes=True)
  with (dest/"XP_A24_I2_BAG_COMPLETE.json").open("x") as f:
   json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
  print("A24_XPI2_NATIVE_BAG_PASS",a.index,row["image_id"],
        "seconds",round(elapsed,4),flush=True)
 finally:
  os.chdir(priorcwd)

def main(a):
 wt,run,mf,row,digest=check(a)
 if a.preflight_only:
  print("A24_XPI2_NATIVE_PREFLIGHT_PASS",a.index,
        row["image_id"],flush=True)
  return
 native(a,run,mf,row,digest)
if __name__=="__main__":
 ap=argparse.ArgumentParser()
 for name in ("run","prep","expected-head","roots-source-head"):
  ap.add_argument("--"+name,required=True)
 ap.add_argument("--index",type=int,required=True)
 ap.add_argument("--gpu",type=int,required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A24_XPI2_NATIVE",str(e),flush=True)
  sys.exit(8)
