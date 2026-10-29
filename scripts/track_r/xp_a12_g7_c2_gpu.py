#!/usr/bin/env python3
"""A12 G7 / XP-C2 native denoiser equivariance: one image, 36 comparisons.

Diagnostic GT EVALUATION ONLY; no new inverse-solver candidate, no
bundled FFHQ data. Native DDPM VPPrecond sigma(t) maps DDPM 0..999 to
continuous EDM sigma, and model.tweedie() is used without changing
weights or forcing non-native denoiser-time schedules.
Observations are deterministic, rotation pair shares one noise draw.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g367_readiness_cpu import sha
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
    NATIVE,PY_DAPS,MODEL_SHA,IMAGENET_MODEL_SHA,verify_clean_sparse)
from scripts.track_r.xp_t_daps4k_worker_gpu import model_config,load_model_native
from scripts.track_r.r0_2_g0_cpu import gt_source
from prdiffusion.track_r.physics import apply_orbit
from PIL import Image
import numpy as np

DATASETS=("ffhq","imagenet")
def check(a):
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_g7_prod_")
      and run.is_dir() and not run.is_symlink(),"NEW_STAGE_ROOT")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(
      ["git","-C",str(wt),*args],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"WORKTREE")
 gate(Path(sys.executable).resolve()==PY_DAPS.resolve()
      and os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
      and a.gpu in range(4)
      and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
      and os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
           str(run/"pycache")+"/"),"SOURCE_GPU_ENV")
 gate(verify_clean_sparse(NATIVE),"NATIVE_DAPS_SOURCE")
 freeze=Path(a.freeze)
 gate(freeze.parent==BASE and freeze.name.startswith("xp_c_c2_")
      and freeze.is_dir(),"C2_FROZEN_GT_SOURCE")
 rec=json.loads((freeze/"XP_C_C2_PREP.json").read_text())
 c=freeze/"XP_C_GT_IMAGES.json"
 c2=freeze/"XP_C2_EQUIVARIANCE_TASKS.json"
 gate(rec["status"]=="XP_C_C2_PREP_PASS"
      and sha(c)==rec["C_manifest_sha256"]
      and sha(c2)==rec["C2_manifest_sha256"],"FROZEN_GT_MANIFEST")
 images=json.loads(c.read_text())
 tasks=json.loads(c2.read_text())
 gate(images["n_images"]==280 and
      tasks["n_tasks"]==10080 and
      tasks["denoiser_time_mapping"].startswith("REQUIRED_BEFORE_GPU"),
      "FROZEN_C2_TASK_SCOPE")
 gate(0<=a.index<280,"INDEX")
 image=images["images"][a.index]
 relevant=[r for r in tasks["tasks"] if
            r["dataset"]==image["dataset"] and
            r["basename"]==image["basename"]]
 gate(len(relevant)==36 and
      set(r["t_ddpm1000"] for r in relevant)==set(range(100,901,100)) and
      set(r["rotation"] for r in relevant)=={"100","111"} and
      set(r["noise_draw"] for r in relevant)=={0,1},"IMAGE_36_GRID")
 gate(sha(image["source_path"])==image["source_sha256"],"GT_ONLY_SOURCE_SHA")
 return run,image,relevant,rec
def load_gt(image):
 import torch
 if image["dataset"]=="ffhq":
  return gt_source(dict(gt_path=image["source_path"],
                        gt_format=image["gt_format"]))
 from torchvision.transforms import v2 as T
 with Image.open(image["source_path"]) as im:
  x=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True),
               T.Resize(256),T.CenterCrop(256)])(im.convert("RGB"))
 return x.unsqueeze(0)*2-1
def execute(a,run,image,tasks,rec):
 import torch
 from scripts.track_r.xp_t_daps4k_worker_gpu import (
       validate_weight_trace)
 torch.cuda.set_device(0)
 device=torch.device("cuda:0")
 source_cwd=Path.cwd()
 sys.path.insert(0,str(NATIVE))
 os.chdir(str(NATIVE))
 try:
  dataset=image["dataset"]
  cfg=(dict(checkpoint=str(ROOT/"models/ffhq_10m.pt"),
      num_channels=128,num_res_blocks=1,attention_resolutions=16)
     if dataset=="ffhq" else dict(
      checkpoint=str(ROOT/"models/imagenet256.pt"),
      num_channels=256,num_res_blocks=2,
      attention_resolutions="32,16,8"))
  expected=MODEL_SHA if dataset=="ffhq" else IMAGENET_MODEL_SHA
  gate(sha(cfg["checkpoint"])==expected,"MODEL_SHA")
  model,load_s,trace=load_model_native(model_config(cfg))
  validate_weight_trace(trace)
  model.eval()
  x=load_gt(image).to(device)
  gate(tuple(x.shape)==(1,3,256,256) and
       bool(torch.isfinite(x).all()),"GT_SHAPE")
  records=[];t0=time.monotonic()
  # Native VPPrecond sigma is exactly (exp(.5*beta_d*t^2+
  # beta_min*t)-1)^.5, where continuous t=index/(M-1).
  # Its sigma_inv maps the model to the pretrained DDPM timestep.
  native=model.model
  gate(native.M==1000 and float(native.beta_d)==19.9 and
       float(native.beta_min)==0.1,"DDPM_1000_NATIVE_MAPPING")
  cache={}
  with torch.no_grad():
   for z in tasks:
    index=int(z["t_ddpm1000"])
    j=int(z["noise_draw"])
    rot=z["rotation"]
    key=(index,j)
    if key not in cache:
     seed=int(z["noise_seed"])
     gen=torch.Generator(device="cpu").manual_seed(seed)
     eps=torch.randn(x.shape,generator=gen).to(device)
     sigma=float(native.sigma(torch.tensor(index/999.,device=device)))
     xt=x+sigma*eps
     base=model.tweedie(xt,sigma)
     denom=float((base-x).square().mean())
     cache[key]=(xt,base,denom,sigma)
    xt,base,denom,sigma=cache[key]
    rotated=apply_orbit(xt,rot)
    transformed=model.tweedie(rotated,sigma)
    reference=apply_orbit(base,rot)
    delta=float((transformed-reference).square().mean())
    records.append(dict(dataset=dataset,basename=image["basename"],
      t_ddpm1000=index,native_sigma=sigma,noise_draw=j,
      rotation=rot,delta=delta,denominator=denom,
      delta_tilde=delta/max(denom,1e-16),
      noise_seed=int(z["noise_seed"])))
  torch.cuda.synchronize()
  elapsed=time.monotonic()-t0
  gate(len(records)==36 and
       0<elapsed<7200 and
       all(np.isfinite(r["delta_tilde"]) for r in records),
       "G7_EQUIVARIANCE_OUTPUT")
  folder=run/"results"
  gate(folder.is_dir(),"OUTPUT")
  path=folder/f"{a.index:03d}_equivariance.json"
  with path.open("x") as f:
   json.dump(dict(status="XP_A12_G7_C2_IMAGE_EQUIVARIANCE_PASS",
     dataset=dataset,basename=image["basename"],
     n=36,records=records,
     model_sha256=expected,strict_checkpoint_load=True,
     native_VP_sigma_mapping=True,denoiser_load_seconds=load_s,
     native_eval_seconds=elapsed,
     source_C_manifest_sha256=rec["C_manifest_sha256"],
     source_C2_manifest_sha256=rec["C2_manifest_sha256"],
     diagnostic_GT_only=True,no_inverse_solver_run=True),
     f,sort_keys=True,indent=2)
   f.write("\n")
  print("XP_A12_G7_C2_IMAGE_PASS",a.index,dataset,
        "n=36","seconds",round(elapsed,2),flush=True)
 finally:os.chdir(source_cwd)
def main():
 p=argparse.ArgumentParser()
 for name in ("run","freeze","expected-head"):
  p.add_argument("--"+name,required=True)
 p.add_argument("--index",type=int,required=True)
 p.add_argument("--gpu",type=int,required=True)
 p.add_argument("--preflight-only",action="store_true")
 a=p.parse_args()
 run,image,tasks,rec=check(a)
 if a.preflight_only:
  print("A12_G7_NATIVE_CPU_PREFLIGHT_PASS",a.index,flush=True)
  return 0
 return execute(a,run,image,tasks,rec)
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G7_C2_NATIVE",e,flush=True)
  sys.exit(8)
