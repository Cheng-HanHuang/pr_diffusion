#!/usr/bin/env python3
"""A13 G6 exact F8 native DAPS 400x5 clean-timing FFHQ worker.

Copies the existing DAPS 400x5 native Hydra command, GT-free signed-y
sitehook, strict UNet checkpoint trace and output verification. Does
NOT use the different DAPS-4K 400x10 method. Externally owned
clean-GPU-only controller accounts process occupied wall time.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from PIL import Image
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g45_prepare_cpu import sha
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
  ORIG_DAPS,DAPS_HEAD,PATCHED_SAMPLER_BLOB,MODEL_SHA,MODEL,git)
from scripts.track_r.xp_f8_legacy_replay_gpu import (
  DAPS_PY,TRACE_SOURCE,native_command,env,check_cfg)
def load(a):
 run=Path(a.run);prep=Path(a.prep)
 wt=Path(__file__).resolve().parents[2]
 gitwt=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                   text=True).strip()
 gate(run.parent==BASE and run.name.startswith("xp_a12_g6_prod_")
      and run.is_dir() and not run.is_symlink(),"RUN")
 gate(prep.parent==BASE and prep.name.startswith("xp_a12_g6_prepare_")
      and prep.is_dir(),"PREP")
 gate(gitwt("rev-parse","HEAD")==a.expected_head
      and not gitwt("status","--porcelain"),"WORKTREE")
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
      and a.gpu in range(4) and
      os.environ.get("PYTHONDONTWRITEBYTECODE")=="1","GPU_ENV")
 gate(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD
      and git(ORIG_DAPS,"hash-object","sampler.py")==PATCHED_SAMPLER_BLOB
      and sha(MODEL)==MODEL_SHA,"DAPS400X5_NATIVE_PROVENANCE")
 rec=json.loads((prep/"XP_A13_G6_PREP_RECEIPT.json").read_text())
 file=prep/"XP_A13_G6_DAPS400X5_TASKS.json"
 gate(rec["status"]=="XP_A13_G6_CLEAN_TIMING_TWO_FAMILIES_PREP_PASS"
      and rec["head"]==a.roots_source_head
      and rec["native_DAPS_config"]=="400x5_NOT_400x10"
      and rec["daps_manifest_sha256"]==sha(file),"INPUT_SHA")
 manifest=json.loads(file.read_text())
 gate(manifest["status"]=="XP_A13_G6_DAPS400X5_FFHQ_TASKS_FROZEN"
      and manifest["native_head"]==DAPS_HEAD
      and manifest["patched_sampler_blob"]==PATCHED_SAMPLER_BLOB
      and manifest["checkpoint_sha256"]==MODEL_SHA
      and manifest["n_roots"]==9
      and 0<=a.index<9,"MANIFEST")
 row=manifest["rows"][a.index]
 gate(sha(row["measurement_path"])==row["measurement_sha256"],
      "MEASUREMENT_SHA")
 return run,rec,row
def execute(a,run,rec,row):
 from PIL import Image
 import torch
 gate(torch.cuda.is_available() and torch.cuda.device_count()==1,
      "ONE_CUDA_GPU")
 dest=run/"gpu"/("root_"+hashlib.sha256(
      row["uid"].encode()).hexdigest()[:24])
 gate(not dest.exists(),"NEW_ROOT_OUTPUT")
 cmd=native_command(row,dest)
 cfg=check_cfg(cmd,row)
 gate(cfg["sampler"]["annealing_scheduler_config"]["num_steps"]==400
      and cfg["sampler"]["diffusion_scheduler_config"]["num_steps"]==5
      and cfg["pixel"]["mcmc_sampler_config"]["num_steps"]==100,
      "NATIVE_F8_DAPS400X5_RESOLVED_CONFIG")
 dest.mkdir(mode=0o700)
 synthetic=dest/"synthetic_input";synthetic.mkdir()
 Image.new("RGB",(256,256),(0,0,0)).save(synthetic/"00000.png")
 hook=dest/"sitehook";hook.mkdir()
 (hook/"sitecustomize.py").write_text(TRACE_SOURCE)
 trace=dest/"model_load_trace.json"
 log=dest/"native_stdout.log"
 t0=time.monotonic()
 with log.open("x") as f:
  p=subprocess.run(cmd,cwd=str(ORIG_DAPS),
      env=env(row,a.gpu,trace),
      stdout=f,stderr=subprocess.STDOUT)
 elapsed=time.monotonic()-t0
 gate(p.returncode==0 and trace.is_file(),
      "NATIVE_DAPS_COMMAND_OR_TRACE")
 model=json.loads(trace.read_text())
 gate(model["class_name"]=="UNetModel"
      and model["strict"] is True
      and model["missing_keys"]==[]
      and model["unexpected_keys"]==[]
      and model["tensor_count"]>=100,"STRICT_CHECKPOINT")
 image=dest/"legacy_daps400x5/samples/00000_run0000.png"
 gate(image.is_file() and image.stat().st_size>1000,
      "NATIVE_DAPS_400X5_SAMPLE")
 with Image.open(image) as img:
  gate(img.size==(256,256) and img.mode in ("RGB","RGBA"),
       "IMAGE256_RGB")
  img.verify()
 gate("Randomly initialize" not in log.read_text(errors="replace"),
      "CHECKPOINT_NOT_RANDOM")
 result=dict(status="XP_A13_G6_DAPS400X5_NATIVE_ROOT_PASS",
   head=a.expected_head,uid=row["uid"],image_id=row["image_id"],
   rep=row["rep"],seed=row["native_seed"],
   measurement_sha256=row["measurement_sha256"],
   source_head=DAPS_HEAD,patched_sampler_blob=PATCHED_SAMPLER_BLOB,
   model_sha256=MODEL_SHA,
   manifest_sha256=rec["daps_manifest_sha256"],
   resolved_config=cfg,checkpoint_trace=model,
   checkpoint_trace_sha256=sha(trace),
   output_png=str(image),output_sha256=sha(image),
   native_log_sha256=sha(log),
   native_sampling_plus_startup_wall_s=elapsed,
   GT_not_accessed=True,no_old_candidate_access=True,
   method="DAPS400x5_native_F8",timing_device="cuda:0")
 with (dest/"XP_A13_G6_DAPS400X5_ROOT.json").open("x") as f:
  json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
 print("XP_A13_G6_DAPS400X5_ROOT_PASS",a.index,
       "wall_s",round(elapsed,3),flush=True)
def main():
 p=argparse.ArgumentParser()
 for x in ("run","prep","expected-head","roots-source-head"):
  p.add_argument("--"+x,required=True)
 p.add_argument("--index",type=int,required=True)
 p.add_argument("--gpu",type=int,required=True)
 p.add_argument("--preflight-only",action="store_true")
 a=p.parse_args()
 run,rec,row=load(a)
 if a.preflight_only:
  print("A13_G6_DAPS400X5_SOURCE_PREFLIGHT_PASS",a.index,flush=True)
  return
 execute(a,run,rec,row)
if __name__=="__main__":
 try:main()
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_G6_DAPS400X5",e,flush=True)
  sys.exit(8)
