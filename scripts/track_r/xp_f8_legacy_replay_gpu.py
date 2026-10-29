#!/usr/bin/env python3
"""A10 legacy DAPS 400x5 one-root replay worker; ORIGINAL PATCHED native DAPS.

All GPU-side inputs are GT-free and historical-candidate-free. Uses native
Hydra CLI and DAPS original ImageDataset over a synthetic placeholder,
with the fixed historical measurement. The placeholder never supplies GT.
Preflight-only mode resolves and verifies Hydra config with GPUs hidden.
Subsequent offline gate determines bitwise or 0.1 dB / label agreement.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.track_r.xp_f8_legacy_preflight_cpu import (
    ROOT,BASE,ORIG_DAPS,DAPS_HEAD,MODEL,MODEL_SHA,B16,A10_DEADLINE,
    sha,read_json,git,fail)
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
    PATCHED_SAMPLER_BLOB)

DAPS_PY=ROOT/"conda-envs/daps/bin/python"
FROZEN_MANIFEST_SHA="f077dd896759d2e3eeaa879394680a0fcef5b816559ee84db46a06a199af2be5"
PRECHECK_SHA="c19c632a831c47926301814f2abfa05675c4f46cf104449e34ba98bc78287237"
PREP=BASE/"xp_f8_legacy_preflight_20261029T052005Z"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
TRACE_SOURCE=r'''
import os,json
try:
 import torch
 original=torch.nn.Module.load_state_dict
 def traced_load(self,weights,*args,**kwargs):
  result=original(self,weights,*args,**kwargs)
  if self.__class__.__name__=="UNetModel":
   p=os.environ.get("F8_STRICT_MODEL_TRACE")
   if p:
    payload=dict(class_name="UNetModel",tensor_count=len(weights),
                 strict=kwargs.get("strict",args[0] if args else True),
                 missing_keys=list(result.missing_keys),
                 unexpected_keys=list(result.unexpected_keys),
                 original_checkpoint_load=True)
    with open(p,"x") as f:json.dump(payload,f,sort_keys=True)
  return result
 torch.nn.Module.load_state_dict=traced_load
except Exception as e:
 raise RuntimeError("STOP_F8_STRICT_MODEL_TRACE_INIT "+str(e))
'''
def safe_output(path,root):
    p=Path(path)
    fail(not p.exists() and not p.is_symlink() and
         p.parent==root and
         p.name.startswith("replay_"),"OUTPUT_ALREADY_PRESENT")
    return p
def load_inputs(a):
    fail(git(Path(__file__).resolve().parents[2],"rev-parse","HEAD")==a.expected_head
         and not git(Path(__file__).resolve().parents[2],"status","--porcelain"),
         "WORKTREE_DRIFT")
    fail(sha(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")==FROZEN_MANIFEST_SHA
         and sha(PREP/"XP_F8_PRECHECK_RECEIPT.json")==PRECHECK_SHA,
         "PREFLIGHT_SHA")
    m=read_json(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")
    r=read_json(PREP/"XP_F8_PRECHECK_RECEIPT.json")
    fail(r["status"]=="XP_F8_REPLAY_INPUT_LINEAGE_PREFLIGHT_PASS"
         and r["native_adapter_certified"] is False
         and r["any_gpu_job_launched"] is False
         and m["n_replay"]==8 and len(m["rows"])==8,
         "PREFLIGHT_SCHEMA")
    fail(0<=a.index<8,"ROW_INDEX")
    row=m["rows"][a.index]
    fail(set(row)=={"uid","source","family","image_id","rep",
                    "native_seed","seed_rule","measurement_path",
                    "measurement_sha256"},"ROW_SCHEMA")
    fail(sha(row["measurement_path"])==row["measurement_sha256"],
         "MEASUREMENT_SHA")
    fail(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD
         and git(ORIG_DAPS,"hash-object","sampler.py")==
             PATCHED_SAMPLER_BLOB,
         "NATIVE_SOURCE_DRIFT")
    # Verify the original FFHQ model is either the same canonical file,
    # or overridden by the verified, absolute canonical checkpoint.
    # The overridden path is checked again in the resolved Hydra config.
    import yaml
    model_yaml=yaml.safe_load(
         (ORIG_DAPS/"configs/model/ffhq256ddpm.yaml").read_text())
    fail(model_yaml["model_config"]["num_channels"]==128
         and model_yaml["model_config"]["num_res_blocks"]==1
         and model_yaml["model_config"]["attention_resolutions"]==16,
         "HIST_MODEL_ARCH")
    proof=m["model_source"]
    for relative,digest in proof["audited_native_source_sha256"].items():
        fail(sha(ORIG_DAPS/relative)==digest,
             "NATIVE_FILE_DRIFT_"+relative)
    fail(sha(MODEL)==MODEL_SHA,"MODEL_SHA")
    fail(sha(ORIG_DAPS/"configs/sampler/edm_daps.yaml")==
         proof["original_sampler_yaml_sha256"],"YAML_SOURCE_SHA")
    fail(sha(ORIG_DAPS/"configs/task/phase_retrieval.yaml")==
         proof["original_task_yaml_sha256"],"TASK_SOURCE_SHA")
    fail(Path(DAPS_PY).is_file(),"DAPS_ENV")
    status=read_json(B16/"B16_PRODUCTION_STATE.json")
    fail(status["status"]=="A10_B16_GENERATION_COMPLETE_WAIT_F8"
         and status["active"]==[],"B16_NOT_FINAL")
    fail(datetime.datetime.now(ZoneInfo("America/Detroit"))<A10_DEADLINE,
         "A10_DEADLINE")
    # Disallow any out-of-budget replay admission; supervisor also audits
    # aggregate in-flight GPU time, including every child process.
    fail((status["gate_spent_gpu_seconds"]+
          status["spent_gpu_seconds"])/3600<70,"A10_BUDGET")
    return row
def native_command(row,dest):
    seed=int(row["native_seed"])
    return [
        str(DAPS_PY),str(ORIG_DAPS/"posterior_sample.py"),
        # Use DAPS original ImageDataset. Its Empty dataset passes an
        # OmegaConf ListConfig to torch.zeros and fails at runtime. An
        # RGB placeholder is created under this root's private output
        # directory; the frozen y overrides synthetic measurement before
        # sampling. The placeholder is NOT a historical or GT image.
        "+data={name:image,root:"+str(dest/"synthetic_input")+
            ",resolution:256,start_id:0,end_id:1,device:cuda}",
        "+measurement_path="+str(row["measurement_path"]),
        "+model=ffhq256ddpm","+sampler=edm_daps",
        "+task=phase_retrieval","task_group=pixel",
        "batch_size=1","gpu=0","seed="+str(seed),
        "num_runs=1","name=legacy_daps400x5",
        "save_dir="+str(dest),"wandb=false","eval_fid=false",
        "hydra.run.dir="+str(dest/"hydra"),
        "model.model_config.model_path="+str(MODEL),
        "sampler.annealing_scheduler_config.num_steps=400",
        "sampler.diffusion_scheduler_config.num_steps=5",
        "save_samples=true","save_traj=false",
        "save_traj_raw_data=false","save_traj_video=false",
    ]
def env(row,gpu=None,trace=None):
    e=os.environ.copy()
    for k in ("B21_CONT_ENABLE","B21_CONT_STATE_PATH","B21_CONT_NOISE_SEED",
              "B21_SAVE_STATE_STEPS"):
        e.pop(k,None)
    seed=str(row["native_seed"])
    e.update(B21_START_NOISE_SEED=seed,B21_SOURCE_SEED=seed,
       B21_MEASUREMENT_PATH=str(row["measurement_path"]),
       B20_LF_ENABLE="0",B20_LF_ALPHA="0.0",
       PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
       OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
       OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",
       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if gpu is None:e["CUDA_VISIBLE_DEVICES"]=""
    else:e["CUDA_VISIBLE_DEVICES"]=str(gpu)
    if trace is not None:
        e["F8_STRICT_MODEL_TRACE"]=str(trace)
        e["PYTHONPATH"]=str(trace.parent/"sitehook") + os.pathsep+e.get("PYTHONPATH","")
    return e
def check_cfg(cmd,row):
    import yaml
    command=cmd+["--cfg","job","--resolve"]
    # Importing the historical DAPS module graph (torch, diffusion,
    # imageio, Hydra) can be slow under CPU co-tenancy. Config checks
    # themselves are not performance claims. Preserve the full resolved
    # YAML gate; raise only after a bounded 300-second startup window.
    try:
        p=subprocess.run(command,cwd=str(ORIG_DAPS),env=env(row),
            capture_output=True,text=True,timeout=300)
    except subprocess.TimeoutExpired as e:
        fail(False,"HYDRA_CONFIG_CPU_STARTUP_TIMEOUT_300S "+
             "row_seed="+str(row["native_seed"]))
    fail(p.returncode==0,"HYDRA_CONFIG_RESOLVE "+p.stderr[-600:])
    cfg=yaml.safe_load(p.stdout)
    fail(isinstance(cfg,dict),"HYDRA_CONFIG_NOT_MAPPING")
    save_dir=next(x.split("=",1)[1] for x in cmd
                  if x.startswith("save_dir="))
    fail(cfg["data"]["name"]=="image"
         and cfg["data"]["root"]==str(Path(save_dir)/"synthetic_input")
         and int(cfg["data"]["resolution"])==256
         and int(cfg["data"]["start_id"])==0
         and int(cfg["data"]["end_id"])==1,
         "SYNTHETIC_IMAGE_DATA_NOT_USED")
    fail(cfg["measurement_path"]==row["measurement_path"],
         "FIXED_MEASUREMENT_NOT_USED")
    fail(cfg["model"]["model_config"]["model_path"]==str(MODEL)
         and int(cfg["sampler"]["annealing_scheduler_config"]["num_steps"])==400
         and int(cfg["sampler"]["diffusion_scheduler_config"]["num_steps"])==5
         and cfg["task_group"]=="pixel"
         and int(cfg["task"]["pixel"]["mcmc_sampler_config"]["num_steps"])==100,
         "LEGACY_RESOLVED_SETTINGS")
    fail(int(cfg["seed"])==int(row["native_seed"]) and
         cfg["task"]["pixel"]["operator"]["name"]=="phase_retrieval",
         "SEED_OR_TASK_MISMATCH")
    return dict(data=cfg["data"],task_group=cfg["task_group"],
        model_path=cfg["model"]["model_config"]["model_path"],
        pixel=cfg["task"]["pixel"],
        sampler=cfg["sampler"],seed=cfg["seed"])
def do_gpu(a,row):
    fail(a.gpu in range(4) and
         os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu),
         "GPU_BINDING")
    root=Path(a.outroot)
    fail(root.is_dir() and root.parent==BASE and
         root.name.startswith("xp_f8_legacy_replay_"),
         "OUTPUT_ROOT")
    dest=safe_output(root/f"replay_{a.index:02d}",root)
    cfg=check_cfg(native_command(row,dest),row)
    dest.mkdir(parents=False,exist_ok=False)
    # Create exactly one deterministic RGB placeholder. The native DAPS
    # ImageDataset loads this as a shape carrier; it is not GT and is
    # never used to form the fixed phase-retrieval measurement.
    from PIL import Image
    synthetic_dir=dest/"synthetic_input"
    synthetic_dir.mkdir(mode=0o700)
    placeholder=synthetic_dir/"00000.png"
    Image.new("RGB",(256,256),color=(0,0,0)).save(placeholder)
    fail(sha(placeholder)==sha(synthetic_dir/"00000.png"),
         "SYNTHETIC_PLACEHOLDER_HASH")
    hook=dest/"sitehook"
    hook.mkdir()
    (hook/"sitecustomize.py").write_text(TRACE_SOURCE)
    trace=dest/"model_load_trace.json"
    log=dest/"native_stdout.log"
    command=native_command(row,dest)
    # Do not modify original native checkout, its configs or datasets.
    # Do not feed GT/historical candidate paths to the subprocess.
    start=time.monotonic()
    with log.open("x") as f:
        completed=subprocess.run(command,cwd=str(ORIG_DAPS),
            env=env(row,a.gpu,trace),stdout=f,
            stderr=subprocess.STDOUT,check=False)
    elapsed=time.monotonic()-start
    fail(completed.returncode==0,
         f"NATIVE_EXIT_{completed.returncode} log={log}")
    fail(trace.is_file(),"MODEL_CHECKPOINT_UNVERIFIED")
    m=json.loads(trace.read_text())
    fail(m["class_name"]=="UNetModel" and m["strict"] is True
         and m["missing_keys"]==[] and m["unexpected_keys"]==[]
         and m["tensor_count"]>=100,"CHECKPOINT_NOT_STRICT")
    sample=dest/"legacy_daps400x5/samples/00000_run0000.png"
    fail(sample.is_file() and not sample.is_symlink() and
         sample.stat().st_size>1000,"NATIVE_SAMPLE_MISSING")
    from PIL import Image
    with Image.open(sample) as img:
        fail(img.size==(256,256) and img.mode in ("RGB","RGBA"),
             "BAD_OUTPUT_PNG")
        img.verify()
    msg=log.read_text(errors="replace")
    fail("Randomly initialize" not in msg,"MODEL_RANDOM_INITIALIZATION")
    receipt=dict(status="XP_F8_LEGACY_REPLAY_NATIVE_ROOT_PASS",
         index=a.index,source=row["source"],image_id=row["image_id"],
         rep=row["rep"],uid=row["uid"],seed=row["native_seed"],
         physical_gpu=a.gpu,runner_head=a.expected_head,
         source_head=DAPS_HEAD,model_sha256=MODEL_SHA,
         measurement_sha256=row["measurement_sha256"],
         replay_input_manifest_sha256=FROZEN_MANIFEST_SHA,
         resolved_native_config=cfg,
         model_trace=m,model_trace_sha256=sha(trace),
         sample_sha256=sha(sample),sample_path=str(sample),
         native_log_sha256=sha(log),native_wall_seconds=elapsed,
         synthetic_placeholder_sha256=sha(placeholder),
         synthetic_placeholder_path=str(placeholder),
         placeholder_used_for_shape_only=True,
         gt_passed_to_native=False,historical_candidate_passed_to_native=False,
         timing_measured=False,co_tenancy_allowed=True,
         offline_replay_comparison_pending=True)
    with (dest/"XP_F8_LEGACY_REPLAY_NATIVE_ROOT.json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("XP_F8_LEGACY_NATIVE_ROOT_PASS",a.index,
          row["source"],row["rep"],"gpu",a.gpu,
          "wall",round(elapsed,2),"output",str(dest),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--gpu",type=int)
    p.add_argument("--outroot")
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    row=load_inputs(a)
    cfg=check_cfg(native_command(row,Path("/tmp/f8-daps-config-probe")),row)
    if a.preflight_only:
        print("XP_F8_LEGACY_NATIVE_HYDRA_PREFLIGHT_PASS",a.index,
              row["source"],row["image_id"],row["rep"],
              "seed",row["native_seed"],
              "steps",cfg["sampler"]["annealing_scheduler_config"]["num_steps"],
              "diffusion_steps",cfg["sampler"]["diffusion_scheduler_config"]["num_steps"],
              flush=True)
        return 0
    fail(a.gpu is not None and a.outroot is not None,"GPU_ARGS")
    do_gpu(a,row)
    return 0
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_LEGACY_NATIVE "+str(e),flush=True)
        sys.exit(8)
