#!/usr/bin/env python3
"""XP-T SITCOM W=1 native GT-free worker; one frozen y, one isolated GPU.

Strictly instruments successful UNetModel.load_state_dict, because the pinned
SITCOM create_model catches checkpoint load errors and otherwise continues
with randomly initialized weights. No GT, historical candidates or test data.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

from scripts.track_r.xp_t_sitcom_prepare_cpu import (
    ROOT,WT,BASE,SITCOM,SITCOM_HEAD,IMAGENET_MODEL_SHA,MODEL_SHA,
    safe,sha,read,settings,check_native_checkout)
from scripts.track_r.xp_sitcom_replay_gpu import folder as replay_folder

RUN=BASE/"xp_t_sitcom_20261026T233451Z"
PREP_HEAD="87fe898e0c876a5a153f91d8acca9c40e00a8120"
MANIFEST_SHA="995915c3b7b7a00b1d4d678a4a013c8cc9c2445a513704a92e513d93d6d97131"
NATIVE_PY=ROOT/"conda-envs/sitcom_ode_bw/bin/python"
ALLOWED_DATASETS={"ffhq","imagenet"}
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def dump(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def record_folder(row):
    token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
    return RUN/"gpu"/("root_"+token)

def config(dataset,preset):
    return {
      "attention_resolutions":preset["attention_resolutions"],
      "channel_mult":"","class_cond":False,"dropout":0.0,
      "image_size":256,"learn_sigma":True,
      "model_path":preset["checkpoint_path"],
      "num_channels":preset["num_channels"],
      "num_head_channels":64,"num_heads":4,"num_heads_upsample":-1,
      "num_res_blocks":preset["num_res_blocks"],"resblock_updown":True,
      "use_checkpoint":False,"use_fp16":False,
      "use_new_attention_order":False,"use_scale_shift_norm":True,
    }

def verify_checkpoint_trace(trace):
    # torch.nn.Module.load_state_dict succeeds only after correct matching
    # checkpoint tensors are copied to the target UNet.
    if not (trace.get("unet_calls")==1 and
            trace.get("missing_keys")==[] and
            trace.get("unexpected_keys")==[] and
            isinstance(trace.get("tensor_count"),int) and
            trace["tensor_count"]>=100 and
            trace.get("strict") is True):
        raise RuntimeError("STOP_XPT_SITCOM_UNVERIFIED_MODEL_WEIGHTS")
    return trace

def immutable_manifest(expected_head):
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_XPT_SITCOM_BYTECODE_WRITING")
    prefix=os.environ.get("PYTHONPYCACHEPREFIX","")
    if not (prefix.startswith(str(RUN/"pycache")+"/") and
            Path(sys.executable).resolve()==NATIVE_PY.resolve()):
        raise RuntimeError("STOP_XPT_SITCOM_ENV")
    work_head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30).strip()
    if work_head!=expected_head or dirty:
        raise RuntimeError("STOP_XPT_SITCOM_WORKTREE")
    if sha(RUN/"XP_T_SITCOM_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_XPT_SITCOM_MANIFEST_SHA")
    prep=read(RUN/"XP_T_SITCOM_PREP.json")
    if (prep.get("status")!="XP_T_SITCOM_PREP_PASS" or
        prep.get("head")!=PREP_HEAD or
        prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("n_roots")!=18):
        raise RuntimeError("STOP_XPT_SITCOM_PREP")
    manifest=read(RUN/"XP_T_SITCOM_TASKS.json")
    if (manifest.get("status")!="XP_T_SITCOM_TASKS_FROZEN" or
        manifest.get("schema")!="prdiff.track_r.xp.sitcom.timing-batch1.v1" or
        manifest.get("head")!=PREP_HEAD or
        manifest.get("native_source_head")!=SITCOM_HEAD or
        manifest.get("n_roots")!=18 or
        manifest.get("n_images")!=6 or
        manifest.get("runs_per_image")!=3 or
        manifest.get("new_roots_only") is not True or
        manifest.get("gt_paths_supplied") is not False or
        manifest.get("historical_paths_supplied") is not False or
        manifest.get("batch_size")!=1):
        raise RuntimeError("STOP_XPT_SITCOM_MANIFEST_SCHEMA")
    rows=manifest.get("rows")
    required={"uid","family","dataset","image_id","measurement_source",
              "rep","seed","measurement_path","measurement_sha256"}
    if (not isinstance(rows,list) or len(rows)!=18 or
        len({x["uid"] for x in rows})!=18 or
        any(set(x)!=required for x in rows)):
        raise RuntimeError("STOP_XPT_SITCOM_ROWS_SCHEMA")
    if [x["dataset"] for x in rows]!=["ffhq"]*9+["imagenet"]*9:
        raise RuntimeError("STOP_XPT_SITCOM_DATASET_ORDER")
    expected_config=settings(read(WT/"configs/b22/b22_2_overnight.json"))
    if expected_config!=manifest.get("sampler_config"):
        raise RuntimeError("STOP_XPT_SITCOM_SAMPLER_CONFIG")
    presets=manifest.get("model_presets")
    for name,digest in (("ffhq",MODEL_SHA),("imagenet",IMAGENET_MODEL_SHA)):
        if presets[name]["checkpoint_sha256"]!=digest:
            raise RuntimeError("STOP_XPT_SITCOM_MODEL_SHA_REFERENCE")
        if sha(presets[name]["checkpoint_path"])!=digest:
            raise RuntimeError("STOP_XPT_SITCOM_CHECKPOINT_CONTENT")
    native_head=subprocess.check_output(
        ["git","-C",str(SITCOM),"rev-parse","HEAD"],text=True,timeout=30).strip()
    native=check_native_checkout(native_head,
       subprocess.check_output(["git","-C",str(SITCOM),"status","--porcelain=v1",
           "--untracked-files=no"],text=True,timeout=30),
       subprocess.check_output(["git","-C",str(SITCOM),"diff","--numstat","HEAD",
           "--","posterior_sample.py"],text=True,timeout=30))
    if (native["tracked_modified"]!=manifest["external_tracked_diff_allowlist"] or
        sha(SITCOM/"posterior_sample.py")!=manifest["external_unused_cli_sha256"]):
        raise RuntimeError("STOP_XPT_SITCOM_NATIVE_DIFF")
    return manifest,rows

def checked_row(manifest,rows,index):
    if index not in range(18):
        raise RuntimeError("STOP_XPT_SITCOM_INDEX")
    row=rows[index]
    dataset=row["dataset"]
    image=row["image_id"]
    from scripts.track_r.xp_t_sitcom_prepare_cpu import seed
    if (dataset not in ALLOWED_DATASETS or
        row["family"]!="SITCOM" or
        row["rep"]!=index%3 or
        row["seed"]!=seed(dataset,image,row["rep"]) or
        row["uid"]!=f"XP_T|SITCOM|{dataset}|{image}|rep{index%3}" or
        row["measurement_source"]!=
        ("S1_b22" if dataset=="ffhq" else "XP_IMAGENET100")):
        raise RuntimeError("STOP_XPT_SITCOM_FROZEN_IDENTITY")
    if sha(row["measurement_path"])!=row["measurement_sha256"]:
        raise RuntimeError("STOP_XPT_SITCOM_Y_SHA")
    return row

def build_native(method,model_cfg,seed,device):
    import numpy as np
    import torch
    # A native independent-seed process per root, with generator seeded
    # BEFORE model construction, exactly B24's independent-seed protocol.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    from forward_operator import get_operator
    from sampler import get_sampler
    from model import get_model
    op=get_operator(name="phase_retrieval",sigma=0.05,oversample=2.0)
    sampler=get_sampler(latent=False,
       annealing_scheduler_config={
         "num_steps":method["anneal_steps"],
         "schedule":method["anneal_schedule"],
         "sigma_max":method["anneal_sigma_max"],
         "sigma_min":method["anneal_sigma_min"],
         "sigma_final":0.0,"timestep":method["timestep"]},
       diffusion_scheduler_config={
         "num_steps":method["diff_steps"],
         "schedule":method["diff_schedule"],
         "sigma_min":method["diff_sigma_min"],
         "sigma_final":0.0,"timestep":method["timestep"]},
       lgvd_config={
         "lr":method["lgvd_lr"],"lr_min_ratio":method["lgvd_lr_min_ratio"],
         "num_steps":method["lgvd_steps"],"tau":method["lgvd_tau"]})
    trace={"unet_calls":0}
    original=torch.nn.Module.load_state_dict
    def checked(self,weights,*args,**kwargs):
        ret=original(self,weights,*args,**kwargs)
        if self.__class__.__name__=="UNetModel":
            trace.update(unet_calls=trace["unet_calls"]+1,
                         missing_keys=list(ret.missing_keys),
                         unexpected_keys=list(ret.unexpected_keys),
                         tensor_count=len(weights),
                         strict=kwargs.get("strict",args[0] if args else True))
        return ret
    begin=time.perf_counter()
    torch.nn.Module.load_state_dict=checked
    try:
        model=get_model(name="ddpm",model_config=model_cfg)
    finally:
        torch.nn.Module.load_state_dict=original
    verify_checkpoint_trace(trace)
    torch.cuda.synchronize(device)
    return op,sampler,model,time.perf_counter()-begin,trace

def generate(a,manifest,row):
    import numpy as np
    import torch
    from scripts.b22.b22_smoke_common import (
         load_measurement,validate_reconstruction,tensor_content_sha256,
         save_model_range_png)
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.gpu)
        or torch.cuda.device_count()!=1 or not torch.cuda.is_available()
        or not 0<=a.gpu<4):
        raise RuntimeError("STOP_XPT_SITCOM_GPU_ENV")
    dest=record_folder(row)
    if dest.parent!=RUN/"gpu" or dest.exists() or dest.is_symlink():
        raise RuntimeError("STOP_XPT_SITCOM_DEST")
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    preset=manifest["model_presets"][row["dataset"]]
    mf_cfg=config(row["dataset"],preset)
    # Verify native architecture against the pinned SITCOM YAML presets
    # in a separate CPU test; these settings exactly mirror that source.
    if (mf_cfg["num_channels"],mf_cfg["num_res_blocks"],mf_cfg["attention_resolutions"])!=(
        (128,1,16) if row["dataset"]=="ffhq" else (256,2,"32,16,8")):
        raise RuntimeError("STOP_XPT_SITCOM_ARCHITECTURE")
    import sys
    native_was=Path.cwd()
    sys.path.insert(0,str(SITCOM))
    os.chdir(str(SITCOM))
    try:
        op,sampler,model,model_s,trace=build_native(
             manifest["sampler_config"],mf_cfg,row["seed"],device)
        observation=load_measurement(row["measurement_path"],device=device)
        if not bool(torch.isfinite(observation).all()):
            raise RuntimeError("STOP_XPT_SITCOM_NONFINITE_Y")
        shape_only=torch.empty((1,3,256,256),dtype=torch.float32,device=device)
        start=torch.cuda.Event(enable_timing=True)
        finish=torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(device)
        t=time.perf_counter()
        start.record()
        x_start=sampler.get_start(shape_only)
        reconstruction=sampler.sample(
            model,x_start,op,observation,evaluator=None,
            record=False,verbose=False)
        finish.record()
        torch.cuda.synchronize(device)
        wall=time.perf_counter()-t
        cuda_seconds=float(start.elapsed_time(finish))/1000
        validate_reconstruction(reconstruction)
        dest.mkdir(parents=True,exist_ok=False)
        path=dest/"reconstruction.pt"
        png=dest/"reconstruction.png"
        cpu=reconstruction.detach().cpu().float()
        torch.save({"reconstruction":cpu},path)
        save_model_range_png(cpu,png)
        receipt={
            "status":"XP_T_SITCOM_ROOT_PASS",
            "runner_head":a.expected_head,"prep_head":PREP_HEAD,
            "manifest_sha256":MANIFEST_SHA,
            "uid":row["uid"],"image_id":row["image_id"],
            "rep":row["rep"],"seed":row["seed"],
            "dataset":row["dataset"],"family":"SITCOM",
            "measurement_sha256":row["measurement_sha256"],
            "native_head":SITCOM_HEAD,
            "model_sha256":preset["checkpoint_sha256"],
            "model_loaded_strictly":True,
            "strict_load_trace":trace,
            "model_config":mf_cfg,
            "sampler_config":manifest["sampler_config"],
            "gpu_physical":a.gpu,
            "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
            "native_sampling_wall_s":wall,
            "native_sampling_cuda_event_s":cuda_seconds,
            "native_model_loading_wall_s":model_s,
            "peak_gpu_allocated_mib":torch.cuda.max_memory_allocated(0)/2**20,
            "reconstruction_sha256":tensor_content_sha256(cpu),
            "reconstruction_pt_sha256":sha(path),
            "reconstruction_png_sha256":sha(png),
            "gt_free_generation":True,
            "historical_candidate_access":False,
            "evaluation_outcomes_embargo":"NOT_SCORED",
        }
        dump(dest/"XP_T_SITCOM_ROOT_COMPLETE.json",receipt)
        print("XP_T_SITCOM_ROOT_PASS",row["dataset"],row["image_id"],
              "rep",row["rep"],"wall",round(wall,3),
              "cuda",round(cuda_seconds,3),flush=True)
        return 0
    finally:
        os.chdir(native_was)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    manifest,rows=immutable_manifest(a.expected_head)
    row=checked_row(manifest,rows,a.index)
    if a.preflight_only:
        print("XP_T_SITCOM_NATIVE_PREFLIGHT_PASS",a.index,row["dataset"],
              row["image_id"],flush=True)
        return 0
    return generate(a,manifest,row)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_SITCOM_WORKER "+str(exc),flush=True)
        sys.exit(8)
